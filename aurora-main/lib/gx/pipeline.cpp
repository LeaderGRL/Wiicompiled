#include "pipeline.hpp"

#include "../gfx/modern/scene_renderer.hpp"
#include "../webgpu/gpu.hpp"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>

#include <absl/container/flat_hash_map.h>

namespace aurora::gx {
static Module Log("aurora::gx");

SceneDrawMetadata capture_scene_draw_metadata() noexcept {
  // This runs while the producer owns the renderer GPU mutex and constructs the sealed DrawData.
  // The asynchronous frame worker only consumes the copied fields later and never touches g_gxState.
  return {
      .projectionType = g_gxState.projType,
      .currentPnMtx = g_gxState.currentPnMtx,
  };
}

namespace {
struct ShaderConfigHash {
  size_t operator()(const ShaderConfig& config) const noexcept { return static_cast<size_t>(xxh3_hash(config)); }
};

struct CachedShaderModule {
  std::condition_variable ready;
  wgpu::ShaderModule module;
  bool compiling = true;
};

std::mutex sShaderModuleCacheMutex;
absl::flat_hash_map<ShaderConfig, std::shared_ptr<CachedShaderModule>, ShaderConfigHash> sShaderModuleCache;

wgpu::ShaderModule cached_shader_module(const ShaderConfig& config) {
  std::shared_ptr<CachedShaderModule> entry;
  {
    std::unique_lock lock{sShaderModuleCacheMutex};
    const auto it = sShaderModuleCache.find(config);
    if (it == sShaderModuleCache.end()) {
      entry = std::make_shared<CachedShaderModule>();
      sShaderModuleCache.emplace(config, entry);
    } else {
      entry = it->second;
      entry->ready.wait(lock, [&] { return !entry->compiling; });
      return entry->module;
    }
  }

  auto module = build_shader(config);
  {
    std::lock_guard lock{sShaderModuleCacheMutex};
    entry->module = module;
    entry->compiling = false;
  }
  entry->ready.notify_all();
  return module;
}
} // namespace

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = cached_shader_module(config.shaderConfig);
  return build_pipeline(config, {}, shader, "GX Pipeline");
}

void clear_shader_module_cache() {
  gfx::modern_scene::shutdown();
  std::lock_guard lock{sShaderModuleCacheMutex};
  sShaderModuleCache.clear();
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass, DrawEncodeState& state,
            bool requireReadyPipeline, const gfx::Range* uniformRangeOverride) {
  if (!gfx::bind_pipeline(data.pipeline, pass, state.currentPipeline, requireReadyPipeline)) {
    return;
  }

  // An interpolated presentation slot re-encodes the identical draw with only this range replaced; overriding here avoids copying the whole DrawData per draw per slot.
  const gfx::Range& uniformRange = uniformRangeOverride != nullptr ? *uniformRangeOverride : data.uniformRange;
  const std::array offsets{uniformRange.offset};
  pass.SetBindGroup(1, gfx::g_uniformBindGroup, offsets.size(), offsets.data());
  // Resolved when the draw was recorded; see GXBindGroups.
  if (data.bindGroups.resolvedTextureBindGroup != nullptr &&
      data.bindGroups.resolvedTextureBindGroup != state.boundTextureBindGroup) {
    wgpuRenderPassEncoderSetBindGroup(pass.Get(), 2, data.bindGroups.resolvedTextureBindGroup, 0, nullptr);
    state.boundTextureBindGroup = data.bindGroups.resolvedTextureBindGroup;
  }
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  if (!state.indexBufferBound) {
    // Bound once for the pass; draws select their range with firstIndex below.
    pass.SetIndexBuffer(gfx::g_indexBuffer, wgpu::IndexFormat::Uint16, 0, wgpu::kWholeSize);
    state.indexBufferBound = true;
  }
  pass.DrawIndexed(data.indexCount, data.instanceCount,
                   static_cast<uint32_t>(data.idxRange.offset / sizeof(uint16_t)));

  // The modern scene path runs immediately after a suitable sealed GX draw. At this point the
  // render pass still has that draw's viewport/scissor/depth attachments, which is the safest
  // possible place to validate a parallel 3D renderer without reading mutable producer state.
  const bool modernSceneWasDrawn = state.modernSceneDrawn;
  gfx::modern_scene::render_after_gx_draw(data, uniformRange, pass, state);

  if (!modernSceneWasDrawn && state.modernSceneDrawn) {
    // The POC uses a different pipeline layout and temporarily occupies bind-group slot 0.
    // Aurora normally binds the GX static group only once at render-pass start, so failing to
    // restore it leaves the next GX draw with an incompatible bind group and can fault inside Dawn.
    // Re-establish the complete GX binding baseline once after the injected draw. This is not a
    // per-draw cost: the modern scene probe can execute at most once per render pass.
    pass.SetBindGroup(0, gfx::g_staticBindGroup);
    pass.SetBindGroup(1, gfx::g_uniformBindGroup, offsets.size(), offsets.data());
    pass.SetBindGroup(2, g_emptyTextureBindGroup);
    state.boundTextureBindGroup = g_emptyTextureBindGroup.Get();
  }
}
} // namespace aurora::gx
