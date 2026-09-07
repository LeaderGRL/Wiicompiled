#include "pipeline.hpp"

#include "../gfx/modern/scene_renderer.hpp"
#include "../webgpu/gpu.hpp"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

namespace aurora::gx {
static Module Log("aurora::gx");

namespace {
SceneAttrSource capture_attr_source(GXAttr attr) noexcept {
  const auto& source = g_gxState.arrays[static_cast<size_t>(attr)];
  return {
      .address = reinterpret_cast<uintptr_t>(source.data),
      .size = source.size,
      .stride = source.stride,
  };
}

bool environment_flag(const char* name) noexcept {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

const bool sInspectSceneDraws = environment_flag("AURORA_MODERN_SCENE_INSPECT");

struct SceneDrawInspectKey {
  uintptr_t positionAddress = 0;
  uintptr_t normalAddress = 0;
  uintptr_t tex0Address = 0;
  uintptr_t pipeline = 0;
  uint32_t vertexCount = 0;
  uint32_t indexCount = 0;
  uint32_t positionStride = 0;

  bool operator==(const SceneDrawInspectKey&) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const SceneDrawInspectKey& key) {
    return H::combine(std::move(h), key.positionAddress, key.normalAddress, key.tex0Address,
                      key.pipeline, key.vertexCount, key.indexCount, key.positionStride);
  }
};

std::mutex sSceneInspectMutex;
absl::flat_hash_set<SceneDrawInspectKey> sSeenSceneDraws;
constexpr size_t kMaximumLoggedSceneDraws = 256;

void inspect_scene_draw(const DrawData& data) {
  if (!sInspectSceneDraws || data.scene.projectionType != GX_PERSPECTIVE) {
    return;
  }

  const SceneDrawInspectKey key{
      .positionAddress = data.scene.positionSource.address,
      .normalAddress = data.scene.normalSource.address,
      .tex0Address = data.scene.tex0Source.address,
      .pipeline = static_cast<uintptr_t>(data.pipeline),
      .vertexCount = data.vtxCount,
      .indexCount = data.indexCount,
      .positionStride = data.scene.positionSource.stride,
  };

  std::lock_guard lock{sSceneInspectMutex};
  if (sSeenSceneDraws.size() >= kMaximumLoggedSceneDraws || !sSeenSceneDraws.insert(key).second) {
    return;
  }

  Log.info(
      "MKART_DRAW id={} pos=0x{:x} posSize={} posStride={} nrm=0x{:x} nrmSize={} nrmStride={} "
      "tex0=0x{:x} tex0Size={} tex0Stride={} vtx={} idx={} pipeline=0x{:x} pnMtx={}",
      sSeenSceneDraws.size() - 1, data.scene.positionSource.address, data.scene.positionSource.size,
      data.scene.positionSource.stride, data.scene.normalSource.address, data.scene.normalSource.size,
      data.scene.normalSource.stride, data.scene.tex0Source.address, data.scene.tex0Source.size,
      data.scene.tex0Source.stride, data.vtxCount, data.indexCount,
      static_cast<uintptr_t>(data.pipeline), data.scene.currentPnMtx);
}
} // namespace

SceneDrawMetadata capture_scene_draw_metadata() noexcept {
  // This runs while the producer owns the renderer GPU mutex and constructs the sealed DrawData.
  // The asynchronous frame worker only consumes the copied fields later and never touches g_gxState.
  return {
      .projectionType = g_gxState.projType,
      .currentPnMtx = g_gxState.currentPnMtx,
      .positionSource = capture_attr_source(GX_VA_POS),
      .normalSource = capture_attr_source(GX_VA_NRM),
      .tex0Source = capture_attr_source(GX_VA_TEX0),
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
  return entry->module;
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

  inspect_scene_draw(data);

  // Inspection mode is intentionally non-invasive: render the original GX stream and only log
  // stable resource identities. This lets us classify menu/course/character draws before enabling
  // a replacement on a specific mesh.
  if (sInspectSceneDraws) {
    return;
  }

  // Only perspective draws can provide a usable scene camera. This metadata was captured on the
  // producer thread together with the draw, so the async frame worker never reads mutable GX state.
  const bool modernSceneWasDrawn = state.modernSceneDrawn;
  if (data.scene.projectionType == GX_PERSPECTIVE) {
    gfx::modern_scene::render_after_gx_draw(data, uniformRange, pass, state);
  }

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
