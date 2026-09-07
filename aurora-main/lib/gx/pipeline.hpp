#pragma once

#include "../gfx/common.hpp"
#include "shader_info.hpp"

namespace aurora::gx {

// Immutable metadata captured while a GX draw is recorded. The frame worker may encode the draw
// after the producer has already started the next frame, so modern rendering code must consume this
// sealed snapshot instead of reading mutable g_gxState during replay.
struct SceneDrawMetadata {
  GXProjectionType projectionType = GX_ORTHOGRAPHIC;
  uint32_t currentPnMtx = 0;
};

// Evaluated on the producer thread by DrawData's default member initializer. Keeping the capture in
// the DrawData construction path means every producer (FIFO decode and raw bridge draws) gets the
// same sealed scene metadata without duplicating call-site logic.
SceneDrawMetadata capture_scene_draw_metadata() noexcept;

struct DrawData {
  gfx::PipelineRef pipeline;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range uniformRange;
  std::array<gfx::Range, MaxInterpolatedFrames> interpolatedUniformRanges;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  GXBindGroups bindGroups;
  uint32_t dstAlpha;
  SceneDrawMetadata scene = capture_scene_draw_metadata();
};

constexpr uint32_t GXPipelineConfigVersion = 19;

constexpr GXFogType effective_pipeline_fog_type(GXFogType fogType, GXZTexOp zTextureOp,
                                                bool zCompLocBeforeTex, GXBlendMode blendMode,
                                                GXLogicOp logicOp) noexcept {
  const bool usesLateZTexture = zTextureOp != GX_ZT_DISABLE && !zCompLocBeforeTex;
  const bool usesUnsupportedLogicFog =
      blendMode == GX_BM_LOGIC && logicOp != GX_LO_OR;
  return usesLateZTexture && usesUnsupportedLogicFog ? GX_FOG_NONE : fogType;
}

struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1;
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  GXPixelFmt pixelFmt;
  uint32_t dstAlpha;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

inline bool valid_pipeline_config(const PipelineConfig& config) noexcept {
  const auto in_range = [](auto value, auto maximum) {
    using Value = decltype(value);
    return static_cast<std::underlying_type_t<Value>>(value) >= 0 &&
           static_cast<std::underlying_type_t<Value>>(value) <=
               static_cast<std::underlying_type_t<Value>>(maximum);
  };
  const bool validSamples = config.msaaSamples == 1 || config.msaaSamples == 2 ||
                            config.msaaSamples == 4 || config.msaaSamples == 8;
  return config.version == GXPipelineConfigVersion && validSamples &&
         in_range(config.depthFunc, GX_ALWAYS) && in_range(config.cullMode, GX_CULL_ALL) &&
         in_range(config.blendMode, GX_BM_SUBTRACT) &&
         in_range(config.blendFacSrc, GX_BL_INVDSTALPHA) &&
         in_range(config.blendFacDst, GX_BL_INVDSTALPHA) &&
         in_range(config.blendOp, GX_LO_SET) && in_range(config.pixelFmt, GX_PF_YUV420);
}

wgpu::RenderPipeline create_pipeline([[maybe_unused]] const PipelineConfig& config);
void clear_shader_module_cache();

// Per-pass encoder state carried across the draws of one render pass, so the replay loop can elide Dawn calls that would re-bind what is already bound.
struct DrawEncodeState {
  gfx::PipelineRef currentPipeline = UINTPTR_MAX;
  // The bind group currently occupying slot 2.
  WGPUBindGroup boundTextureBindGroup = nullptr;
  // The pass-wide index buffer binding is established lazily by the first indexed draw; every draw then addresses its range with firstIndex instead of a per-draw SetIndexBuffer.
  bool indexBufferBound = false;
  // The modern scene POC injects at most one probe per GX render pass. Keeping this in the existing
  // per-pass state makes the gate thread-local to the encoder and requires no shared synchronization.
  bool modernSceneDrawn = false;
};

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass, DrawEncodeState& state,
            bool requireReadyPipeline, const gfx::Range* uniformRangeOverride = nullptr);

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
