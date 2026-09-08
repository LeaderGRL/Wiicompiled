#pragma once

#include <webgpu/webgpu_cpp.h>

namespace aurora::gx {
struct DrawData;
struct DrawEncodeState;
}

namespace aurora::gfx {
struct Range;

namespace modern_scene {

// The scene path stays opt-in until the camera/depth bridge is validated on real gameplay.
bool enabled() noexcept;

// Called immediately after a GX draw while its viewport, scissor, color attachment, depth attachment
// and sealed camera-uniform range are still active. This avoids reading producer-visible GX state
// while the asynchronous frame worker encodes a previous frame.
void render_after_gx_draw(const gx::DrawData& draw, const Range& effectiveUniformRange,
                          const wgpu::RenderPassEncoder& pass, gx::DrawEncodeState& state) noexcept;

// Releases GPU objects owned by the scene renderer.
void shutdown() noexcept;

} // namespace modern_scene
} // namespace aurora::gfx
