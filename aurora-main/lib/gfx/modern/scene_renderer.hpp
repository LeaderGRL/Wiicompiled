#pragma once

#include <cstdint>

#include <webgpu/webgpu_cpp.h>

namespace aurora::gx {
struct DrawData;
struct DrawEncodeState;
}

namespace aurora::gfx {
struct Range;

namespace modern_scene {

// The scene path stays opt-in until the camera bridge is validated on real gameplay captures.
bool enabled() noexcept;

// Observes one already-sealed GX draw. The effective uniform range is the exact range used by this
// presentation slot, so interpolated frames automatically select their interpolated camera state.
void observe_gx_draw(const gx::DrawData& draw, const Range& effectiveUniformRange,
                     gx::DrawEncodeState& state) noexcept;

// Draws the modern scene probe into the same color/depth attachments as the GX pass.
void flush_gx_pass(const wgpu::RenderPassEncoder& pass, const wgpu::Extent3D& targetSize,
                   uint32_t msaaSamples, gx::DrawEncodeState& state) noexcept;

// Releases GPU objects owned by the scene renderer.
void shutdown() noexcept;

} // namespace modern_scene
} // namespace aurora::gfx
