#pragma once

#include <webgpu/webgpu_cpp.h>

namespace aurora::gx {
struct DrawData;
struct DrawEncodeState;
}

namespace aurora::gfx {
struct Range;

namespace modern_scene {

// The scene renderer is intentionally opt-in while the camera bridge is being validated.
bool enabled() noexcept;

// Observe one already-sealed GX draw. The effective uniform range is the range actually used by
// this presentation slot, so interpolated frames automatically feed the matching GX camera data.
void observe_gx_draw(const gx::DrawData& draw, const Range& effectiveUniformRange,
                     const wgpu::RenderPassEncoder& pass, gx::DrawEncodeState& state) noexcept;

// Flushes one modern scene draw before the owning GX render pass is closed.
void flush_gx_pass(gx::DrawEncodeState& state) noexcept;

// Releases all GPU resources. Lazy initialization allows a later renderer restart to recreate them.
void shutdown() noexcept;

} // namespace modern_scene
} // namespace aurora::gfx
