#pragma once

#include <cstdint>

#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx::modern_poc {

// Initializes the opt-in modern renderer proof of concept.
// The probe is enabled only when AURORA_MODERN_RENDERER_POC is set to a non-zero value.
void initialize() noexcept;

// Releases all WebGPU objects owned by the proof of concept.
void shutdown() noexcept;

// Returns whether the proof of concept was explicitly enabled by the user.
bool enabled() noexcept;

// Draws a small PBR probe into the active Aurora presentation pass.
// This intentionally uses a standalone WebGPU pipeline instead of the GX/TEV shader generator.
void render(const wgpu::RenderPassEncoder& pass, const wgpu::Extent3D& targetSize) noexcept;

} // namespace aurora::gfx::modern_poc
