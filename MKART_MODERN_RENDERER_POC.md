# MKart modern renderer proof of concept

This branch adds an opt-in standalone WebGPU/WGSL renderer path next to Aurora's existing GX/TEV rendering.

## Enable

Set the environment variable before launching WiiCompiled:

```powershell
$env:AURORA_MODERN_RENDERER_POC="1"
```

When enabled, the presentation snapshot pass draws a metallic blue PBR probe in the upper-right corner before ImGui.

Without the environment variable, the POC does not create the custom pipeline and does not alter frame output.

## What the shader demonstrates

The custom WGSL shader uses:

- GGX normal distribution
- Smith geometry term
- Fresnel-Schlick
- metallic/roughness shading
- procedural fine normal detail
- HDR-bright emissive energy
- ACES-style tone mapping
- alpha blending

The shader is not generated from GX/TEV state. It is a separate `wgpu::RenderPipeline` created directly through the same Dawn/WebGPU API Aurora already uses.

## Why this matters

A visible probe over a running Mario Kart Wii frame proves that a non-GX modern renderer can coexist with WiiCompiled's native rendering path. This first step is intentionally presentation-only and does not replace game geometry yet.

The next steps are:

1. move the custom path into the scene/EFB render stage;
2. add host-side vertex/index buffers and model/view/projection uniforms;
3. synchronize the modern camera with Mario Kart Wii;
4. load a static glTF mesh;
5. add base-color, normal, ORM and emissive textures;
6. bridge a known Mario Kart object transform to the host renderer;
7. add directional lighting and shadow maps;
8. add HDR intermediate targets and bloom/color grading;
9. add GPU particles for drift sparks, boosts, smoke, dust and splashes;
10. build a local-only asset conversion pipeline without distributing Nintendo assets.

## Current validation

The new renderer module was independently checked with Clang 17 using C++20, `-Wall -Wextra -Werror`, against a minimal WebGPU API stub matching the interfaces it uses.

A full WiiCompiled build still needs to be run on a machine with the project's normal Dawn/toolchain dependencies. The decisive runtime test is simple: build this branch and launch with `AURORA_MODERN_RENDERER_POC=1`; the PBR probe should appear over the game image.
