# MKart modern scene renderer — POC 2

POC 1 proved that a standalone WebGPU/WGSL pipeline can coexist with WiiCompiled. POC 2 moves the experiment into the GX scene itself: a real indexed 3D mesh is rendered while a GX render pass is active and reuses the exact sealed GX projection-uniform range of the draw it follows.

## Architecture

The scene renderer lives under `aurora-main/lib/gfx/modern/` and is compiled into `aurora_gx` rather than `aurora_core`.

The important constraint is the asynchronous frame worker. The producer can already be recording the next Mario Kart Wii frame while a previous sealed frame is encoded. The scene renderer therefore does **not** read mutable `gx::g_gxState` during encode.

Instead it hooks immediately after an already-sealed GX draw in `gx::render()` and binds the same `gfx::g_uniformBuffer` range that draw used. The GX triangle uniform prefix is stable:

- bytes `0..79`: GX draw header / viewport / array ranges;
- bytes `80..143`: effective GX projection matrix.

The custom WGSL shader maps that prefix and uses the same row-vector projection convention as Aurora's generated GX shader.

This has two useful consequences:

1. the modern draw consumes immutable data belonging to the sealed frame;
2. when frame interpolation replays a draw with an interpolated uniform range, the modern draw receives the matching interpolated range automatically.

The current POC renders camera-space geometry. A world/model transform bridge comes next.

## GPU resources

The renderer owns persistent resources created lazily on first use:

- one immutable vertex buffer;
- one immutable index buffer;
- one GX-camera bind-group layout;
- one bind group referencing Aurora's existing shared uniform buffer;
- one PBR WGSL shader module;
- one pipeline layout;
- one depth-aware render pipeline.

Static geometry uses buffers mapped at creation, avoiding per-frame uploads and avoiding a queue write from the encode path.

After the custom draw, only the `DrawEncodeState` bindings that were invalidated by the custom pipeline are reset. The next GX draw restores its own pipeline/index/texture bindings normally.

## Enable

For this first scene validation, disable the old presentation probe and enable the scene POC:

```powershell
Remove-Item Env:AURORA_MODERN_RENDERER_POC -ErrorAction SilentlyContinue
$env:AURORA_MODERN_SCENE_POC="1"
```

The initial implementation intentionally requires **MSAA = 1/off**. The current sealed `DrawData` does not yet carry the render pass sample count, and WebGPU requires the pipeline sample count to exactly match the active attachments. With MSAA disabled, both main and offscreen targets are single-sampled, so the experiment is deterministic.

## Test

1. Build this branch using the same procedure that validated POC 1.
2. Launch WiiCompiled with `AURORA_MODERN_SCENE_POC=1`.
3. Set MSAA to 1/off if necessary.
4. Enter a normal race where a substantial 3D scene is being rendered.
5. Look for an orange/red metallic cube rendered by the modern PBR pipeline.

Unlike POC 1, this cube is not a presentation overlay. It is submitted inside a GX render pass and shares that pass's viewport, scissor and depth attachment.

### Depth troubleshooting

Depth testing is enabled by default and depth writes are disabled, so the POC can be occluded by existing Mario Kart Wii geometry without modifying the Wii depth buffer.

If no cube is visible, first test the exact same build with depth comparison disabled:

```powershell
$env:AURORA_MODERN_SCENE_IGNORE_DEPTH="1"
```

Restart the process after changing the variable because renderer feature flags are sampled at process startup.

If the cube appears only with `AURORA_MODERN_SCENE_IGNORE_DEPTH=1`, the WebGPU scene path and GX camera bridge are working and only the depth convention needs adjustment.

## Candidate draw selection

Until perspective/scene metadata is explicitly carried in sealed `DrawData`, POC 2 uses an allocation-free heuristic: the first non-instanced GX draw in a pass with at least 300 indices and a sufficiently large uniform range is used as the camera source. This intentionally keeps the change small and avoids touching the large FIFO decoder during the initial validation.

The next architecture step is to carry explicit immutable scene metadata in the sealed draw command so selection no longer relies on a heuristic.

## Next milestones

1. validate the in-pass indexed mesh on real gameplay;
2. carry explicit perspective / pass sample-count metadata in sealed draw commands;
3. expose a stable camera snapshot instead of selecting a representative draw;
4. add a host-side model transform / scene object bridge;
5. anchor a modern mesh in Mario Kart Wii world space;
6. load glTF/GLB meshes instead of compiled test geometry;
7. add base-color, normal, ORM and emissive textures;
8. add material and texture caches keyed by asset identity;
9. add directional lighting and shadow maps;
10. add an HDR scene target, bloom and color grading;
11. add GPU particle systems and event bridges for drift/boost/smoke/splash VFX;
12. add local-only MK8/MK8DX asset conversion without distributing Nintendo assets.
