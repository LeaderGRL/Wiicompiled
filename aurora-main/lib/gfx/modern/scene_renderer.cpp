#include "scene_renderer.hpp"

#include "../common.hpp"
#include "../../gx/pipeline.hpp"
#include "../../webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace aurora::gfx::modern_scene {
namespace {

using webgpu::g_device;
using webgpu::g_graphicsConfig;

static Module Log("aurora::gfx::modern_scene");

// GX triangle draws place the 80-byte scalar/array header first, followed by the 64-byte
// effective projection and then at least one 48-byte post-transform matrix.
constexpr uint32_t kGxCameraBytes = 192;
constexpr uint32_t kMinimumCandidateIndices = 300;
constexpr const char* kRendererBuildId = "scene-poc-r6";

struct Vertex {
  float px;
  float py;
  float pz;
  float nx;
  float ny;
  float nz;
};
static_assert(sizeof(Vertex) == 24);

struct SceneParams {
  // xyz = local-space translation, w = uniform scale.
  std::array<float, 4> translationScale{};
  // x = metallic, y = roughness, z = diagnostic mode, w = exposure.
  std::array<float, 4> material{};
};
static_assert(sizeof(SceneParams) == 32);

constexpr std::array<Vertex, 24> kCubeVertices{{
    {-1.f, -1.f,  1.f,  0.f,  0.f,  1.f}, { 1.f, -1.f,  1.f,  0.f,  0.f,  1.f},
    { 1.f,  1.f,  1.f,  0.f,  0.f,  1.f}, {-1.f,  1.f,  1.f,  0.f,  0.f,  1.f},
    { 1.f, -1.f, -1.f,  0.f,  0.f, -1.f}, {-1.f, -1.f, -1.f,  0.f,  0.f, -1.f},
    {-1.f,  1.f, -1.f,  0.f,  0.f, -1.f}, { 1.f,  1.f, -1.f,  0.f,  0.f, -1.f},
    {-1.f, -1.f, -1.f, -1.f,  0.f,  0.f}, {-1.f, -1.f,  1.f, -1.f,  0.f,  0.f},
    {-1.f,  1.f,  1.f, -1.f,  0.f,  0.f}, {-1.f,  1.f, -1.f, -1.f,  0.f,  0.f},
    { 1.f, -1.f,  1.f,  1.f,  0.f,  0.f}, { 1.f, -1.f, -1.f,  1.f,  0.f,  0.f},
    { 1.f,  1.f, -1.f,  1.f,  0.f,  0.f}, { 1.f,  1.f,  1.f,  1.f,  0.f,  0.f},
    {-1.f,  1.f,  1.f,  0.f,  1.f,  0.f}, { 1.f,  1.f,  1.f,  0.f,  1.f,  0.f},
    { 1.f,  1.f, -1.f,  0.f,  1.f,  0.f}, {-1.f,  1.f, -1.f,  0.f,  1.f,  0.f},
    {-1.f, -1.f, -1.f,  0.f, -1.f,  0.f}, { 1.f, -1.f, -1.f,  0.f, -1.f,  0.f},
    { 1.f, -1.f,  1.f,  0.f, -1.f,  0.f}, {-1.f, -1.f,  1.f,  0.f, -1.f,  0.f},
}};

constexpr std::array<uint16_t, 36> kCubeIndices{{
     0,  1,  2,  0,  2,  3,
     4,  5,  6,  4,  6,  7,
     8,  9, 10,  8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23,
}};

std::mutex g_resourceMutex;
wgpu::ShaderModule g_shaderModule;
wgpu::BindGroupLayout g_sceneLayout;
wgpu::BindGroup g_sceneBindGroup;
wgpu::PipelineLayout g_pipelineLayout;
wgpu::RenderPipeline g_pipeline;
wgpu::Buffer g_vertexBuffer;
wgpu::Buffer g_indexBuffer;
wgpu::Buffer g_sceneParamsBuffer;
std::atomic_bool g_initialized{false};
std::atomic_bool g_loggedActivation{false};
std::atomic_bool g_loggedMsaaRejection{false};
std::atomic_bool g_loggedCandidate{false};
std::atomic_bool g_loggedDraw{false};

bool environment_flag(const char* name) noexcept {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

float environment_float(const char* name, float fallback, float minimum, float maximum) noexcept {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }

  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (errno != 0 || end == value || !std::isfinite(parsed)) {
    return fallback;
  }
  return std::clamp(parsed, minimum, maximum);
}

const bool g_enabled = environment_flag("AURORA_MODERN_SCENE_POC");
const bool g_ignoreDepth = environment_flag("AURORA_MODERN_SCENE_IGNORE_DEPTH");
const bool g_diagnostic = environment_flag("AURORA_MODERN_SCENE_DIAGNOSTIC");

SceneParams build_scene_params() noexcept {
  return {
      .translationScale = {
          environment_float("AURORA_MODERN_SCENE_X", 0.0f, -100000.0f, 100000.0f),
          environment_float("AURORA_MODERN_SCENE_Y", 0.0f, -100000.0f, 100000.0f),
          environment_float("AURORA_MODERN_SCENE_Z", 0.0f, -100000.0f, 100000.0f),
          environment_float("AURORA_MODERN_SCENE_SCALE", 82.0f, 0.01f, 10000.0f),
      },
      .material = {
          environment_float("AURORA_MODERN_SCENE_METALLIC", 0.72f, 0.0f, 1.0f),
          environment_float("AURORA_MODERN_SCENE_ROUGHNESS", 0.22f, 0.04f, 1.0f),
          g_diagnostic ? 1.0f : 0.0f,
          environment_float("AURORA_MODERN_SCENE_EXPOSURE", 1.0f, 0.1f, 8.0f),
      },
  };
}

template <typename T, size_t N>
wgpu::Buffer create_static_buffer(const std::array<T, N>& source, wgpu::BufferUsage usage,
                                  const char* label) {
  const wgpu::BufferDescriptor descriptor{
      .label = label,
      .usage = usage,
      .size = sizeof(source),
      .mappedAtCreation = true,
  };
  auto buffer = g_device.CreateBuffer(&descriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(source)), source.data(), sizeof(source));
  buffer.Unmap();
  return buffer;
}

wgpu::Buffer create_scene_params_buffer() {
  const SceneParams params = build_scene_params();
  const wgpu::BufferDescriptor descriptor{
      .label = "MKart Modern Scene Parameters",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(SceneParams),
      .mappedAtCreation = true,
  };
  auto buffer = g_device.CreateBuffer(&descriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(params)), &params, sizeof(params));
  buffer.Unmap();
  return buffer;
}

void create_geometry() {
  g_vertexBuffer = create_static_buffer(kCubeVertices, wgpu::BufferUsage::Vertex,
                                        "MKart Modern Scene Cube Vertices");
  g_indexBuffer = create_static_buffer(kCubeIndices, wgpu::BufferUsage::Index,
                                       "MKart Modern Scene Cube Indices");
  g_sceneParamsBuffer = create_scene_params_buffer();
}

void create_pipeline() {
  wgpu::ShaderSourceWGSL shaderSource{};
  shaderSource.code = R"""(
const PI: f32 = 3.14159265359;

struct GxCameraUniform {
    header: array<vec4<u32>, 5>,
    projection: mat4x4<f32>,
    modelView: mat3x4<f32>,
};

struct SceneParams {
    translationScale: vec4<f32>,
    material: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instanceIndex: u32,
};

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) viewPosition: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tint: vec3<f32>,
};

@group(0) @binding(0) var<uniform> gxCamera: GxCameraUniform;
@group(0) @binding(1) var<uniform> scene: SceneParams;

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let nDotH = max(dot(n, h), 0.0);
    let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 0.0001);
}

fn geometry_schlick_ggx(nDotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) * 0.125;
    return nDotV / max(nDotV * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    return geometry_schlick_ggx(max(dot(n, v), 0.0), roughness) *
           geometry_schlick_ggx(max(dot(n, l), 0.0), roughness);
}

fn fresnel_schlick(cosTheta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + vec3<f32>(b))) /
                 (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

fn diagnostic_tint(index: u32) -> vec3<f32> {
    if (index == 0u) { return vec3<f32>(1.0, 0.20, 0.02); }
    return vec3<f32>(0.02, 0.78, 1.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let diagnosticMode = scene.material.z > 0.5;
    let cameraSpaceControl = diagnosticMode && input.instanceIndex == 1u;
    let scale = scene.translationScale.w;
    let localPosition = input.position * scale + scene.translationScale.xyz;

    // Match Aurora's normal GX transform path: local-space position first enters the currently
    // selected post-transform matrix, then the effective GX projection.
    var viewPosition = vec4<f32>(localPosition, 1.0) * gxCamera.modelView;
    var viewNormal = vec4<f32>(input.normal, 0.0) * gxCamera.modelView;

    // The second diagnostic instance deliberately bypasses model-view. It remains a known-good
    // camera-space control while instance 0 validates the draw-local transform bridge.
    if (cameraSpaceControl) {
        viewPosition = input.position * scale + vec3<f32>(-220.0, 0.0, -650.0);
        viewNormal = input.normal;
    }

    output.viewPosition = viewPosition;
    output.normal = normalize(viewNormal);
    output.tint = select(vec3<f32>(0.96, 0.23, 0.035),
                         diagnostic_tint(input.instanceIndex), diagnosticMode);

    var clipPosition = vec4<f32>(viewPosition, 1.0) * gxCamera.projection;
    // Aurora uses reversed Z and negates clip-space Z after projection.
    clipPosition.z = -clipPosition.z;
    output.clipPosition = clipPosition;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let v = normalize(-input.viewPosition);
    let l = normalize(vec3<f32>(-0.42, 0.72, 0.55));
    let h = normalize(v + l);

    let baseColor = input.tint;
    let metallic = scene.material.x;
    let roughness = scene.material.y;
    let f0 = vec3<f32>(0.04) * (1.0 - metallic) + baseColor * metallic;

    let nDotL = max(dot(n, l), 0.0);
    let nDotV = max(dot(n, v), 0.0);
    let fresnel = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let ndf = distribution_ggx(n, h, roughness);
    let geometry = geometry_smith(n, v, l, roughness);
    let specular = (ndf * geometry * fresnel) / max(4.0 * nDotV * nDotL, 0.0001);
    let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * baseColor / PI;

    let direct = (diffuse + specular) * vec3<f32>(5.5, 4.8, 4.2) * nDotL;
    let ambient = baseColor * 0.07 + f0 * 0.09;
    let rim = pow(1.0 - nDotV, 3.0) * baseColor * 0.32;
    let mapped = aces_tonemap((ambient + direct + rim) * scene.material.w);
    let displayColor = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(displayColor, 1.0);
}
)""";

  const wgpu::ShaderModuleDescriptor shaderDescriptor{
      .nextInChain = &shaderSource,
      .label = "MKart Modern Scene PBR Shader",
  };
  g_shaderModule = g_device.CreateShaderModule(&shaderDescriptor);

  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{
              .type = wgpu::BufferBindingType::Uniform,
              .hasDynamicOffset = true,
          },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{
              .type = wgpu::BufferBindingType::Uniform,
              .hasDynamicOffset = false,
              .minBindingSize = sizeof(SceneParams),
          },
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "MKart Modern Scene Layout",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  g_sceneLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);

  const std::array bindEntries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .buffer = g_uniformBuffer,
          .size = gx::MaxUniformSize,
      },
      wgpu::BindGroupEntry{
          .binding = 1,
          .buffer = g_sceneParamsBuffer,
          .size = sizeof(SceneParams),
      },
  };
  const wgpu::BindGroupDescriptor bindDescriptor{
      .label = "MKart Modern Scene Bind Group",
      .layout = g_sceneLayout,
      .entryCount = bindEntries.size(),
      .entries = bindEntries.data(),
  };
  g_sceneBindGroup = g_device.CreateBindGroup(&bindDescriptor);

  const std::array layouts{g_sceneLayout};
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "MKart Modern Scene Pipeline Layout",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  g_pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);

  const std::array vertexAttributes{
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x3, .offset = 12, .shaderLocation = 1},
  };
  const std::array vertexLayouts{
      wgpu::VertexBufferLayout{
          .arrayStride = sizeof(Vertex),
          .stepMode = wgpu::VertexStepMode::Vertex,
          .attributeCount = vertexAttributes.size(),
          .attributes = vertexAttributes.data(),
      },
  };

  const wgpu::ColorTargetState colorTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .writeMask = wgpu::ColorWriteMask::All,
  };
  const wgpu::FragmentState fragmentState{
      .module = g_shaderModule,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &colorTarget,
  };
  const wgpu::DepthStencilState depthState{
      .format = g_graphicsConfig.depthFormat,
      .depthWriteEnabled = false,
      .depthCompare = g_ignoreDepth ? wgpu::CompareFunction::Always : wgpu::CompareFunction::GreaterEqual,
  };
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = "MKart Modern Scene PBR Pipeline",
      .layout = g_pipelineLayout,
      .vertex = wgpu::VertexState{
          .module = g_shaderModule,
          .entryPoint = "vs_main",
          .bufferCount = vertexLayouts.size(),
          .buffers = vertexLayouts.data(),
      },
      .primitive = wgpu::PrimitiveState{
          .topology = wgpu::PrimitiveTopology::TriangleList,
          .frontFace = wgpu::FrontFace::CCW,
          .cullMode = wgpu::CullMode::None,
      },
      .depthStencil = &depthState,
      .multisample = wgpu::MultisampleState{.count = 1},
      .fragment = &fragmentState,
  };
  g_pipeline = g_device.CreateRenderPipeline(&pipelineDescriptor);
}

void initialize() noexcept {
  if (!g_enabled || g_initialized.load(std::memory_order_acquire)) {
    return;
  }

  std::lock_guard lock{g_resourceMutex};
  if (g_initialized.load(std::memory_order_relaxed)) {
    return;
  }

  create_geometry();
  create_pipeline();
  g_initialized.store(true, std::memory_order_release);
}

bool candidate_draw(const gx::DrawData& draw, const Range& uniformRange) noexcept {
  if (g_graphicsConfig.msaaSamples != 1) {
    if (!g_loggedMsaaRejection.exchange(true, std::memory_order_relaxed)) {
      Log.warn("POC rejected scene draws because MSAA sample count is {} (expected 1)",
               g_graphicsConfig.msaaSamples);
    }
    return false;
  }

  // currentPnMtx == 0 guarantees that the first uploaded post-transform matrix is the draw's active
  // model-view matrix in both Aurora's absolute and compact matrix layouts.
  return draw.scene.projectionType == GX_PERSPECTIVE && draw.scene.currentPnMtx == 0 &&
         draw.instanceCount == 1 && draw.indexCount >= kMinimumCandidateIndices &&
         uniformRange.size >= kGxCameraBytes;
}

} // namespace

bool enabled() noexcept { return g_enabled; }

void render_after_gx_draw(const gx::DrawData& draw, const Range& effectiveUniformRange,
                          const wgpu::RenderPassEncoder& pass, gx::DrawEncodeState& state) noexcept {
  if (!g_enabled || state.modernSceneDrawn) {
    return;
  }

  if (!g_loggedActivation.exchange(true, std::memory_order_relaxed)) {
    const SceneParams params = build_scene_params();
    Log.info("POC enabled: build={} msaaSamples={} ignoreDepth={} diagnostic={} reversedZ=true modelView=true position=({}, {}, {}) scale={} metallic={} roughness={}",
             kRendererBuildId, g_graphicsConfig.msaaSamples, g_ignoreDepth ? "true" : "false",
             g_diagnostic ? "true" : "false", params.translationScale[0], params.translationScale[1],
             params.translationScale[2], params.translationScale[3], params.material[0], params.material[1]);
  }

  if (!candidate_draw(draw, effectiveUniformRange)) {
    return;
  }

  if (!g_loggedCandidate.exchange(true, std::memory_order_relaxed)) {
    Log.info("Selected GX model-view anchor: indices={} instances={} uniformOffset={} uniformSize={} projectionType={} currentPnMtx={} projectionOffset=80 modelViewOffset=144",
             draw.indexCount, draw.instanceCount, effectiveUniformRange.offset, effectiveUniformRange.size,
             static_cast<uint32_t>(draw.scene.projectionType), draw.scene.currentPnMtx);
  }

  initialize();
  if (!g_initialized.load(std::memory_order_acquire)) {
    return;
  }

  const std::array dynamicOffsets{effectiveUniformRange.offset};
  pass.SetPipeline(g_pipeline);
  pass.SetBindGroup(0, g_sceneBindGroup, dynamicOffsets.size(), dynamicOffsets.data());
  pass.SetVertexBuffer(0, g_vertexBuffer, 0, sizeof(kCubeVertices));
  pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16, 0, sizeof(kCubeIndices));
  const uint32_t instanceCount = g_diagnostic ? 2u : 1u;
  pass.DrawIndexed(static_cast<uint32_t>(kCubeIndices.size()), instanceCount);

  if (!g_loggedDraw.exchange(true, std::memory_order_relaxed)) {
    Log.info("Issued modern scene DrawIndexed: build={} indices={} instances={} cameraUniformOffset={} modelView=true",
             kRendererBuildId, kCubeIndices.size(), instanceCount, effectiveUniformRange.offset);
  }

  state.modernSceneDrawn = true;
  state.currentPipeline = UINTPTR_MAX;
  state.boundTextureBindGroup = nullptr;
  state.indexBufferBound = false;
}

void shutdown() noexcept {
  std::lock_guard lock{g_resourceMutex};
  g_initialized.store(false, std::memory_order_release);
  g_sceneParamsBuffer = {};
  g_indexBuffer = {};
  g_vertexBuffer = {};
  g_pipeline = {};
  g_pipelineLayout = {};
  g_sceneBindGroup = {};
  g_sceneLayout = {};
  g_shaderModule = {};
}

} // namespace aurora::gfx::modern_scene
