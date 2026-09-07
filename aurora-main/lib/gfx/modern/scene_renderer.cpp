#include "scene_renderer.hpp"

#include "../common.hpp"
#include "../../gx/pipeline.hpp"
#include "../../webgpu/gpu.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace aurora::gfx::modern_scene {
namespace {

using webgpu::g_device;
using webgpu::g_graphicsConfig;

static Module Log("aurora::gfx::modern_scene");

constexpr uint32_t kGxCameraBytes = 144;
constexpr uint32_t kMinimumCandidateIndices = 300;

struct Vertex {
  float px;
  float py;
  float pz;
  float nx;
  float ny;
  float nz;
};
static_assert(sizeof(Vertex) == 24);

// Twenty-four vertices keep each cube face flat shaded and are also the shape we will later replace
// with imported glTF vertex streams. The geometry is uploaded once and never rewritten per frame.
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
wgpu::BindGroupLayout g_cameraLayout;
wgpu::BindGroup g_cameraBindGroup;
wgpu::PipelineLayout g_pipelineLayout;
wgpu::RenderPipeline g_pipeline;
wgpu::Buffer g_vertexBuffer;
wgpu::Buffer g_indexBuffer;
std::atomic_bool g_initialized{false};
std::atomic_bool g_loggedActivation{false};
std::atomic_bool g_loggedMsaaRejection{false};
std::atomic_bool g_loggedCandidate{false};
std::atomic_bool g_loggedDraw{false};

bool environment_flag(const char* name) noexcept {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

const bool g_enabled = environment_flag("AURORA_MODERN_SCENE_POC");
const bool g_ignoreDepth = environment_flag("AURORA_MODERN_SCENE_IGNORE_DEPTH");

template <typename T, size_t N>
wgpu::Buffer create_static_buffer(const std::array<T, N>& source, wgpu::BufferUsage usage, const char* label) {
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

void create_geometry() {
  // Mapped-at-creation avoids a queue write while a render pass is being encoded. These immutable
  // buffers are initialized once, then shared by every frame without CPU/GPU synchronization.
  g_vertexBuffer = create_static_buffer(kCubeVertices, wgpu::BufferUsage::Vertex,
                                        "MKart Modern Scene Cube Vertices");
  g_indexBuffer = create_static_buffer(kCubeIndices, wgpu::BufferUsage::Index,
                                       "MKart Modern Scene Cube Indices");
}

void create_pipeline() {
  wgpu::ShaderSourceWGSL shaderSource{};
  shaderSource.code = R"""(
const PI: f32 = 3.14159265359;

// This prefix exactly mirrors the stable triangle-draw prefix produced by gx::build_uniform.
// Five vec4 values consume the first 80 bytes; the GX projection matrix begins at byte 80.
struct GxCameraUniform {
    header: array<vec4<u32>, 5>,
    projection: mat4x4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) viewPosition: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@group(0) @binding(0) var<uniform> gxCamera: GxCameraUniform;

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

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // The first scene milestone is deliberately camera-space geometry. It proves that an arbitrary
    // indexed 3D mesh can share MKW's projection, viewport and depth attachment. World transforms
    // are the next bridge and do not require changing this material/pipeline architecture.
    let viewPosition = input.position * 82.0 + vec3<f32>(185.0, -25.0, -650.0);
    output.viewPosition = viewPosition;
    output.normal = input.normal;

    // Aurora's GX shaders use row-vector multiplication; use the exact same convention and bytes.
    output.clipPosition = vec4<f32>(viewPosition, 1.0) * gxCamera.projection;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let v = normalize(-input.viewPosition);
    let l = normalize(vec3<f32>(-0.42, 0.72, 0.55));
    let h = normalize(v + l);

    let baseColor = vec3<f32>(0.96, 0.23, 0.035);
    let metallic = 0.72;
    let roughness = 0.22;
    let f0 = vec3<f32>(0.04) * (1.0 - metallic) + baseColor * metallic;

    let nDotL = max(dot(n, l), 0.0);
    let nDotV = max(dot(n, v), 0.0);
    let fresnel = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let ndf = distribution_ggx(n, h, roughness);
    let geometry = geometry_smith(n, v, l, roughness);
    let specular = (ndf * geometry * fresnel) / max(4.0 * nDotV * nDotL, 0.0001);
    let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * baseColor / PI;

    let direct = (diffuse + specular) * vec3<f32>(5.5, 4.8, 4.2) * nDotL;
    let ambient = baseColor * 0.055 + f0 * 0.085;
    let rim = pow(1.0 - nDotV, 3.0) * vec3<f32>(0.35, 0.08, 0.02);
    let mapped = aces_tonemap(ambient + direct + rim);
    let displayColor = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(displayColor, 1.0);
}
)""";

  const wgpu::ShaderModuleDescriptor shaderDescriptor{
      .nextInChain = &shaderSource,
      .label = "MKart Modern Scene PBR Shader",
  };
  g_shaderModule = g_device.CreateShaderModule(&shaderDescriptor);

  const std::array cameraEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
                  .hasDynamicOffset = true,
              },
      },
  };
  const wgpu::BindGroupLayoutDescriptor cameraLayoutDescriptor{
      .label = "MKart Modern Scene GX Camera Layout",
      .entryCount = cameraEntries.size(),
      .entries = cameraEntries.data(),
  };
  g_cameraLayout = g_device.CreateBindGroupLayout(&cameraLayoutDescriptor);

  const std::array cameraBindEntries{
      wgpu::BindGroupEntry{
          .binding = 0,
          .buffer = g_uniformBuffer,
          .size = gx::MaxUniformSize,
      },
  };
  const wgpu::BindGroupDescriptor cameraBindDescriptor{
      .label = "MKart Modern Scene GX Camera Bind Group",
      .layout = g_cameraLayout,
      .entryCount = cameraBindEntries.size(),
      .entries = cameraBindEntries.data(),
  };
  g_cameraBindGroup = g_device.CreateBindGroup(&cameraBindDescriptor);

  const std::array layouts{g_cameraLayout};
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "MKart Modern Scene Pipeline Layout",
      .bindGroupLayoutCount = layouts.size(),
      .bindGroupLayouts = layouts.data(),
  };
  g_pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);

  const std::array vertexAttributes{
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x3,
          .offset = 0,
          .shaderLocation = 0,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x3,
          .offset = 12,
          .shaderLocation = 1,
      },
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
      .depthCompare = g_ignoreDepth ? wgpu::CompareFunction::Always : wgpu::CompareFunction::LessEqual,
  };
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = "MKart Modern Scene PBR Pipeline",
      .layout = g_pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = g_shaderModule,
              .entryPoint = "vs_main",
              .bufferCount = vertexLayouts.size(),
              .buffers = vertexLayouts.data(),
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
              .frontFace = wgpu::FrontFace::CCW,
              .cullMode = wgpu::CullMode::Back,
          },
      .depthStencil = &depthState,
      .multisample =
          wgpu::MultisampleState{
              .count = 1,
          },
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
  // Until pass sample-count metadata is carried in the sealed command, only run with MSAA disabled.
  // Then every main/offscreen render attachment is single-sampled and pipeline compatibility is exact.
  if (g_graphicsConfig.msaaSamples != 1) {
    if (!g_loggedMsaaRejection.exchange(true, std::memory_order_relaxed)) {
      Log.warn("POC rejected scene draws because MSAA sample count is {} (expected 1)",
               g_graphicsConfig.msaaSamples);
    }
    return false;
  }

  // Large non-instanced triangle draws are a cheap, allocation-free proxy for a 3D scene draw.
  // UI and point/line expansion are normally much smaller or use multiple instances.
  return draw.instanceCount == 1 && draw.indexCount >= kMinimumCandidateIndices &&
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
    Log.info("POC enabled: msaaSamples={} ignoreDepth={}", g_graphicsConfig.msaaSamples,
             g_ignoreDepth ? "true" : "false");
  }

  if (!candidate_draw(draw, effectiveUniformRange)) {
    return;
  }

  if (!g_loggedCandidate.exchange(true, std::memory_order_relaxed)) {
    Log.info("Selected GX camera candidate: indices={} instances={} uniformOffset={} uniformSize={}",
             draw.indexCount, draw.instanceCount, effectiveUniformRange.offset,
             effectiveUniformRange.size);
  }

  initialize();
  if (!g_initialized.load(std::memory_order_acquire)) {
    return;
  }

  // Bind the exact sealed GX uniform range used by the draw we follow. When Aurora replays an
  // interpolated presentation slot, pipeline.cpp passes the interpolated range here automatically.
  const std::array dynamicOffsets{effectiveUniformRange.offset};
  pass.SetPipeline(g_pipeline);
  pass.SetBindGroup(0, g_cameraBindGroup, dynamicOffsets.size(), dynamicOffsets.data());
  pass.SetVertexBuffer(0, g_vertexBuffer, 0, sizeof(kCubeVertices));
  pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16, 0, sizeof(kCubeIndices));
  pass.DrawIndexed(static_cast<uint32_t>(kCubeIndices.size()));

  if (!g_loggedDraw.exchange(true, std::memory_order_relaxed)) {
    Log.info("Issued modern scene DrawIndexed: indices={} cameraUniformOffset={}",
             kCubeIndices.size(), effectiveUniformRange.offset);
  }

  state.modernSceneDrawn = true;

  // The custom pipeline replaced bindings that GX caches in DrawEncodeState. Invalidate only those
  // encoder-side memos; the next GX draw will restore its pipeline/index/texture state normally.
  state.currentPipeline = UINTPTR_MAX;
  state.boundTextureBindGroup = nullptr;
  state.indexBufferBound = false;
}

void shutdown() noexcept {
  std::lock_guard lock{g_resourceMutex};
  g_initialized.store(false, std::memory_order_release);
  g_indexBuffer = {};
  g_vertexBuffer = {};
  g_pipeline = {};
  g_pipelineLayout = {};
  g_cameraBindGroup = {};
  g_cameraLayout = {};
  g_shaderModule = {};
}

} // namespace aurora::gfx::modern_scene
