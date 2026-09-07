#include "modern_renderer_poc.hpp"

#include "../webgpu/gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>

namespace aurora::gfx::modern_poc {
namespace {

using webgpu::g_device;
using webgpu::g_graphicsConfig;

std::mutex g_stateMutex;
wgpu::ShaderModule g_shaderModule;
wgpu::PipelineLayout g_pipelineLayout;
wgpu::RenderPipeline g_pipeline;
bool g_initialized = false;

bool read_enabled_from_environment() noexcept {
  const char* value = std::getenv("AURORA_MODERN_RENDERER_POC");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

const bool g_enabled = read_enabled_from_environment();

void create_pipeline() {
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = R"""(
const PI: f32 = 3.14159265359;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
);

var<private> texcoords: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 2.0),
    vec2<f32>(2.0, 0.0),
);

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let nDotH = max(dot(n, h), 0.0);
    let nDotH2 = nDotH * nDotH;
    let denominator = nDotH2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 0.0001);
}

fn geometry_schlick_ggx(nDotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 0.0001);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let nDotV = max(dot(n, v), 0.0);
    let nDotL = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(nDotV, roughness) * geometry_schlick_ggx(nDotL, roughness);
}

fn fresnel_schlick(cosTheta: f32, f0: vec3<f32>) -> vec3<f32> {
    let oneMinusCos = clamp(1.0 - cosTheta, 0.0, 1.0);
    return f0 + (vec3<f32>(1.0) - f0) * pow(oneMinusCos, 5.0);
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
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    out.uv = texcoords[vertexIndex];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - vec2<f32>(1.0);
    let radiusSquared = dot(p, p);
    if (radiusSquared > 1.0) {
        discard;
    }

    let z = sqrt(max(1.0 - radiusSquared, 0.0));
    var n = normalize(vec3<f32>(p.x, -p.y, z));

    // Procedural fine normal detail proves this shader is not limited to GX/TEV inputs.
    let normalDetail = 0.035 * sin(p.x * 48.0) * sin(p.y * 43.0);
    n = normalize(n + vec3<f32>(normalDetail, -normalDetail, 0.0));

    let v = vec3<f32>(0.0, 0.0, 1.0);
    let l = normalize(vec3<f32>(-0.45, 0.55, 0.80));
    let h = normalize(v + l);

    let baseColor = vec3<f32>(0.035, 0.28, 0.95);
    let metallic = 0.78;
    let roughnessVariation = 0.5 + 0.5 * sin((p.x + p.y) * 31.0);
    let roughness = 0.16 + roughnessVariation * 0.12;

    let f0 = vec3<f32>(0.04) * (1.0 - metallic) + baseColor * metallic;
    let ndf = distribution_ggx(n, h, roughness);
    let geometry = geometry_smith(n, v, l, roughness);
    let fresnel = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let nDotV = max(dot(n, v), 0.0);
    let nDotL = max(dot(n, l), 0.0);
    let specular = (ndf * geometry * fresnel) / max(4.0 * nDotV * nDotL, 0.0001);

    let kD = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic);
    let diffuse = kD * baseColor / PI;
    let keyLight = vec3<f32>(5.0, 4.4, 3.8);

    let ambient = baseColor * 0.045 + f0 * 0.10;
    let rim = pow(1.0 - nDotV, 3.0) * vec3<f32>(0.05, 0.35, 1.8);

    // HDR-bright emission gives a later bloom chain useful signal to threshold.
    let radius = sqrt(radiusSquared);
    let emissionMask = exp(-abs(radius - 0.72) * 55.0);
    let emission = vec3<f32>(0.03, 0.65, 3.2) * emissionMask * 2.0;

    let hdrColor = ambient + (diffuse + specular) * keyLight * nDotL + rim + emission;
    let mapped = aces_tonemap(hdrColor);
    let gammaCorrected = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gammaCorrected, 0.96);
}
)""";

  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "MKart Modern Renderer POC Shader",
  };
  g_shaderModule = g_device.CreateShaderModule(&moduleDescriptor);

  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "MKart Modern Renderer POC Layout",
      .bindGroupLayoutCount = 0,
      .bindGroupLayouts = nullptr,
  };
  g_pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  const wgpu::BlendState blendState{
      .color =
          wgpu::BlendComponent{
              .operation = wgpu::BlendOperation::Add,
              .srcFactor = wgpu::BlendFactor::SrcAlpha,
              .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha,
          },
      .alpha =
          wgpu::BlendComponent{
              .operation = wgpu::BlendOperation::Add,
              .srcFactor = wgpu::BlendFactor::One,
              .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha,
          },
  };
  const wgpu::ColorTargetState colorTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .blend = &blendState,
      .writeMask = wgpu::ColorWriteMask::All,
  };
  const wgpu::FragmentState fragmentState{
      .module = g_shaderModule,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &colorTarget,
  };
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = "MKart Modern Renderer POC Pipeline",
      .layout = g_pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = g_shaderModule,
              .entryPoint = "vs_main",
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
          },
      .multisample =
          wgpu::MultisampleState{
              .count = 1,
          },
      .fragment = &fragmentState,
  };
  g_pipeline = g_device.CreateRenderPipeline(&pipelineDescriptor);
}

} // namespace

void initialize() noexcept {
  if (!g_enabled) {
    return;
  }

  std::lock_guard lock{g_stateMutex};
  if (g_initialized) {
    return;
  }
  create_pipeline();
  g_initialized = true;
}

void shutdown() noexcept {
  std::lock_guard lock{g_stateMutex};
  g_pipeline = {};
  g_pipelineLayout = {};
  g_shaderModule = {};
  g_initialized = false;
}

bool enabled() noexcept { return g_enabled; }

void render(const wgpu::RenderPassEncoder& pass, const wgpu::Extent3D& targetSize) noexcept {
  if (!g_enabled || targetSize.width == 0 || targetSize.height == 0) {
    return;
  }

  initialize();
  if (!g_initialized) {
    return;
  }

  const float targetWidth = static_cast<float>(targetSize.width);
  const float targetHeight = static_cast<float>(targetSize.height);
  const float maxProbeSize = std::min(targetWidth, targetHeight);
  const float probeSize = std::min(maxProbeSize, std::max(64.0f, maxProbeSize * 0.28f));
  const float margin = std::min(24.0f, probeSize * 0.08f);
  const float left = std::max(0.0f, targetWidth - probeSize - margin);
  const float top = std::max(0.0f, margin);

  pass.SetPipeline(g_pipeline);
  pass.SetViewport(left, top, probeSize, probeSize, 0.0f, 1.0f);
  pass.SetScissorRect(static_cast<uint32_t>(left), static_cast<uint32_t>(top),
                      static_cast<uint32_t>(probeSize), static_cast<uint32_t>(probeSize));
  pass.Draw(3);
}

} // namespace aurora::gfx::modern_poc
