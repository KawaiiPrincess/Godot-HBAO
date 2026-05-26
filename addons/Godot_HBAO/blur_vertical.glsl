/* clang-format off */
#[vertex]

#version 450

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {
    // old code, ARM driver bug on Mali-GXXx GPUs and Vulkan API 1.3.xxx
    // https://github.com/godotengine/godot/pull/92817#issuecomment-2168625982
    //vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
    //gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
    //uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0

    vec2 vertex_base;
    if (gl_VertexIndex == 0) {
        vertex_base = vec2(-1.0, -1.0);
    } else if (gl_VertexIndex == 1) {
        vertex_base = vec2(-1.0, 3.0);
    } else {
        vertex_base = vec2(3.0, -1.0);
    }
    gl_Position = vec4(vertex_base, 0.0, 1.0);
    uv_interp = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

#[fragment]

#version 450

layout(set = 0, binding = 1) uniform sampler2D depth_texture;
layout(set = 0, binding = 2) uniform sampler2D blur_image;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

layout(set=2, binding=0) uniform uniformBuffer {
    mat4 inv_proj;
    mat4 proj;
} mat;

layout(set=3, binding=0) uniform SceneBuffer {
    float Bias;
    float Strength;
    float Radius;
    float Sharpness;
    float Power;
    float null1;
    float null2;
    float LargeStrength;
} Scene;

layout(push_constant, std430) uniform Params {
    vec2 raster_size;
    vec2 reserved;
} params;

const int KERNEL_RADIUS = 3;

const float max24int = 256.0 * 256.0 * 256.0 - 1.0;

vec2 AORes = params.raster_size / 2.0;
float g_Sharpness = Scene.Sharpness;


float decode_depth(vec2 value) {
    return dot(value, vec2(1.0f, 1.0f/255.0f));
}

void BlurFunction(vec2 uv, float r, vec4 center_c, float center_d, inout float w_total, inout vec4 c_total)
{
    vec4  tex = texture( blur_image, uv );

    float c = tex.a;
    float d = decode_depth(tex.rg) * 2.0 - 1.0;

    const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
    const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

    float ddiff = (d - center_d) * g_Sharpness;
    float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
    w_total += w;

    c_total += c*w;
}

void main() {
    vec2 sampler_uv = uv_interp;

    vec4 ssao = texture(blur_image, sampler_uv);

    float depth = decode_depth(ssao.rg) * 2.0 - 1.0;

    float inc = 1.0/AORes.y * 2.0;

    float w_total = 1.0;
    vec4 c_total = ssao;

    for (float y = 1; y <= KERNEL_RADIUS; ++y)
    {
        vec2 blur_uv = (sampler_uv) + vec2(0.0,inc) * y;
        BlurFunction(blur_uv, y, ssao, depth, w_total, c_total);
    }

    for (float y = 1; y <= KERNEL_RADIUS; ++y)
    {
        vec2 blur_uv = sampler_uv - vec2(0.0,inc) * y;
        BlurFunction(blur_uv, y, ssao, depth, w_total, c_total);
    }

    c_total = vec4(c_total.a,c_total.a,c_total.a,c_total.a);

    vec4 final_color = c_total/w_total;

    // Write back to our color buffer.
    frag_color = final_color;
}
