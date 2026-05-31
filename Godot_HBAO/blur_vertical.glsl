#[vertex]

#version 450

layout(location = 0) out vec2 uv_interp;

void main() {
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

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D blur_input;

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

const int KERNEL_RADIUS = 5;

vec2 AORes = params.raster_size;
vec2 InvAORes = vec2(1.0/AORes.x, 1.0/AORes.y);
float g_Sharpness = Scene.Sharpness;

float decode_depth(vec3 value) {
    float unsigned_value = dot(value, vec3(1.0f, 1.0f/255.0f, 1.0/65025.0f));
    return ((unsigned_value) * 2.0 - 1.0);
}

void BlurFunction(vec2 uv, float r, vec4 center_c, float center_d, inout float w_total, inout vec4 c_total)
{
    vec4  tex = texture( blur_input, uv );

    float c = tex.a;
    float d = decode_depth(tex.rgb);

    float BlurSigma = float(KERNEL_RADIUS) * 0.5;
    float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

    float ddiff = (d - center_d) * g_Sharpness;
    float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
    w_total += w;

    c_total += c*w;
}

void main() {
    vec2 uv = uv_interp;
    vec2 size = AORes;

    vec4 ssao = texture(blur_input, uv);

    float depth = decode_depth(ssao.rgb);

    float inc = 2.0/AORes.y;

    float w_total = 1.0;
    vec4 c_total = ssao;

    for (float x = 1; x <= KERNEL_RADIUS; ++x)
    {
        vec2 blur_uv = uv + vec2(0.0,inc) * x;
        BlurFunction(blur_uv, x, ssao, depth, w_total, c_total);
    }

    for (float x = 1; x <= KERNEL_RADIUS; ++x)
    {
        vec2 blur_uv = uv - vec2(0.0,inc) * x;
        BlurFunction(blur_uv, x, ssao, depth, w_total, c_total);
    }

    c_total = vec4(c_total.a,c_total.a,c_total.a,c_total.a);

    vec4 final_color = c_total/w_total;

    // Write back to our color buffer.
    frag_color = final_color;
}
