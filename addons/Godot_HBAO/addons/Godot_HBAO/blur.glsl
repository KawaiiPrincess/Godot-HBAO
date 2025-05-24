#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D color_image;
layout(set = 0, binding = 1) uniform sampler2D depth_texture;
layout(r8, set = 0, binding = 2) uniform image2D blur_image;

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

const int KERNEL_RADIUS = 8;

vec2 AORes = params.raster_size;
float g_Sharpness = Scene.Sharpness;

void lineardepth(vec2 uv, float depth, mat4 invmatrix, inout float linear_depth) {
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth);

    vec4 view = invmatrix * vec4(ndc, 1.0);
    view.xyz /= view.w;
    linear_depth = view.z ;
}

void BlurFunction(vec2 uv, ivec2 fragcoord, float r, vec4 center_c, float center_d, inout float w_total, inout vec4 c_total)
    {
    vec4  c = imageLoad( blur_image, fragcoord );
    float d = texture( depth_texture, uv).x;
    lineardepth(uv, d, mat.proj, d);

    const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
    const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

    float ddiff = (d - center_d) * g_Sharpness;
    float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
    w_total += w;

    c_total += c*w;
}

void main() {
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(params.raster_size);

    // Prevent reading/writing out of bounds.
    if (uv.x >= size.x || uv.y >= size.y) {
        return;
    }

    vec4 color = imageLoad(color_image, uv);
    vec4 blur = imageLoad(blur_image, uv);

    vec2 sampler_uv = (vec2(uv) + 0.5) / size;

    float depth = texture(depth_texture, sampler_uv).r;

    float sky = depth;
    sky *= 20000;
    sky = clamp(sky,0.0,1.0);
    sky = 1.0 - sky;

    lineardepth(sampler_uv, depth, mat.proj, depth);

    float flip = -1.0;
    float w_total = 1.0;
    vec4 final_color = vec4(1.0);
    vec4 c_total = blur;

    for (float x = 1; x <= KERNEL_RADIUS; ++x)
    {
        vec2 direction = (int(x) % 2 == 0) ? vec2(flip * (1/AORes.x),0) : vec2(-flip * (1/AORes.x),flip * (1/AORes.y));
        vec2 blur_uv = (sampler_uv) - direction * x;
        ivec2 blur_fragcoord = ivec2( blur_uv * size);
        BlurFunction(blur_uv, blur_fragcoord, x, blur, depth, w_total, c_total);
        flip = (int(x) % 2 == 0) ? 1.0 : -1.0;
    }

    for (float y = 1; y <= KERNEL_RADIUS; ++y)
    {
        vec2 direction = (int(y) % 2 == 0) ? vec2(0,flip * (1/AORes.y)) : vec2(flip * (1/AORes.x),flip * (1/AORes.y));
        vec2 blur_uv = sampler_uv - direction * y;
        ivec2 blur_fragcoord = ivec2( blur_uv * size);
        BlurFunction(blur_uv, blur_fragcoord, y, blur, depth, w_total, c_total);
        flip = (int(y) % 2 == 0) ? 1.0 : -1.0;
    }

    final_color = c_total/w_total;

    final_color = clamp(final_color,0.0,1.0) + sky;
    final_color = clamp(final_color,0.0,1.0);

    final_color *= color;

    // Write back to our color buffer.
    imageStore(color_image, uv, final_color);
}
