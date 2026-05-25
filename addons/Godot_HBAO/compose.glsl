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

layout(set = 0, binding = 1) uniform sampler2D color_image;
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

void main() {
    vec4 color = texture(color_image, uv_interp);
    frag_color = color * texture(blur_image,uv_interp);
}
