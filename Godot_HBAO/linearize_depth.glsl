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

layout(set = 0, binding = 0) uniform sampler2D depth_input;

layout(set=2, binding=0) uniform uniformBuffer {
    mat4 inv_proj;
    mat4 proj;
} mat;

layout(push_constant, std430) uniform Params {
    vec2 raster_size;
    vec2 reserved;
} params;

vec2 depthRes = params.raster_size * 2;
vec2 InvdepthRes = vec2(1.0/depthRes.x, 1.0/depthRes.y);

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

void linearize_depth(in vec2 uv, inout float depth) {
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth);
    vec4 view = mat.proj * vec4(ndc, 1.0);
    view.xyz /= view.w;
    depth = view.z;
}

void main() {
    vec2 base_texelCoord = uv_interp;

    vec2 texelCoord[4];
    texelCoord[0] = (base_texelCoord + vec2(1*InvdepthRes.x,1*InvdepthRes.y));
    texelCoord[1] = (base_texelCoord + vec2(3*InvdepthRes.x,1*InvdepthRes.y));
    texelCoord[2] = (base_texelCoord + vec2(1*InvdepthRes.x,3*InvdepthRes.y));
    texelCoord[3] = (base_texelCoord + vec2(3*InvdepthRes.x,3*InvdepthRes.y));

    float gathered_depth[4];
    gathered_depth[0] = textureGather(depth_input,texelCoord[0]).x;
    gathered_depth[1] = textureGather(depth_input,texelCoord[1]).x;
    gathered_depth[2] = textureGather(depth_input,texelCoord[2]).x;
    gathered_depth[3] = textureGather(depth_input,texelCoord[3]).x;

    linearize_depth(uv_interp,gathered_depth[0]);
    linearize_depth(uv_interp,gathered_depth[1]);
    linearize_depth(uv_interp,gathered_depth[2]);
    linearize_depth(uv_interp,gathered_depth[3]);

    float gatheredTexelMins = min(min(gathered_depth[0], gathered_depth[1]),
                                  min(gathered_depth[2], gathered_depth[3]));

    frag_color = vec4(gatheredTexelMins);
}
