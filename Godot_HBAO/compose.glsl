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

layout(set = 0, binding = 0) uniform sampler2D ssao_color;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

void main() {
    vec4 color = texture(ssao_color, uv_interp / 2.0);
    frag_color = vec4(color.a,color.a,color.a,1.0);
}
