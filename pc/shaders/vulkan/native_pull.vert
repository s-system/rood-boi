#version 450
#extension GL_GOOGLE_include_directive : require
// Native lists without a geometry stage (MoltenVK: Metal has none): six
// vertices per polygon, each running the polygon's stage (native_geom_body)
// on the polygon's four vertices, read from the mesh's storage buffer, and
// keeping the one vertex it is (two triangles at most; a polygon that emits
// fewer leaves zero-area triangles).
//
// Draw: vkCmdDraw(polygons * 6, 1, first * 6, instance). gl_InstanceIndex 0
// draws the mesh's polygons in order; k + 1 reads polygon numbers from the
// index buffer (four indices a polygon, as the indexed geometry-shader draws
// use), starting at its polygon k, so NATIVE_PRIMITIVE_ID counts from there.
#include "native_common.glsl"
layout(std430, binding = 1) readonly buffer Bones {
    vec4 bones[];
};
// render.c's mesh_vertex, 40 bytes: position[3], normal[3] (floats),
// r g b code, u v flags bone, tpage | clut << 16, bias | uv_effect << 8 | scope << 16.
layout(std430, binding = 7) readonly buffer Vertices {
    uint words[];
};
layout(std430, binding = 8) readonly buffer Order {
    uint indices[];
};
layout(location = 0) noperspective centroid out vec3 o_color_affine;
layout(location = 1) noperspective centroid out vec2 o_uv_affine;
layout(location = 2) smooth centroid out vec3 o_color_persp;
layout(location = 3) smooth centroid out vec2 o_uv_persp;
layout(location = 4) smooth centroid out float o_z;
layout(location = 5) flat out uvec4 o_state;
layout(location = 6) flat out vec3 o_normal;
layout(location = 7) flat out uvec4 o_bounds;
layout(location = 8) smooth centroid out vec3 o_view;
layout(location = 9) flat out int o_cut;
layout(location = 10) noperspective out vec3 o_bary;
layout(location = 11) flat out uint o_order;
// The polygon's vertices, as native.vert hands them to the geometry stage.
vec3 g_pos[4];
vec3 g_normal[4];
uvec4 g_color[4];
uvec4 g_uvf[4];
uvec2 g_page[4];
uint g_bias[4];
uint g_effect[4];
uint g_scope[4];
// The stage's outputs, kept for the vertex this invocation is.
vec3 v_color_affine;
vec2 v_uv_affine;
vec3 v_color_persp;
vec2 v_uv_persp;
float v_z;
uvec4 v_state;
vec3 v_normal;
uvec4 v_bounds;
vec3 v_view;
int v_cut;
vec3 v_bary;
uint v_order;
vec4 pull_position;
int pull_slot, pull_emitted = 0;
uint pull_primitive;
void pull_emit() {
    if (pull_emitted++ != pull_slot) return;
    gl_Position = pull_position;
    o_color_affine = v_color_affine;
    o_uv_affine = v_uv_affine;
    o_color_persp = v_color_persp;
    o_uv_persp = v_uv_persp;
    o_z = v_z;
    o_state = v_state;
    o_normal = v_normal;
    o_bounds = v_bounds;
    o_view = v_view;
    o_cut = v_cut;
    o_bary = v_bary;
    o_order = v_order;
}
uvec4 bytes4(uint w) { return uvec4(w & 255u, (w >> 8) & 255u, (w >> 16) & 255u, w >> 24); }
void fetch(int i, uint vertex) {
    uint w = vertex * 10u;
    vec3 position = uintBitsToFloat(uvec3(words[w], words[w + 1u], words[w + 2u]));
    g_normal[i] = uintBitsToFloat(uvec3(words[w + 3u], words[w + 4u], words[w + 5u]));
    g_color[i] = bytes4(words[w + 6u]);
    g_uvf[i] = bytes4(words[w + 7u]);
    g_page[i] = uvec2(words[w + 8u] & 0xffffu, words[w + 8u] >> 16);
    g_bias[i] = words[w + 9u] & 255u;
    g_effect[i] = (words[w + 9u] >> 8) & 255u;
    g_scope[i] = words[w + 9u] >> 16;
    // native.vert: into GTE view space through the vertex's bone.
    uint b = min(g_uvf[i].w, 63u);
    mat3 rotation = mat3(bones[b * 4u].xyz, bones[b * 4u + 1u].xyz, bones[b * 4u + 2u].xyz);
    g_pos[i] = rotation * position + bones[b * 4u + 3u].xyz;
}
#define gl_Position pull_position
#define EmitVertex() pull_emit()
#define EndPrimitive()
#define NATIVE_PRIMITIVE_ID int(pull_primitive)
#include "native_geom_body.glsl"
#undef gl_Position
void main() {
    uint polygon = uint(gl_VertexIndex) / 6u;
    pull_slot = gl_VertexIndex % 6;
    if (gl_InstanceIndex == 0) {
        pull_primitive = polygon;
    } else {
        pull_primitive = polygon - uint(gl_InstanceIndex - 1);
        polygon = indices[polygon * 4u] / 4u;
    }
    for (int i = 0; i < 4; i++)
        fetch(i, polygon * 4u + uint(i));
    // A vertex the stage does not emit: a zero-area triangle.
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    native_geom_main();
}
