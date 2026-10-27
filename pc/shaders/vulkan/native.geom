#version 450
#extension GL_GOOGLE_include_directive : require
// Native lists (render.c's geometry stage): the PS1's per-polygon rules on
// real view-space positions, then the polygon's triangles projected as the
// GTE projects, with real clipping and exact per-pixel depth.
layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 6) out;
#include "native_common.glsl"
layout(location = 0) in vec3 g_pos[];
layout(location = 1) in vec3 g_normal[];
layout(location = 2) flat in uvec4 g_color[];
layout(location = 3) flat in uvec4 g_uvf[];
layout(location = 4) flat in uvec2 g_page[];
layout(location = 5) flat in uint g_bias[];
layout(location = 6) flat in uint g_effect[];
layout(location = 7) flat in uint g_scope[];
layout(location = 0) noperspective centroid out vec3 v_color_affine;
layout(location = 1) noperspective centroid out vec2 v_uv_affine;
layout(location = 2) smooth centroid out vec3 v_color_persp;
layout(location = 3) smooth centroid out vec2 v_uv_persp;
layout(location = 4) smooth centroid out float v_z;
layout(location = 5) flat out uvec4 v_state; // flags, tpage, clut, effect | halved << 8 | emissive scope bits << 16
layout(location = 6) flat out vec3 v_normal;
layout(location = 7) flat out uvec4 v_bounds;
layout(location = 8) smooth centroid out vec3 v_view; // view space, rows' scales divided out (cutaway)
layout(location = 9) flat out int v_cut; // bit 0: the polygon stands between the eye and the cutaway focus;
                                         // bits 1-3: the triangle's real edges (opposite corner 0, 1, 2);
                                         // bits 4-7: occlusion subjects the polygon stands in front of
layout(location = 10) noperspective out vec3 v_bary; // the corner weights (a cutaway ghost's edges)
layout(location = 11) flat out uint v_order; // room polygons: PS1 draw order key (bucket << 10 | sequence; smaller paints later)
#define NATIVE_PRIMITIVE_ID gl_PrimitiveIDIn
#include "native_geom_body.glsl"
void main() { native_geom_main(); }
