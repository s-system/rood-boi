// The particles' composite (particles.h; pc/docs/Particles.md), shared by
// scaled.frag and OpenGL's present shader (gl_shaders.py PC_GLSL_PARTICLES):
// the mote that owns this pixel's game pixel (the key image: one texel per
// game pixel of the display, depth << 16 | mote, 0xFFFFFFFF none), its
// colour (one texel per mote, 256 wide) when it is in front of the scene
// there (d: the scene's depth, SZ / 65536; UI pixels have none). Whole game
// pixels, fetched, never filtered.
#ifndef PARTICLES_GLSL
#define PARTICLES_GLSL
// PARTICLES_SEPARATE (scaled.frag): the images come as textures, fetched
// through the shader's sampler s_nearest (Metal allows 16 samplers a stage).
#ifdef PARTICLES_SEPARATE
#define PARTICLES_KEYS utexture2D
#define PARTICLES_COLOURS texture2D
#define PARTICLES_FETCH_KEY(t, p) texelFetch(usampler2D(t, s_nearest), p, 0)
#define PARTICLES_FETCH_COLOUR(t, p) texelFetch(sampler2D(t, s_nearest), p, 0)
#else
#define PARTICLES_KEYS usampler2D
#define PARTICLES_COLOURS sampler2D
#define PARTICLES_FETCH_KEY(t, p) texelFetch(t, p, 0)
#define PARTICLES_FETCH_COLOUR(t, p) texelFetch(t, p, 0)
#endif
vec3 particles_at(PARTICLES_KEYS keys, PARTICLES_COLOURS colours, ivec2 display, vec2 uv, float d) {
    ivec2 cell = clamp(ivec2(floor(uv * vec2(display))), ivec2(0), display - 1);
    uint k = PARTICLES_FETCH_KEY(keys, cell).r;
    float z = d * 65536.0;
    if (k == 0xffffffffu || z < 0.5 || (d < 1.0 && float(k >> 16) >= z)) return vec3(0.0);
    int i = int(k & 0xffffu);
    return max(PARTICLES_FETCH_COLOUR(colours, ivec2(i & 255, i >> 8)).rgb, vec3(0.0));
}
#endif
