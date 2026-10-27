/* The Metal backend's shader table (pc/tools/metal_shaders.py generates
 * mtl_shaders.c and rood.metallib from vk_shaders.h): for each Vulkan shader,
 * its Metal function and where its resources go. */
#ifndef ROOD_MTL_SHADERS_H
#define ROOD_MTL_SHADERS_H
#include <stdint.h>

enum { MTL_ARG_TEXTURE = 1, MTL_ARG_SAMPLER = 2, MTL_ARG_BUFFER = 3, MTL_ARG_STRUCTURE = 4,
    MTL_ARG_ATOMIC = 5 /* a storage image's atomics: a buffer the image is a view of */ };
typedef struct {
    int binding; /* the Vulkan binding (set 0) */
    int id;      /* the argument's index in the argument buffer (8 bytes each) */
    int kind;    /* MTL_ARG_* */
} mtl_argument;
typedef struct {
    uint32_t spirv_bytes;   /* the SPIR-V module's size... */
    uint64_t spirv_hash;    /* ...and the first 8 bytes of its SHA-256, little-endian */
    const char* function;   /* in rood.metallib */
    int set_buffer;         /* buffer index of set 0's argument buffer (-1: none) */
    int push_buffer;        /* buffer index of the push constants (-1: none) */
    const mtl_argument* arguments;
    int argument_count;
    int workgroup[3];       /* compute: the local size */
} mtl_shader_info;
extern const mtl_shader_info mtl_shaders[];
extern const int mtl_shader_count;
#endif
