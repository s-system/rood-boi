/* Metal backend core (ROOD_RENDERER=metal; Apple platforms).
 *
 * The Vulkan renderer (vk_hw.c, vk_present.c, vk_render.inc, ...) records
 * through vk_core.h and a few dozen vkCmd and vkCreate entry points. Here
 * those are implemented on Metal directly, in place of vk_core.c and a
 * Vulkan driver: the renderer's passes, order and shaders stay the ones the
 * other platforms run, and nothing goes through Vulkan or MoltenVK.
 *
 * Shaders: pc/tools/metal_shaders.py translates every SPIR-V module of
 * vk_shaders.h to MSL (rood.metallib, next to the executable) with a table
 * (mtl_shaders.h); vk_shader() finds a module's entry by its size and hash.
 * Descriptors: Vulkan's one push descriptor set is a Metal 3 argument buffer
 * (each resource an 8-byte GPU id or address), filled from the pushed
 * writes at each draw or dispatch; push constants are bytes set per stage.
 *
 * Recording: one command buffer per frame; a rendering scope is a render
 * command encoder (load and store every attachment, as vk_core.c's dynamic
 * rendering), dispatches go to a compute encoder and transfers to a blit
 * encoder. Metal's hazard tracking orders encoders; resources reached
 * through argument buffers are declared to each encoder (useResource), so
 * they are tracked too. Images have no layouts here: barriers inside an
 * encoder are memory barriers, between encoders nothing.
 *
 * Handles: Vulkan's non-dispatchable handles are pointers on 64-bit
 * platforms; each points at one of the structs below. Built without ARC:
 * the structs retain their Metal objects. */
#define VK_NO_PROTOTYPES
#include "vk_core.h"
#include "mtl_shaders.h"
#include "frame_cost.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>
#include <TargetConditionals.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <MetalFX/MetalFX.h>
#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <CommonCrypto/CommonDigest.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_DEFINE(name) PFN_##name name;
VK_GLOBAL_FUNCTIONS(VK_DEFINE)
VK_INSTANCE_FUNCTIONS(VK_DEFINE)
VK_DEVICE_FUNCTIONS(VK_DEFINE)
VK_OPTIONAL_DEVICE_FUNCTIONS(VK_DEFINE)
#undef VK_DEFINE
PFN_vkGetInstanceProcAddr vk_get_instance_proc_addr;
vk_context vk;
int pc_vk_active;

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> library;

/* ---- handles ---- */

typedef struct {
    id<MTLTexture> texture; /* a swapchain image's: the drawable's, set at each acquire */
    VkFormat format;
    int depth, samples;
    id<MTLBuffer> backing; /* an R32_UINT storage image: the buffer it is a linear view of (image atomics) */
} mtl_image;
/* metal_shaders.py LINEAR_ALIGN: the row stride's alignment of images atomics reach through a buffer. */
#define MTL_LINEAR_ALIGN 256
typedef struct {
    mtl_image* image; /* the view shows the whole image (vk_core.c makes no other) */
} mtl_view;
typedef struct {
    id<MTLBuffer> buffer;
    VkDeviceSize size;
} mtl_buffer;
typedef struct {
    id<MTLSamplerState> state;
} mtl_sampler;
typedef struct {
    const mtl_shader_info* info;
    id<MTLFunction> function;
} mtl_module;
typedef struct {
    int count;
    VkDescriptorType types[32];
} mtl_set_layout;
typedef struct {
    uint32_t push_bytes;
} mtl_layout;
typedef struct {
    int compute;
    id<MTLRenderPipelineState> render;
    id<MTLComputePipelineState> kernel;
    const mtl_shader_info *vs, *fs, *cs;
    MTLPrimitiveType primitive;
    int has_depth;
    float blend_constant; /* vk_core.c's: alpha 0.25 for PS1 mode 3 (B + F/4), else 0.5 */
} mtl_pipeline;
/* An acceleration structure (vk_rt.inc; see "ray tracing" below). */
typedef struct mtl_structure {
    id<MTLAccelerationStructure> structure;
    int top;
    NSMutableArray* bottoms; /* a TLAS's BLASes (residency) */
} mtl_structure;
typedef struct {
    CAMetalLayer* layer;
    SDL_MetalView view;
    SDL_Window* window;
} mtl_surface;
typedef struct {
    mtl_surface* surface;
    int count, next;
    VkExtent2D extent;
    mtl_image images[3];
    id<CAMetalDrawable> drawables[3];
} mtl_swapchain;

static MTLPixelFormat pixel_format(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return MTLPixelFormatRGBA8Unorm;
    case VK_FORMAT_B8G8R8A8_UNORM: return MTLPixelFormatBGRA8Unorm;
    case VK_FORMAT_R8G8B8A8_UINT: return MTLPixelFormatRGBA8Uint;
    case VK_FORMAT_R8G8B8A8_SNORM: return MTLPixelFormatRGBA8Snorm;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return MTLPixelFormatRGBA16Float;
    case VK_FORMAT_R16G16B16A16_UNORM: return MTLPixelFormatRGBA16Unorm;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return MTLPixelFormatRGBA32Float;
    case VK_FORMAT_R16_UINT: return MTLPixelFormatR16Uint;
    case VK_FORMAT_R16_SFLOAT: return MTLPixelFormatR16Float;
    case VK_FORMAT_R16G16_UINT: return MTLPixelFormatRG16Uint;
    case VK_FORMAT_R16G16_SINT: return MTLPixelFormatRG16Sint;
    case VK_FORMAT_R32_UINT: return MTLPixelFormatR32Uint;
    case VK_FORMAT_R32_SFLOAT: return MTLPixelFormatR32Float;
    case VK_FORMAT_R8_UINT: return MTLPixelFormatR8Uint;
    case VK_FORMAT_R8_UNORM: return MTLPixelFormatR8Unorm;
    case VK_FORMAT_R8G8_UNORM: return MTLPixelFormatRG8Unorm;
    case VK_FORMAT_D32_SFLOAT: return MTLPixelFormatDepth32Float;
    default:
        fprintf(stderr, "[mtl] no pixel format for VkFormat %d\n", (int)f);
        return MTLPixelFormatInvalid;
    }
}
static int format_is_integer(VkFormat f)
{
    return f == VK_FORMAT_R8G8B8A8_UINT || f == VK_FORMAT_R16_UINT || f == VK_FORMAT_R16G16_UINT
        || f == VK_FORMAT_R16G16_SINT || f == VK_FORMAT_R32_UINT || f == VK_FORMAT_R8_UINT;
}
static size_t texel_size(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R8G8_UNORM: return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 4;
    }
}
static MTLVertexFormat vertex_format(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R32G32B32_SFLOAT: return MTLVertexFormatFloat3;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return MTLVertexFormatFloat4;
    case VK_FORMAT_R32G32_SFLOAT: return MTLVertexFormatFloat2;
    case VK_FORMAT_R32_SFLOAT: return MTLVertexFormatFloat;
    case VK_FORMAT_R8G8B8A8_UNORM: return MTLVertexFormatUChar4Normalized;
    case VK_FORMAT_R8G8B8A8_UINT: return MTLVertexFormatUChar4;
    case VK_FORMAT_R8G8B8A8_SNORM: return MTLVertexFormatChar4Normalized;
    case VK_FORMAT_R8G8B8A8_SINT: return MTLVertexFormatChar4;
    case VK_FORMAT_R16G16_UINT: return MTLVertexFormatUShort2;
    case VK_FORMAT_R16G16_SINT: return MTLVertexFormatShort2;
    case VK_FORMAT_R32_UINT: return MTLVertexFormatUInt;
    case VK_FORMAT_R32G32_UINT: return MTLVertexFormatUInt2;
    case VK_FORMAT_R16G16B16A16_UINT: return MTLVertexFormatUShort4;
    case VK_FORMAT_R16G16B16A16_SINT: return MTLVertexFormatShort4;
    default:
        fprintf(stderr, "[mtl] no vertex format for VkFormat %d\n", (int)f);
        return MTLVertexFormatInvalid;
    }
}

/* ---- recording state ---- */

enum { ENC_NONE, ENC_RENDER, ENC_COMPUTE, ENC_BLIT };
#define VERTEX_BUFFER_INDEX 30 /* the vertex input's buffer (above every shader's own) */
typedef struct {
    VkDescriptorType type;
    id resource;       /* texture, sampler, buffer or acceleration structure */
    VkDeviceSize offset; /* buffers */
    int set;
    NSArray* bottoms;  /* a TLAS's BLASes: declared with it */
    id<MTLBuffer> backing; /* a storage image's buffer (MTL_ARG_ATOMIC) */
} binding_slot;
typedef struct {
    mtl_pipeline* pipeline;
    binding_slot slots[32];
    unsigned char push[256];
} bind_point;
static struct {
    id<MTLCommandBuffer> cb;
    int encoder;
    id<MTLRenderCommandEncoder> render;
    id<MTLComputeCommandEncoder> compute;
    id<MTLBlitCommandEncoder> blit;
    bind_point graphics, kernel;
    mtl_pipeline* applied; /* the render pipeline set on the encoder */
    int depth_test, depth_write;
    VkCompareOp depth_compare;
    id<MTLDepthStencilState> applied_depth;
    VkViewport viewport;
    VkRect2D scissor;
    int viewport_set, scissor_set;
    mtl_buffer* vertex_buffer;
    VkDeviceSize vertex_offset;
    mtl_buffer* index_buffer;
    VkDeviceSize index_offset;
    VkIndexType index_type;
    vk_attachments attachments; /* the open rendering scope's */
    int attachment_samples;
    VkFormat attachment_formats[6]; /* [5]: depth */
} rec;
static int rendering;
static int timer_current = -1; /* the open encoder's GPU timer (ROOD_PERF) */

static void end_encoder(void)
{
    switch (rec.encoder) {
    case ENC_RENDER: [rec.render endEncoding]; rec.render = nil; break;
    case ENC_COMPUTE: [rec.compute endEncoding]; rec.compute = nil; break;
    case ENC_BLIT: [rec.blit endEncoding]; rec.blit = nil; break;
    }
    rec.encoder = ENC_NONE;
    timer_current = -1;
}

/* ---- frames ---- */

typedef struct deferred {
    int kind; /* 0 buffer, 1 image */
    vk_buffer buffer;
    vk_image image;
} deferred;
typedef struct {
    vk_frame_callback callback;
    void* context;
} frame_callback;
#define ARENA_CHUNK (16u << 20)
typedef struct {
    vk_buffer buffer;
    VkDeviceSize used;
} arena_chunk;
static struct {
    dispatch_semaphore_t done; /* signalled by the frame's command buffer */
    int submitted;
    arena_chunk* chunks;
    int chunk_count, chunk_current;
    deferred* destroy;
    int destroy_count, destroy_capacity;
    frame_callback* callbacks;
    int callback_count, callback_capacity;
    double gpu_ms; /* written by the completion handler */
} frames[VK_FRAMES];
static int perf = -1;
static double gpu_sum, gpu_max;
static int gpu_count;
static id<MTLCommandBuffer> last_committed;

/* ROOD_PERF sections (vk_section): every encoder's start and end sampled
 * into the frame slot's counter buffer (Apple GPUs sample at stage
 * boundaries only), its time added to the section current when it began;
 * read when the slot comes round. GPU ticks become nanoseconds through
 * sampleTimestamps:gpuTimestamp: pairs. */
#define MTL_SECTIONS 127
#define MTL_TIMED_ENCODERS 1024
static struct {
    id<MTLCounterSampleBuffer> samples;
    int count; /* encoders timed this frame */
    const char* names[MTL_TIMED_ENCODERS];
    const char* details[MTL_TIMED_ENCODERS]; /* ROOD_PERF=detail: the encoder's first shader */
} timers[VK_FRAMES];
static const char* current_section = "draw";
static int perf_detail = -1; /* ROOD_PERF=detail */
/* The open encoder's first shader names it (ROOD_PERF=detail). */
static void timer_note(const char* function)
{
    if (perf_detail < 0) {
        const char* v = getenv("ROOD_PERF");
        perf_detail = v && !strcmp(v, "detail");
    }
    if (perf_detail && timer_current >= 0 && !timers[vk.frame].details[timer_current])
        timers[vk.frame].details[timer_current] = function;
}
static struct {
    const char* name;
    double sum;
    int encoders;
} section_totals[MTL_SECTIONS], section_rpc[MTL_SECTIONS];
static int section_total_count, section_rpc_count, section_rpc_frames;
static double gpu_tick_ns = 1.0;
static MTLTimestamp calibration_cpu, calibration_gpu;
static int timers_ready(void)
{
    static int ready = -1;
    if (ready >= 0)
        return ready;
    ready = 0;
    if (perf <= 0 || ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
        return 0;
    id<MTLCounterSet> timestamps = nil;
    for (id<MTLCounterSet> set in device.counterSets)
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
            timestamps = set;
    if (!timestamps)
        return 0;
    MTLCounterSampleBufferDescriptor* d = [[MTLCounterSampleBufferDescriptor new] autorelease];
    d.counterSet = timestamps;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = 2 * MTL_TIMED_ENCODERS;
    for (int i = 0; i < VK_FRAMES; i++) {
        NSError* error = nil;
        timers[i].samples = [device newCounterSampleBufferWithDescriptor:d error:&error];
        if (!timers[i].samples) {
            fprintf(stderr, "[mtl] no GPU timers: %s\n", error.localizedDescription.UTF8String);
            return 0;
        }
    }
    [device sampleTimestamps:&calibration_cpu gpuTimestamp:&calibration_gpu];
    ready = 1;
    return 1;
}
/* The sample indices for a new encoder of this frame (-1: not timed). */
static int timer_slot(void)
{
    if (!timers_ready() || timers[vk.frame].count >= MTL_TIMED_ENCODERS)
        return -1;
    int i = timers[vk.frame].count++;
    timers[vk.frame].names[i] = current_section;
    timers[vk.frame].details[i] = NULL;
    timer_current = i;
    return i;
}
static void add_section(const char* name, double ms, int encoders)
{
    int k = 0;
    while (k < section_total_count && strcmp(section_totals[k].name, name))
        ++k;
    if (k == section_total_count && k < MTL_SECTIONS)
        section_totals[section_total_count++] = (typeof(section_totals[0])){ name, 0 };
    if (k < MTL_SECTIONS)
        section_totals[k].sum += ms, section_totals[k].encoders += encoders;
    k = 0;
    while (k < section_rpc_count && strcmp(section_rpc[k].name, name))
        ++k;
    if (k == section_rpc_count && k < MTL_SECTIONS)
        section_rpc[section_rpc_count++] = (typeof(section_rpc[0])){ name, 0 };
    if (k < MTL_SECTIONS)
        section_rpc[k].sum += ms;
}
static void read_timers(int slot)
{
    int count = timers[slot].count;
    timers[slot].count = 0;
    if (!count || !timers[slot].samples)
        return;
    MTLTimestamp cpu, gpu;
    [device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    if (gpu > calibration_gpu + 1000000 && cpu > calibration_cpu)
        gpu_tick_ns = (double)(cpu - calibration_cpu) / (double)(gpu - calibration_gpu);
    NSData* data = [timers[slot].samples resolveCounterRange:NSMakeRange(0, (NSUInteger)(2 * count))];
    if (!data)
        return;
    /* Passes overlap on a tile GPU (a pass's vertices run under the previous
     * one's fragments), so a section's time is from the previous section's
     * last end to its own (the first: from the frame's first start). */
    const MTLCounterResultTimestamp* t = data.bytes;
    static uint64_t previous_end; /* the last timed frame's end: its passes overlap this one's start */
    uint64_t mark = 0, end = 0;
    const char* name = NULL;
    int run = 0; /* encoders in the current section */
    for (int i = 0; i < count; i++) {
        uint64_t a = t[2 * i].timestamp, b = t[2 * i + 1].timestamp;
        if (a == MTLCounterErrorValue || b == MTLCounterErrorValue)
            continue;
        if (!mark)
            mark = a > previous_end ? a : previous_end;
        const char* label = timers[slot].details[i] ? timers[slot].details[i] : timers[slot].names[i];
        if (name && strcmp(name, label)) {
            add_section(name, end > mark ? (double)(end - mark) * gpu_tick_ns * 1e-6 : 0.0, run);
            mark = end > mark ? end : mark;
            run = 0;
        }
        ++run;
        name = label;
        end = b > end ? b : end;
    }
    if (name)
        add_section(name, end > mark ? (double)(end - mark) * gpu_tick_ns * 1e-6 : 0.0, run);
    previous_end = end > previous_end ? end : previous_end;
    ++section_rpc_frames;
}

VkCommandBuffer vk_cmd(void)
{
    if (!rec.cb) {
        rec.cb = [[queue commandBufferWithUnretainedReferences] retain];
        if (perf < 0)
            perf = getenv("ROOD_PERF") != NULL;
    }
    return (VkCommandBuffer)(uintptr_t)1; /* recording goes to rec.cb */
}
static id<MTLRenderCommandEncoder> render_encoder(void)
{
    return rec.encoder == ENC_RENDER ? rec.render : nil;
}
static id<MTLComputeCommandEncoder> compute_encoder(void)
{
    if (rendering)
        vk_end_rendering();
    if (rec.encoder != ENC_COMPUTE) {
        end_encoder();
        vk_cmd();
        MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
        int t = timer_slot();
        if (t >= 0) {
            pass.sampleBufferAttachments[0].sampleBuffer = timers[vk.frame].samples;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = (NSUInteger)(2 * t);
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = (NSUInteger)(2 * t + 1);
        }
        rec.compute = [rec.cb computeCommandEncoderWithDescriptor:pass];
        rec.encoder = ENC_COMPUTE;
    }
    return rec.compute;
}
static id<MTLBlitCommandEncoder> blit_encoder(void)
{
    if (rendering)
        vk_end_rendering();
    if (rec.encoder != ENC_BLIT) {
        end_encoder();
        vk_cmd();
        MTLBlitPassDescriptor* pass = [MTLBlitPassDescriptor blitPassDescriptor];
        int t = timer_slot();
        if (t >= 0) {
            pass.sampleBufferAttachments[0].sampleBuffer = timers[vk.frame].samples;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = (NSUInteger)(2 * t);
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = (NSUInteger)(2 * t + 1);
        }
        rec.blit = [rec.cb blitCommandEncoderWithDescriptor:pass];
        rec.encoder = ENC_BLIT;
    }
    return rec.blit;
}
void vk_barrier(void)
{
    pc_cost_count(PC_COUNT_BARRIERS, 1);
    /* Between encoders Metal's tracking orders the work. Inside one: */
#if TARGET_OS_OSX /* (iOS has no render encoder barriers: render passes end before what reads them) */
    if (rec.encoder == ENC_RENDER)
        [rec.render memoryBarrierWithScope:MTLBarrierScopeTextures | MTLBarrierScopeBuffers | MTLBarrierScopeRenderTargets
                               afterStages:MTLRenderStageFragment | MTLRenderStageVertex
                              beforeStages:MTLRenderStageFragment | MTLRenderStageVertex];
    else
#endif
    if (rec.encoder == ENC_COMPUTE)
        [rec.compute memoryBarrierWithScope:MTLBarrierScopeTextures | MTLBarrierScopeBuffers];
}
int vk_rendering_open(void) { return rendering; }
void vk_end_rendering(void)
{
    if (!rendering)
        return;
    rendering = 0;
    end_encoder();
}
void vk_begin_rendering(const vk_attachments* a)
{
    if (rendering && !memcmp(a, &rec.attachments, sizeof *a))
        return;
    vk_end_rendering();
    end_encoder();
    vk_cmd();
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    rec.attachment_samples = 1;
    for (int i = 0; i < 6; i++)
        rec.attachment_formats[i] = VK_FORMAT_UNDEFINED;
    for (int i = 0; i < a->color_count; i++) {
        mtl_view* v = (mtl_view*)a->colors[i];
        pass.colorAttachments[i].texture = v->image->texture;
        pass.colorAttachments[i].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[i].storeAction = MTLStoreActionStore;
        rec.attachment_samples = v->image->samples;
        rec.attachment_formats[i] = v->image->format;
    }
    if (a->depth) {
        mtl_view* v = (mtl_view*)a->depth;
        pass.depthAttachment.texture = v->image->texture;
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        rec.attachment_samples = v->image->samples;
        rec.attachment_formats[5] = v->image->format;
    }
    pass.renderTargetWidth = (NSUInteger)a->width;
    pass.renderTargetHeight = (NSUInteger)a->height;
    int t = timer_slot();
    if (t >= 0) {
        pass.sampleBufferAttachments[0].sampleBuffer = timers[vk.frame].samples;
        pass.sampleBufferAttachments[0].startOfVertexSampleIndex = (NSUInteger)(2 * t);
        pass.sampleBufferAttachments[0].endOfVertexSampleIndex = MTLCounterDontSample;
        pass.sampleBufferAttachments[0].startOfFragmentSampleIndex = MTLCounterDontSample;
        pass.sampleBufferAttachments[0].endOfFragmentSampleIndex = (NSUInteger)(2 * t + 1);
    }
    rec.render = [rec.cb renderCommandEncoderWithDescriptor:pass];
    rec.encoder = ENC_RENDER;
    rec.applied = NULL;
    rec.applied_depth = nil;
    rec.attachments = *a;
    rendering = 1;
    pc_cost_count(PC_COUNT_PASSES, 1);
}

static void run_frame_done(int slot)
{
    typeof(frames[0])* f = &frames[slot];
    for (int i = 0; i < f->callback_count; i++)
        f->callbacks[i].callback(f->callbacks[i].context);
    f->callback_count = 0;
    for (int i = 0; i < f->destroy_count; i++) {
        if (f->destroy[i].kind == 0)
            vk_buffer_destroy(&f->destroy[i].buffer);
        else
            vk_image_destroy(&f->destroy[i].image);
    }
    f->destroy_count = 0;
}
void vk_on_frame_done(vk_frame_callback callback, void* context)
{
    typeof(frames[0])* f = &frames[vk.frame];
    if (f->callback_count == f->callback_capacity) {
        int capacity = f->callback_capacity ? f->callback_capacity * 2 : 8;
        frame_callback* grown = realloc(f->callbacks, (size_t)capacity * sizeof *grown);
        if (!grown) {
            callback(context);
            return;
        }
        f->callbacks = grown;
        f->callback_capacity = capacity;
    }
    f->callbacks[f->callback_count].callback = callback;
    f->callbacks[f->callback_count++].context = context;
}
static void submit(void)
{
    typeof(frames[0])* f = &frames[vk.frame];
    vk_cmd(); /* an empty frame still submits (its slot is waited for) */
    vk_end_rendering();
    end_encoder();
    dispatch_semaphore_t done = f->done;
    double* gpu_ms = &f->gpu_ms;
    [rec.cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.status == MTLCommandBufferStatusError)
            fprintf(stderr, "[mtl] command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
        /* The GPU time this frame added: frames overlap (the next starts
         * under the previous one's last passes), so from the later of its
         * start and the previous frame's end. Handlers run in commit order. */
        static double previous_end;
        double start = cb.GPUStartTime > previous_end ? cb.GPUStartTime : previous_end;
        *gpu_ms = cb.GPUEndTime > start ? (cb.GPUEndTime - start) * 1000.0 : 0.0;
        previous_end = cb.GPUEndTime;
        dispatch_semaphore_signal(done);
    }];
    [rec.cb commit];
    [last_committed release];
    last_committed = rec.cb; /* keeps the reference */
    rec.cb = nil;
    f->submitted = 1;
    ++vk.serial;
}
static void wait_slot(int slot)
{
    typeof(frames[0])* f = &frames[slot];
    uint64_t cost = pc_cost_begin();
    if (f->submitted)
        dispatch_semaphore_wait(f->done, DISPATCH_TIME_FOREVER);
    pc_cost_end(PC_COST_FENCE, cost);
    f->submitted = 0;
}
void vk_flush_wait(void)
{
    submit();
    wait_slot(vk.frame);
    typeof(frames[0])* f = &frames[vk.frame];
    for (int i = 0; i < f->destroy_count; i++) {
        if (f->destroy[i].kind == 0)
            vk_buffer_destroy(&f->destroy[i].buffer);
        else
            vk_image_destroy(&f->destroy[i].image);
    }
    f->destroy_count = 0;
}
void vk_section(const char* name)
{
    if (perf <= 0)
        return;
    vk_end_rendering(); /* (as vk_core.c: a section starts with a new encoder) */
    end_encoder();
    current_section = name;
}
/* RPC perf: the sections' GPU time per frame since the last reset (vk_core.c's format). */
int pc_vk_perf_sections_json(char* out, int size, int reset)
{
    int n = 0;
    if (section_rpc_frames && size > 16) {
        n += snprintf(out, (size_t)size, ",\"sections\":{");
        for (int k = 0; k < section_rpc_count && n < size - 64; k++)
            n += snprintf(out + n, (size_t)(size - n), "%s\"%s\":%.4f", k ? "," : "", section_rpc[k].name,
                section_rpc[k].sum / section_rpc_frames);
        n += snprintf(out + n, (size_t)(size - n), "}");
    } else if (size > 0)
        out[0] = 0;
    if (reset)
        section_rpc_count = section_rpc_frames = 0;
    return n < size ? n : size - 1;
}
extern void pc_perf_gpu_add(double ms) __attribute__((weak)); /* lighting_quality.cpp: RPC perf */
void vk_frame_submit(VkSemaphore wait, VkPipelineStageFlags wait_stage, VkSemaphore signal)
{
    (void)wait, (void)wait_stage, (void)signal; /* one queue: commit order is the order */
    submit();
    vk.frame = (vk.frame + 1) % VK_FRAMES;
    wait_slot(vk.frame);
    typeof(frames[0])* f = &frames[vk.frame];
    if (perf > 0 && vk.serial > VK_FRAMES) {
        double ms = f->gpu_ms;
        gpu_sum += ms;
        if (pc_perf_gpu_add)
            pc_perf_gpu_add(ms);
        gpu_max = ms > gpu_max ? ms : gpu_max;
        if (++gpu_count == 300) {
            fprintf(stderr, "[perf] Metal GPU time per presented frame avg %.2f ms, max %.2f ms\n", gpu_sum / 300,
                gpu_max);
            if (section_total_count) {
                char line[1024];
                int used = snprintf(line, sizeof line, "[perf] GPU sections per frame:");
                for (int k = 0; k < section_total_count && used < (int)sizeof line - 64; k++)
                    used += snprintf(line + used, sizeof line - (size_t)used, " %s %.3f ms (%.1f passes)%s",
                        section_totals[k].name, section_totals[k].sum / 300, section_totals[k].encoders / 300.0,
                        k + 1 < section_total_count ? "," : "");
                fprintf(stderr, "%s\n", line);
                section_total_count = 0;
            }
            gpu_sum = gpu_max = 0;
            gpu_count = 0;
        }
    }
    read_timers(vk.frame);
    current_section = "draw"; /* the frame's drawing until its first section */
    run_frame_done(vk.frame);
    for (int i = 0; i < f->chunk_count; i++)
        f->chunks[i].used = 0;
    f->chunk_current = 0;
}

/* ---- memory ---- */

/* Buffers by GPU address: ray tracing passes geometry, instances and
 * scratch memory as device addresses (vk_rt.inc); a build needs the buffer. */
static mtl_buffer** addressed;
static int addressed_count, addressed_capacity;
static void register_address(mtl_buffer* m)
{
    if (addressed_count == addressed_capacity) {
        addressed_capacity = addressed_capacity ? addressed_capacity * 2 : 256;
        addressed = realloc(addressed, (size_t)addressed_capacity * sizeof *addressed);
    }
    addressed[addressed_count++] = m;
}
static void unregister_address(mtl_buffer* m)
{
    for (int i = addressed_count; i-- > 0;)
        if (addressed[i] == m) {
            addressed[i] = addressed[--addressed_count];
            return;
        }
}
static mtl_buffer* buffer_at(VkDeviceAddress address, NSUInteger* offset)
{
    for (int i = addressed_count; i-- > 0;) {
        uint64_t base = addressed[i]->buffer.gpuAddress;
        if (address >= base && address < base + addressed[i]->buffer.length) {
            *offset = (NSUInteger)(address - base);
            return addressed[i];
        }
    }
    *offset = 0;
    return NULL;
}
int vk_buffer_create(vk_buffer* b, VkDeviceSize size, VkBufferUsageFlags usage, int host)
{
    (void)usage;
    memset(b, 0, sizeof *b);
    /* Unified memory: host-visible buffers are shared; device-only private. */
    MTLResourceOptions options = host ? MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache
                                      : MTLResourceStorageModePrivate;
    if (host == 1)
        options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    id<MTLBuffer> buffer = [device newBufferWithLength:(NSUInteger)(size ? size : 16) options:options];
    if (!buffer) {
        fprintf(stderr, "[mtl] cannot allocate a %llu-byte buffer\n", (unsigned long long)size);
        return 0;
    }
    mtl_buffer* m = calloc(1, sizeof *m);
    m->buffer = buffer;
    m->size = size;
    register_address(m);
    b->buffer = (VkBuffer)m;
    b->memory = (VkDeviceMemory)(uintptr_t)1;
    b->size = size;
    b->mapped = host ? buffer.contents : NULL;
    b->address = buffer.gpuAddress;
    return 1;
}
void vk_buffer_destroy(vk_buffer* b)
{
    if (b->buffer) {
        mtl_buffer* m = (mtl_buffer*)b->buffer;
        unregister_address(m);
        [m->buffer release];
        free(m);
    }
    memset(b, 0, sizeof *b);
}
static deferred* push_deferred(void)
{
    typeof(frames[0])* f = &frames[vk.frame];
    if (f->destroy_count == f->destroy_capacity) {
        int capacity = f->destroy_capacity ? f->destroy_capacity * 2 : 64;
        deferred* grown = realloc(f->destroy, (size_t)capacity * sizeof *grown);
        if (!grown)
            return NULL;
        f->destroy = grown;
        f->destroy_capacity = capacity;
    }
    return &f->destroy[f->destroy_count++];
}
void vk_buffer_destroy_later(vk_buffer* b)
{
    if (!b->buffer)
        return;
    deferred* d = push_deferred();
    if (!d) {
        vkDeviceWaitIdle(vk.device);
        vk_buffer_destroy(b);
        return;
    }
    d->kind = 0;
    d->buffer = *b;
    memset(b, 0, sizeof *b);
}

static int make_image(vk_image* image, MTLTextureDescriptor* desc, VkFormat format, int samples, VkImageUsageFlags usage)
{
    memset(image, 0, sizeof *image);
    int depth = format == VK_FORMAT_D32_SFLOAT;
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                 | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        desc.usage |= MTLTextureUsageRenderTarget; /* (transfer destinations: clears and blits draw) */
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
        desc.usage |= MTLTextureUsageShaderWrite;
    id<MTLBuffer> backing = nil;
    id<MTLTexture> texture;
    if (format == VK_FORMAT_R32_UINT && (usage & VK_IMAGE_USAGE_STORAGE_BIT) && desc.textureType == MTLTextureType2D) {
        /* SPIRV-Cross's image atomics (atomic_uint over the image's memory): a linear texture on a buffer. */
        NSUInteger row = (desc.width * 4 + MTL_LINEAR_ALIGN - 1) & ~(NSUInteger)(MTL_LINEAR_ALIGN - 1);
        backing = [device newBufferWithLength:row * desc.height options:MTLResourceStorageModePrivate];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage &= ~MTLTextureUsageRenderTarget;
        texture = [backing newTextureWithDescriptor:desc offset:0 bytesPerRow:row];
    } else
        texture = [device newTextureWithDescriptor:desc];
    if (!texture) {
        fprintf(stderr, "[mtl] cannot make a %dx%d texture (VkFormat %d)\n", (int)desc.width, (int)desc.height,
            (int)format);
        return 0;
    }
    mtl_image* m = calloc(1, sizeof *m);
    m->texture = texture;
    m->backing = backing;
    m->format = format;
    m->depth = depth;
    m->samples = samples;
    mtl_view* v = calloc(1, sizeof *v);
    v->image = m;
    image->image = (VkImage)m;
    image->memory = (VkDeviceMemory)(uintptr_t)1;
    image->view = (VkImageView)v;
    image->format = format;
    image->aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    image->width = (int)desc.width;
    image->height = (int)desc.height;
    return 1;
}
static void clear_texture(mtl_image* m);
int vk_image_create(vk_image* image, int width, int height, VkFormat format, VkImageUsageFlags usage)
{
    return vk_image_create_ms(image, width, height, format, usage, 1);
}
int vk_image_create_ms(vk_image* image, int width, int height, VkFormat format, VkImageUsageFlags usage, int samples)
{
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format(format)
                                                                                    width:(NSUInteger)width
                                                                                   height:(NSUInteger)height
                                                                                mipmapped:NO];
    if (samples > 1) {
        desc.textureType = MTLTextureType2DMultisample;
        desc.sampleCount = (NSUInteger)samples;
    }
    if (!make_image(image, desc, format, samples > 1 ? samples : 1, usage))
        return 0;
    /* As vk_core.c: an image starts cleared (colour 0, depth far). */
    clear_texture((mtl_image*)image->image);
    return 1;
}
int vk_image_create_3d(vk_image* image, int width, int height, int depth, VkFormat format, VkImageUsageFlags usage)
{
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor new] autorelease];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = pixel_format(format);
    desc.width = (NSUInteger)width;
    desc.height = (NSUInteger)height;
    desc.depth = (NSUInteger)depth;
    return make_image(image, desc, format, 1, usage & ~(VkImageUsageFlags)VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}
static void destroy_view(mtl_view* v) { free(v); }
static void destroy_image(mtl_image* m)
{
    [m->texture release];
    [m->backing release];
    free(m);
}
void vk_image_destroy(vk_image* image)
{
    if (image->view)
        destroy_view((mtl_view*)image->view);
    if (image->image)
        destroy_image((mtl_image*)image->image);
    memset(image, 0, sizeof *image);
}
void vk_image_destroy_later(vk_image* image)
{
    if (!image->image)
        return;
    deferred* d = push_deferred();
    if (!d) {
        vkDeviceWaitIdle(vk.device);
        vk_image_destroy(image);
        return;
    }
    d->kind = 1;
    d->image = *image;
    memset(image, 0, sizeof *image);
}

/* ---- arena ---- */

vk_slice vk_arena(VkDeviceSize size, VkDeviceSize alignment)
{
    typeof(frames[0])* f = &frames[vk.frame];
    vk_slice slice = { VK_NULL_HANDLE, 0, NULL, 0 };
    if (alignment < 16)
        alignment = 16;
    for (;;) {
        if (f->chunk_current < f->chunk_count) {
            arena_chunk* c = &f->chunks[f->chunk_current];
            VkDeviceSize start = (c->used + alignment - 1) & ~(alignment - 1);
            if (start + size <= c->buffer.size) {
                c->used = start + size;
                slice.buffer = c->buffer.buffer;
                slice.offset = start;
                slice.data = (char*)c->buffer.mapped + start;
                slice.address = c->buffer.address + start;
                return slice;
            }
            ++f->chunk_current;
            continue;
        }
        arena_chunk* grown = realloc(f->chunks, (size_t)(f->chunk_count + 1) * sizeof *grown);
        if (!grown)
            return slice;
        f->chunks = grown;
        arena_chunk* c = &f->chunks[f->chunk_count];
        VkDeviceSize chunk = size + alignment > ARENA_CHUNK ? size + alignment : ARENA_CHUNK;
        if (!vk_buffer_create(&c->buffer, chunk, 0, 1)) {
            fprintf(stderr, "[mtl] out of arena memory (%llu bytes)\n", (unsigned long long)size);
            return slice;
        }
        c->used = 0;
        f->chunk_current = f->chunk_count++;
    }
}

/* ---- utility pipelines: clears and scaled copies ----
 * Metal clears inside a render encoder by drawing, and has no scaled blit:
 * small pipelines made from source at run time, per target format. */

static id<MTLLibrary> utility;
static const char* const UTILITY_SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct V { float4 p [[position]]; float2 uv; };\n"
    "vertex V utility_quad(uint i [[vertex_id]], constant float4& rect [[buffer(0)]], constant float4& uv [[buffer(1)]]) {\n"
    "    float2 c = float2(i & 1, i >> 1);\n"
    "    V v; v.p = float4(mix(rect.xy, rect.zw, c), 0.0, 1.0); v.uv = mix(uv.xy, uv.zw, c); return v; }\n"
    "struct DepthOut { float d [[depth(any)]]; };\n"
    "fragment DepthOut clear_depth(V v [[stage_in]], constant float4& value [[buffer(0)]]) {\n"
    "    DepthOut o; o.d = value.x; return o; }\n"
    "fragment float4 clear_float(V v [[stage_in]], constant float4& value [[buffer(0)]]) { return value; }\n"
    "fragment uint4 clear_uint(V v [[stage_in]], constant uint4& value [[buffer(0)]]) { return value; }\n"
    "fragment int4 clear_int(V v [[stage_in]], constant int4& value [[buffer(0)]]) { return value; }\n"
    "fragment float4 copy_float(V v [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    "    return t.sample(s, v.uv); }\n"
    "fragment uint4 copy_uint(V v [[stage_in]], texture2d<uint> t [[texture(0)]]) {\n"
    "    return t.read(uint2(v.uv * float2(t.get_width(), t.get_height()))); }\n"
    /* Dear ImGui's draw lists (vk_overlay.cpp): its vertex, an orthographic projection of the display. */
    "struct ImVert { packed_float2 pos; packed_float2 uv; uint col; };\n"
    "struct OV { float4 p [[position]]; float2 uv; float4 col; };\n"
    "vertex OV overlay_vertex(uint i [[vertex_id]], const device ImVert* v [[buffer(0)]], constant float4& display [[buffer(1)]]) {\n"
    "    OV o; float2 p = (float2(v[i].pos) - display.xy) / display.zw;\n"
    "    o.p = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0); o.uv = float2(v[i].uv);\n"
    "    o.col = unpack_unorm4x8_to_float(v[i].col); return o; }\n"
    "fragment float4 overlay_fragment(OV o [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    "    return o.col * t.sample(s, o.uv); }\n";
static id<MTLFunction> utility_function(const char* name)
{
    if (!utility) {
        NSError* error = nil;
        utility = [device newLibraryWithSource:@(UTILITY_SOURCE) options:nil error:&error];
        if (!utility) {
            fprintf(stderr, "[mtl] utility shaders: %s\n", error.localizedDescription.UTF8String);
            abort();
        }
    }
    return [[utility newFunctionWithName:@(name)] autorelease];
}
/* A utility pipeline drawing into attachment `target` only (-1: depth only)
 * of the attachment set `formats` ([5] depth) at `samples`. */
typedef struct {
    VkFormat formats[6];
    int samples, target, kind; /* kind: 0 clear, 1 copy (linear), 2 copy (nearest) */
    id<MTLRenderPipelineState> state;
} utility_entry;
static utility_entry* utilities;
static int utility_count;
static id<MTLRenderPipelineState> utility_pipeline(const VkFormat formats[6], int samples, int target, int kind)
{
    for (int i = 0; i < utility_count; i++)
        if (!memcmp(utilities[i].formats, formats, sizeof utilities[i].formats) && utilities[i].samples == samples
            && utilities[i].target == target && utilities[i].kind == kind)
            return utilities[i].state;
    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor new] autorelease];
    d.vertexFunction = utility_function("utility_quad");
    VkFormat f = target >= 0 ? formats[target] : VK_FORMAT_UNDEFINED;
    const char* fragment = target < 0 ? "clear_depth"
        : kind ? (format_is_integer(f) ? "copy_uint" : "copy_float")
        : f == VK_FORMAT_R16G16_SINT ? "clear_int"
        : format_is_integer(f) ? "clear_uint"
                               : "clear_float";
    d.fragmentFunction = utility_function(fragment);
    for (int i = 0; i < 5; i++)
        if (formats[i] != VK_FORMAT_UNDEFINED) {
            d.colorAttachments[i].pixelFormat = pixel_format(formats[i]);
            d.colorAttachments[i].writeMask = i == target ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
        }
    if (formats[5] != VK_FORMAT_UNDEFINED)
        d.depthAttachmentPixelFormat = pixel_format(formats[5]);
    d.rasterSampleCount = (NSUInteger)(samples > 1 ? samples : 1);
    NSError* error = nil;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) {
        fprintf(stderr, "[mtl] utility pipeline: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }
    utilities = realloc(utilities, (size_t)(utility_count + 1) * sizeof *utilities);
    utility_entry* e = &utilities[utility_count++];
    memcpy(e->formats, formats, sizeof e->formats);
    e->samples = samples, e->target = target, e->kind = kind, e->state = state;
    return state;
}
static id<MTLDepthStencilState> depth_state(int test, int write, VkCompareOp op);
/* Draw the quad (rect: pixels of the attachment, uv: the source's 0..1). */
static void utility_draw(id<MTLRenderCommandEncoder> e, int width, int height, const VkRect2D* r, const float uv[4])
{
    float rect[4] = { 2.0f * r->offset.x / width - 1.0f, 1.0f - 2.0f * r->offset.y / height,
        2.0f * (r->offset.x + (int)r->extent.width) / width - 1.0f,
        1.0f - 2.0f * (r->offset.y + (int)r->extent.height) / height };
    static const float whole[4] = { 0, 0, 1, 1 };
    [e setViewport:(MTLViewport){ 0, 0, (double)width, (double)height, 0, 1 }];
    [e setScissorRect:(MTLScissorRect){ 0, 0, (NSUInteger)width, (NSUInteger)height }];
    [e setVertexBytes:rect length:sizeof rect atIndex:0];
    [e setVertexBytes:uv ? uv : whole length:sizeof whole atIndex:1];
    [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}
static void restore_render_state(void);
/* vkCmdClearAttachments inside the open scope: one quad per attachment and rectangle. */
static void clear_in_scope(const VkClearAttachment* a, int count, const VkClearRect* rects, int rect_count)
{
    id<MTLRenderCommandEncoder> e = render_encoder();
    if (!e)
        return;
    for (int i = 0; i < count; i++) {
        int target = (a[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) ? (int)a[i].colorAttachment : -1;
        id<MTLRenderPipelineState> state = utility_pipeline(rec.attachment_formats, rec.attachment_samples, target, 0);
        if (!state)
            continue;
        [e setRenderPipelineState:state];
        [e setDepthStencilState:depth_state(target < 0, target < 0, VK_COMPARE_OP_ALWAYS)];
        if (target < 0) {
            float value[4] = { a[i].clearValue.depthStencil.depth, 0, 0, 0 };
            [e setFragmentBytes:value length:sizeof value atIndex:0];
        } else
            [e setFragmentBytes:&a[i].clearValue.color length:16 atIndex:0];
        for (int r = 0; r < rect_count; r++)
            utility_draw(e, rec.attachments.width, rec.attachments.height, &rects[r].rect, NULL);
    }
    restore_render_state();
}
/* A whole image cleared outside any scope (vk_image_create, vkCmdClearColorImage). */
static void clear_texture_value(mtl_image* m, const VkClearColorValue* color, float depth)
{
    if (m->backing) { /* a linear view of a buffer: fill the buffer (a byte pattern: 0 and ~0 are) */
        uint32_t v = color->uint32[0];
        if ((v & 0xff) * 0x01010101u != v)
            fprintf(stderr, "[mtl] clearing an atomic image to %08x: filled with its low byte\n", v);
        [blit_encoder() fillBuffer:m->backing range:NSMakeRange(0, m->backing.length) value:(uint8_t)v];
        return;
    }
    if (!(m->texture.usage & MTLTextureUsageRenderTarget) || m->texture.textureType == MTLTextureType3D)
        return;
    if (rendering)
        vk_end_rendering();
    end_encoder();
    vk_cmd();
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (m->depth) {
        pass.depthAttachment.texture = m->texture;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = depth;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
    } else {
        pass.colorAttachments[0].texture = m->texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        if (format_is_integer(m->format))
            pass.colorAttachments[0].clearColor = MTLClearColorMake(color->uint32[0], color->uint32[1],
                color->uint32[2], color->uint32[3]);
        else
            pass.colorAttachments[0].clearColor = MTLClearColorMake(color->float32[0], color->float32[1],
                color->float32[2], color->float32[3]);
    }
    [[rec.cb renderCommandEncoderWithDescriptor:pass] endEncoding];
}
static void clear_texture(mtl_image* m)
{
    VkClearColorValue zero = { { 0, 0, 0, 0 } };
    clear_texture_value(m, &zero, 1.0f);
}

/* ---- device ---- */

extern const char* pc_metal_library_path(void) __attribute__((weak));
static int load_library(void)
{
    char path[2048];
    uint32_t size = sizeof path;
    const char* env = getenv("ROOD_METALLIB");
    if (env)
        snprintf(path, sizeof path, "%s", env);
    else if (_NSGetExecutablePath(path, &size) == 0) {
        char* slash = strrchr(path, '/');
        snprintf(slash ? slash + 1 : path, sizeof path - (size_t)(slash ? slash + 1 - path : 0), "rood.metallib");
    } else
        return 0;
    NSError* error = nil;
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
    if (!library)
        fprintf(stderr, "[mtl] cannot load %s: %s\n", path, error.localizedDescription.UTF8String);
    return library != nil;
}
static void install_entry_points(void);
int vk_core_init(const char* const* instance_extensions, uint32_t extension_count)
{
    (void)instance_extensions, (void)extension_count;
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "[mtl] no Metal device\n");
        return 0;
    }
    if (![device supportsFamily:MTLGPUFamilyMetal3]) {
        fprintf(stderr, "[mtl] %s lacks Metal 3\n", device.name.UTF8String);
        return 0;
    }
    if (!load_library())
        return 0;
    queue = [device newCommandQueue];
    /* Metal ray tracing (Apple GPUs from the M3/A17 in hardware); ROOD_VK_RT=0 keeps it off, as on Vulkan. */
    const char* rt = getenv("ROOD_VK_RT");
    vk.ray_query = device.supportsRaytracing && !(rt && *rt == '0');
    vk.scratch_alignment = 256;
    install_entry_points();
    vk.instance = (VkInstance)(uintptr_t)1;
    vk.gpu = (VkPhysicalDevice)(uintptr_t)1;
    vk.device = (VkDevice)(uintptr_t)1;
    vk.queue = (VkQueue)(uintptr_t)1;
    vk.api_version = VK_API_VERSION_1_3;
    snprintf(vk.properties.deviceName, sizeof vk.properties.deviceName, "%s", device.name.UTF8String);
    vk.properties.deviceType = device.hasUnifiedMemory ? VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
                                                       : VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    vk.properties.vendorID = 0x106b;
    vk.properties.limits.minUniformBufferOffsetAlignment = 256;
    vk.properties.limits.minStorageBufferOffsetAlignment = 16;
    vk.properties.limits.maxPerStageDescriptorSamplers = 2048; /* argument buffers (tier 2) */
    vk.properties.limits.timestampPeriod = 1.0f;
    vk.timestamp_period = 1.0f;
    vk.geometry_shader = 0;
    vk.depth_format = VK_FORMAT_D32_SFLOAT;
    vk.msaa_counts = 1;
    for (int s = 2; s <= 8; s *= 2)
        if ([device supportsTextureSampleCount:(NSUInteger)s])
            vk.msaa_counts |= (VkSampleCountFlags)s;
    vk.sample_locations = 0;
    vk.sample_location_counts = 0;
    for (int i = 0; i < VK_FRAMES; i++)
        frames[i].done = dispatch_semaphore_create(0);
    MTLSamplerDescriptor* s = [[MTLSamplerDescriptor new] autorelease];
    s.sAddressMode = s.tAddressMode = s.rAddressMode = MTLSamplerAddressModeClampToEdge;
    s.supportArgumentBuffers = YES;
    mtl_sampler* nearest = calloc(1, sizeof *nearest);
    nearest->state = [device newSamplerStateWithDescriptor:s];
    s.minFilter = s.magFilter = MTLSamplerMinMagFilterLinear;
    mtl_sampler* linear = calloc(1, sizeof *linear);
    linear->state = [device newSamplerStateWithDescriptor:s];
    vk.nearest = (VkSampler)nearest;
    vk.linear = (VkSampler)linear;
    fprintf(stderr, "[mtl] %s, Metal 3, %d shaders\n", device.name.UTF8String, mtl_shader_count);
    return 1;
}
void vk_core_shutdown(void)
{
    if (!device)
        return;
    vkDeviceWaitIdle(vk.device);
    for (int i = 0; i < VK_FRAMES; i++)
        run_frame_done(i);
}
void vk_core_teardown(VkSurfaceKHR surface)
{
    (void)surface;
    vk_core_shutdown();
    memset(&vk, 0, sizeof vk);
    pc_vk_active = 0;
}
void vk_pipeline_cache_note(uint64_t ns) { (void)ns; }
int pc_vk_device_lost(void) { return vk.lost; }
void vk_device_lost(const char* where)
{
    if (vk.lost)
        return;
    vk.lost = 1;
    fprintf(stderr, "[mtl] device lost (%s)\n", where);
}
int vk_check(VkResult result, const char* what)
{
    if (result != VK_SUCCESS)
        fprintf(stderr, "[mtl] %s failed (%d)\n", what, (int)result);
    return result == VK_SUCCESS;
}
int pc_vk_caps(int* ray_query, int* integrated, char* name, int size)
{
    if (!device)
        return 0;
    *ray_query = vk.ray_query;
    *integrated = device.hasUnifiedMemory;
    snprintf(name, (size_t)size, "%s", device.name.UTF8String);
    return 1;
}

/* ---- pipelines ---- */

VkShaderModule vk_shader(const uint32_t* code, size_t bytes)
{
    /* The table's hash: SHA-256's first 8 bytes. */
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(code, (CC_LONG)bytes, digest);
    uint64_t hash;
    memcpy(&hash, digest, sizeof hash);
    for (int i = 0; i < mtl_shader_count; i++)
        if (mtl_shaders[i].spirv_bytes == bytes && mtl_shaders[i].spirv_hash == hash) {
            mtl_module* m = calloc(1, sizeof *m);
            m->info = &mtl_shaders[i];
            m->function = [library newFunctionWithName:@(mtl_shaders[i].function)];
            if (m->function && m->function.functionConstantsDictionary.count) {
                /* SPIRV-Cross's optional constants (spvLinearTextureAlignmentOverride): unset, the defaults. */
                NSError* error = nil;
                MTLFunctionConstantValues* none = [[MTLFunctionConstantValues new] autorelease];
                id<MTLFunction> specialized = [library newFunctionWithName:@(mtl_shaders[i].function)
                                                            constantValues:none
                                                                     error:&error];
                [m->function release];
                m->function = specialized;
                if (!specialized)
                    fprintf(stderr, "[mtl] %s: %s\n", mtl_shaders[i].function, error.localizedDescription.UTF8String);
            }
            if (!m->function) {
                fprintf(stderr, "[mtl] %s is not in rood.metallib\n", mtl_shaders[i].function);
                free(m);
                return VK_NULL_HANDLE;
            }
            return (VkShaderModule)m;
        }
    fprintf(stderr, "[mtl] a shader (%zu bytes) is not in the Metal table: rebuild it (pc/tools/metal_shaders.py)\n",
        bytes);
    return VK_NULL_HANDLE;
}
VkDescriptorSetLayout vk_push_layout(const VkDescriptorType* types, int count)
{
    if (count > 32)
        return VK_NULL_HANDLE;
    mtl_set_layout* l = calloc(1, sizeof *l);
    l->count = count;
    memcpy(l->types, types, (size_t)count * sizeof *types);
    return (VkDescriptorSetLayout)l;
}
static void blend_state(MTLRenderPipelineColorAttachmentDescriptor* c, int blend, unsigned mask)
{
    c.writeMask = (mask & 1 ? MTLColorWriteMaskRed : 0) | (mask & 2 ? MTLColorWriteMaskGreen : 0)
                | (mask & 4 ? MTLColorWriteMaskBlue : 0) | (mask & 8 ? MTLColorWriteMaskAlpha : 0);
    if (blend < 0 || !mask)
        return;
    c.blendingEnabled = YES;
    c.rgbBlendOperation = c.alphaBlendOperation = blend == 2 || blend == 6 ? MTLBlendOperationReverseSubtract
                                                                           : MTLBlendOperationAdd;
    switch (blend) {
    case 0: /* B/2 + F/2 */
        c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = MTLBlendFactorBlendAlpha;
        c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = MTLBlendFactorBlendAlpha;
        break;
    case 3: /* B + F/4 */
        c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = MTLBlendFactorBlendAlpha;
        c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = MTLBlendFactorOne;
        break;
    case 4: /* over */
        c.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        c.sourceAlphaBlendFactor = MTLBlendFactorOne;
        c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        break;
    case 5: /* B + F * alpha */
    case 6: /* B - F * alpha */
        c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = MTLBlendFactorOne;
        break;
    default: /* B + F, B - F */
        c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = MTLBlendFactorOne;
        c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = MTLBlendFactorOne;
        break;
    }
}
VkPipeline vk_pipeline(const vk_pipeline_desc* d)
{
    mtl_module *vs = (mtl_module*)d->vertex, *fs = (mtl_module*)d->fragment;
    if (!vs || d->geometry) {
        fprintf(stderr, "[mtl] a pipeline without a vertex shader, or with a geometry shader\n");
        return VK_NULL_HANDLE;
    }
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor new] autorelease];
    desc.vertexFunction = vs->function;
    desc.fragmentFunction = fs ? fs->function : nil;
    if (d->binding) {
        MTLVertexDescriptor* input = [MTLVertexDescriptor vertexDescriptor];
        for (int i = 0; i < d->attribute_count; i++) {
            const VkVertexInputAttributeDescription* a = &d->attributes[i];
            input.attributes[a->location].format = vertex_format(a->format);
            input.attributes[a->location].offset = a->offset;
            input.attributes[a->location].bufferIndex = VERTEX_BUFFER_INDEX;
        }
        input.layouts[VERTEX_BUFFER_INDEX].stride = d->binding->stride;
        input.layouts[VERTEX_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerVertex;
        desc.vertexDescriptor = input;
    }
    for (int i = 0; i < d->color_count; i++) {
        desc.colorAttachments[i].pixelFormat = pixel_format(d->color_formats[i]);
        blend_state(desc.colorAttachments[i], format_is_integer(d->color_formats[i]) ? -1 : d->blend,
            d->write_masks[i]);
    }
    if (d->depth_format != VK_FORMAT_UNDEFINED)
        desc.depthAttachmentPixelFormat = pixel_format(d->depth_format);
    desc.rasterSampleCount = (NSUInteger)(d->samples > 1 ? d->samples : 1);
    switch (d->topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle; break;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine; break;
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint; break;
    default: desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle; break;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!state) {
        fprintf(stderr, "[mtl] pipeline %s + %s: %s\n", vs->info->function, fs ? fs->info->function : "-",
            error.localizedDescription.UTF8String);
        return VK_NULL_HANDLE;
    }
    mtl_pipeline* p = calloc(1, sizeof *p);
    p->render = state;
    p->vs = vs->info;
    p->fs = fs ? fs->info : NULL;
    p->has_depth = d->depth_format != VK_FORMAT_UNDEFINED;
    p->primitive = d->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ? MTLPrimitiveTypeTriangleStrip
                 : d->topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST     ? MTLPrimitiveTypeLine
                 : d->topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST    ? MTLPrimitiveTypePoint
                                                                      : MTLPrimitiveTypeTriangle;
    p->blend_constant = d->blend == 3 ? 0.25f : 0.5f;
    return (VkPipeline)p;
}
VkPipeline vk_compute_pipeline(VkShaderModule module, VkPipelineLayout layout)
{
    (void)layout;
    mtl_module* m = (mtl_module*)module;
    if (!m)
        return VK_NULL_HANDLE;
    NSError* error = nil;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:m->function error:&error];
    if (!state) {
        fprintf(stderr, "[mtl] compute pipeline %s: %s\n", m->info->function, error.localizedDescription.UTF8String);
        return VK_NULL_HANDLE;
    }
    mtl_pipeline* p = calloc(1, sizeof *p);
    p->compute = 1;
    p->kernel = state;
    p->cs = m->info;
    return (VkPipeline)p;
}

/* ---- descriptors ---- */

static VkWriteDescriptorSet* next_write(vk_descriptors* d, int binding, VkDescriptorType type)
{
    if (d->count >= (int)(sizeof d->writes / sizeof d->writes[0])) {
        fprintf(stderr, "[mtl] more than %d descriptor writes\n", d->count);
        abort();
    }
    VkWriteDescriptorSet* w = &d->writes[d->count];
    *w = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = (uint32_t)binding,
        .descriptorCount = 1,
        .descriptorType = type };
    return w;
}
void vk_descriptors_image(vk_descriptors* d, int binding, VkImageView view, VkSampler sampler)
{
    VkWriteDescriptorSet* w = next_write(d, binding,
        d->sampled_images ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    d->images[d->count] = (VkDescriptorImageInfo){ d->sampled_images ? VK_NULL_HANDLE : sampler, view,
        VK_IMAGE_LAYOUT_GENERAL };
    w->pImageInfo = &d->images[d->count++];
}
void vk_descriptors_sampler(vk_descriptors* d, int binding, VkSampler sampler)
{
    VkWriteDescriptorSet* w = next_write(d, binding, VK_DESCRIPTOR_TYPE_SAMPLER);
    d->images[d->count] = (VkDescriptorImageInfo){ sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    w->pImageInfo = &d->images[d->count++];
}
void vk_descriptors_storage_image(vk_descriptors* d, int binding, VkImageView view)
{
    VkWriteDescriptorSet* w = next_write(d, binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    d->images[d->count] = (VkDescriptorImageInfo){ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
    w->pImageInfo = &d->images[d->count++];
}
void vk_descriptors_buffer(vk_descriptors* d, int binding, VkDescriptorType type, VkBuffer buffer, VkDeviceSize offset,
    VkDeviceSize range)
{
    VkWriteDescriptorSet* w = next_write(d, binding, type);
    d->buffers[d->count] = (VkDescriptorBufferInfo){ buffer, offset, range };
    w->pBufferInfo = &d->buffers[d->count++];
}
void vk_descriptors_structure(vk_descriptors* d, int binding, VkAccelerationStructureKHR structure)
{
    VkWriteDescriptorSet* w = next_write(d, binding, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    int s = d->structure_count++;
    d->structure_handles[s] = structure;
    d->structures[s] = (VkWriteDescriptorSetAccelerationStructureKHR){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &d->structure_handles[s] };
    w->pNext = &d->structures[s];
    d->count++;
}
void vk_descriptors_push(vk_descriptors* d, VkPipelineBindPoint point, VkPipelineLayout layout)
{
    if (d->count > 0)
        vkCmdPushDescriptorSetKHR(vk_cmd(), point, layout, 0, (uint32_t)d->count, d->writes);
    d->count = 0;
    d->structure_count = 0;
}

/* Fill `info`'s argument buffer from the bound slots; declare its resources
 * to the encoder (residency, hazard tracking). Returns the arena slice. */
static int is_attachment(id<MTLTexture> t)
{
    if (!rendering)
        return 0;
    for (int i = 0; i < rec.attachments.color_count; i++)
        if (((mtl_view*)rec.attachments.colors[i])->image->texture == t)
            return 1;
    return rec.attachments.depth && ((mtl_view*)rec.attachments.depth)->image->texture == t;
}
/* ---- the Vulkan entry points the renderer calls ---- */

static id combined_samplers[2][32]; /* [graphics, compute][binding] */
static void push_descriptor_set(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t set,
    uint32_t count, const VkWriteDescriptorSet* writes)
{
    (void)cmd, (void)layout, (void)set;
    bind_point* bp = point == VK_PIPELINE_BIND_POINT_COMPUTE ? &rec.kernel : &rec.graphics;
    for (int i = 0; i < 32; i++) {
        [bp->slots[i].resource release];
        [bp->slots[i].bottoms release];
        [bp->slots[i].backing release];
        bp->slots[i] = (binding_slot){ 0 };
    }
    /* Combined image samplers: the texture at the binding's texture id, the
     * sampler at its sampler id. A slot keeps one resource: the sampler goes
     * in combined_samplers. */
    int which = point == VK_PIPELINE_BIND_POINT_COMPUTE;
    for (int i = 0; i < 32; i++) {
        [combined_samplers[which][i] release];
        combined_samplers[which][i] = nil;
    }
    for (uint32_t i = 0; i < count; i++) {
        const VkWriteDescriptorSet* w = &writes[i];
        if (w->dstBinding >= 32)
            continue;
        binding_slot* s = &bp->slots[w->dstBinding];
        s->type = w->descriptorType;
        s->set = 1;
        switch (w->descriptorType) {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            mtl_view* v = (mtl_view*)w->pImageInfo->imageView;
            s->resource = v ? [v->image->texture retain] : nil;
            s->backing = v ? [v->image->backing retain] : nil;
            if (w->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && w->pImageInfo->sampler)
                combined_samplers[which][w->dstBinding] = [((mtl_sampler*)w->pImageInfo->sampler)->state retain];
            break;
        }
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            s->resource = [((mtl_sampler*)w->pImageInfo->sampler)->state retain];
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
            mtl_buffer* b = (mtl_buffer*)w->pBufferInfo->buffer;
            s->resource = b ? [b->buffer retain] : nil;
            s->offset = w->pBufferInfo->offset;
            break;
        }
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
            const VkWriteDescriptorSetAccelerationStructureKHR* as = w->pNext;
            mtl_structure* st = as && as->accelerationStructureCount ? (mtl_structure*)as->pAccelerationStructures[0] : NULL;
            s->resource = st ? [st->structure retain] : nil;
            [s->bottoms release];
            s->bottoms = st && st->bottoms ? [st->bottoms retain] : nil;
            break;
        }
        default:
            s->set = 0;
            break;
        }
    }
}
/* `info`'s argument buffer from the pushed descriptors, its resources
 * declared to the encoder (residency and hazard tracking). */
static vk_slice arguments_for(const mtl_shader_info* info, int compute, id encoder, MTLRenderStages stages)
{
    bind_point* bp = compute ? &rec.kernel : &rec.graphics;
    int top = 0;
    for (int i = 0; i < info->argument_count; i++)
        top = info->arguments[i].id + 1 > top ? info->arguments[i].id + 1 : top;
    vk_slice slice = vk_arena((VkDeviceSize)(top ? top : 1) * 8, 16);
    uint64_t* words = slice.data;
    for (int i = 0; i < info->argument_count; i++) {
        const mtl_argument* a = &info->arguments[i];
        const binding_slot* s = a->binding < 32 ? &bp->slots[a->binding] : NULL;
        words[a->id] = 0;
        if (!s || !s->set)
            continue;
        id resource = a->kind == MTL_ARG_SAMPLER && s->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
            ? combined_samplers[compute][a->binding]
            : s->resource;
        MTLResourceUsage usage = s->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || s->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            ? MTLResourceUsageRead | MTLResourceUsageWrite
            : MTLResourceUsageRead;
        id<MTLResource> used = nil;
        if (!resource)
            continue;
        if (a->kind == MTL_ARG_TEXTURE) {
            id<MTLTexture> t = resource;
            MTLResourceID rid = t.gpuResourceID;
            memcpy(&words[a->id], &rid, 8);
            if (!is_attachment(t))
                used = t;
        } else if (a->kind == MTL_ARG_SAMPLER) {
            id<MTLSamplerState> sampler = resource;
            MTLResourceID rid = sampler.gpuResourceID;
            memcpy(&words[a->id], &rid, 8);
        } else if (a->kind == MTL_ARG_BUFFER) {
            id<MTLBuffer> b = resource;
            words[a->id] = b.gpuAddress + s->offset;
            used = b;
        } else if (a->kind == MTL_ARG_ATOMIC) {
            if (!s->backing)
                continue;
            words[a->id] = s->backing.gpuAddress;
            used = s->backing;
            usage = MTLResourceUsageRead | MTLResourceUsageWrite;
        } else if (a->kind == MTL_ARG_STRUCTURE) {
            id<MTLAccelerationStructure> st = resource;
            MTLResourceID rid = st.gpuResourceID;
            memcpy(&words[a->id], &rid, 8);
            used = st;
            for (id<MTLAccelerationStructure> bottom in s->bottoms) {
                if (!compute)
                    [(id<MTLRenderCommandEncoder>)encoder useResource:bottom usage:MTLResourceUsageRead stages:stages];
                else
                    [(id<MTLComputeCommandEncoder>)encoder useResource:bottom usage:MTLResourceUsageRead];
            }
        }
        if (used) {
            if (!compute)
                [(id<MTLRenderCommandEncoder>)encoder useResource:used usage:usage stages:stages];
            else
                [(id<MTLComputeCommandEncoder>)encoder useResource:used usage:usage];
        }
    }
    return slice;
}

static id<MTLDepthStencilState> depth_states[2][2][8];
static id<MTLDepthStencilState> depth_state(int test, int write, VkCompareOp op)
{
    static const MTLCompareFunction functions[8] = { MTLCompareFunctionNever, MTLCompareFunctionLess,
        MTLCompareFunctionEqual, MTLCompareFunctionLessEqual, MTLCompareFunctionGreater, MTLCompareFunctionNotEqual,
        MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways };
    int o = op >= 0 && op < 8 ? (int)op : 7;
    if (!test)
        o = 7;
    id<MTLDepthStencilState>* slot = &depth_states[!!test][!!write][o];
    if (!*slot) {
        MTLDepthStencilDescriptor* d = [[MTLDepthStencilDescriptor new] autorelease];
        d.depthCompareFunction = functions[o];
        d.depthWriteEnabled = write ? YES : NO;
        *slot = [device newDepthStencilStateWithDescriptor:d];
    }
    return *slot;
}
static void apply_viewport(id<MTLRenderCommandEncoder> e)
{
    int w = rec.attachments.width, h = rec.attachments.height;
    if (rec.viewport_set)
        [e setViewport:(MTLViewport){ rec.viewport.x, rec.viewport.y, rec.viewport.width, rec.viewport.height,
                           rec.viewport.minDepth, rec.viewport.maxDepth }];
    else
        [e setViewport:(MTLViewport){ 0, 0, (double)w, (double)h, 0, 1 }];
    int x0 = rec.scissor_set ? rec.scissor.offset.x : 0, y0 = rec.scissor_set ? rec.scissor.offset.y : 0;
    int x1 = rec.scissor_set ? x0 + (int)rec.scissor.extent.width : w;
    int y1 = rec.scissor_set ? y0 + (int)rec.scissor.extent.height : h;
    x0 = x0 < 0 ? 0 : x0 > w ? w : x0, x1 = x1 < x0 ? x0 : x1 > w ? w : x1;
    y0 = y0 < 0 ? 0 : y0 > h ? h : y0, y1 = y1 < y0 ? y0 : y1 > h ? h : y1;
    [e setScissorRect:(MTLScissorRect){ (NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0) }];
}
/* After a utility draw: the renderer's pipeline and state again at the next draw. */
static void restore_render_state(void)
{
    rec.applied = NULL;
    rec.applied_depth = nil;
}
static int prepare_draw(void)
{
    mtl_pipeline* p = rec.graphics.pipeline;
    id<MTLRenderCommandEncoder> e = render_encoder();
    if (!p || !e || p->compute)
        return 0;
    timer_note(p->fs ? p->fs->function : p->vs->function);
    if (rec.applied != p) {
        [e setRenderPipelineState:p->render];
        [e setBlendColorRed:0 green:0 blue:0 alpha:p->blend_constant];
        rec.applied = p;
    }
    id<MTLDepthStencilState> ds = depth_state(p->has_depth && rec.depth_test, p->has_depth && rec.depth_write,
        rec.depth_compare);
    if (ds != rec.applied_depth) {
        [e setDepthStencilState:ds];
        rec.applied_depth = ds;
    }
    apply_viewport(e);
    if (rec.vertex_buffer)
        [e setVertexBuffer:rec.vertex_buffer->buffer offset:(NSUInteger)rec.vertex_offset atIndex:VERTEX_BUFFER_INDEX];
    if (p->vs->set_buffer >= 0) {
        vk_slice s = arguments_for(p->vs, 0, e, MTLRenderStageVertex);
        [e setVertexBuffer:((mtl_buffer*)s.buffer)->buffer offset:(NSUInteger)s.offset atIndex:(NSUInteger)p->vs->set_buffer];
    }
    if (p->vs->push_buffer >= 0)
        [e setVertexBytes:rec.graphics.push length:sizeof rec.graphics.push atIndex:(NSUInteger)p->vs->push_buffer];
    if (p->fs) {
        if (p->fs->set_buffer >= 0) {
            vk_slice s = arguments_for(p->fs, 0, e, MTLRenderStageFragment);
            [e setFragmentBuffer:((mtl_buffer*)s.buffer)->buffer offset:(NSUInteger)s.offset
                         atIndex:(NSUInteger)p->fs->set_buffer];
        }
        if (p->fs->push_buffer >= 0)
            [e setFragmentBytes:rec.graphics.push length:sizeof rec.graphics.push atIndex:(NSUInteger)p->fs->push_buffer];
    }
    return 1;
}
static void cmd_draw(VkCommandBuffer cmd, uint32_t vertices, uint32_t instances, uint32_t first, uint32_t first_instance)
{
    (void)cmd;
    if (!prepare_draw() || !vertices || !instances)
        return;
    [rec.render drawPrimitives:rec.graphics.pipeline->primitive vertexStart:first vertexCount:vertices
                 instanceCount:instances baseInstance:first_instance];
}
static void cmd_draw_indexed(VkCommandBuffer cmd, uint32_t indices, uint32_t instances, uint32_t first, int32_t offset,
    uint32_t first_instance)
{
    (void)cmd;
    if (!prepare_draw() || !indices || !instances || !rec.index_buffer)
        return;
    int wide = rec.index_type == VK_INDEX_TYPE_UINT32;
    [rec.render drawIndexedPrimitives:rec.graphics.pipeline->primitive
                           indexCount:indices
                            indexType:wide ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                          indexBuffer:rec.index_buffer->buffer
                    indexBufferOffset:(NSUInteger)rec.index_offset + first * (wide ? 4u : 2u)
                        instanceCount:instances
                           baseVertex:offset
                         baseInstance:first_instance];
}
static void cmd_dispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z)
{
    (void)cmd;
    mtl_pipeline* p = rec.kernel.pipeline;
    if (!p || !p->compute || !x || !y || !z)
        return;
    id<MTLComputeCommandEncoder> e = compute_encoder();
    timer_note(p->cs->function);
    [e setComputePipelineState:p->kernel];
    if (p->cs->set_buffer >= 0) {
        vk_slice s = arguments_for(p->cs, 1, e, 0);
        [e setBuffer:((mtl_buffer*)s.buffer)->buffer offset:(NSUInteger)s.offset atIndex:(NSUInteger)p->cs->set_buffer];
    }
    if (p->cs->push_buffer >= 0)
        [e setBytes:rec.kernel.push length:sizeof rec.kernel.push atIndex:(NSUInteger)p->cs->push_buffer];
    [e dispatchThreadgroups:MTLSizeMake(x, y, z)
        threadsPerThreadgroup:MTLSizeMake((NSUInteger)p->cs->workgroup[0], (NSUInteger)p->cs->workgroup[1],
                                  (NSUInteger)p->cs->workgroup[2])];
}
static void cmd_bind_pipeline(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipeline pipeline)
{
    (void)cmd;
    if (point == VK_PIPELINE_BIND_POINT_COMPUTE)
        rec.kernel.pipeline = (mtl_pipeline*)pipeline;
    else
        rec.graphics.pipeline = (mtl_pipeline*)pipeline;
}
static void cmd_push_constants(VkCommandBuffer cmd, VkPipelineLayout layout, VkShaderStageFlags stages,
    uint32_t offset, uint32_t size, const void* values)
{
    (void)cmd, (void)layout;
    if (offset + size > sizeof rec.graphics.push)
        return;
    if (stages & VK_SHADER_STAGE_COMPUTE_BIT)
        memcpy(rec.kernel.push + offset, values, size);
    if (stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT))
        memcpy(rec.graphics.push + offset, values, size);
}
static void cmd_bind_vertex_buffers(VkCommandBuffer cmd, uint32_t first, uint32_t count, const VkBuffer* buffers,
    const VkDeviceSize* offsets)
{
    (void)cmd;
    if (first == 0 && count >= 1) {
        rec.vertex_buffer = (mtl_buffer*)buffers[0];
        rec.vertex_offset = offsets[0];
    }
}
static void cmd_bind_index_buffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkIndexType type)
{
    (void)cmd;
    rec.index_buffer = (mtl_buffer*)buffer;
    rec.index_offset = offset;
    rec.index_type = type;
}
static void cmd_set_viewport(VkCommandBuffer cmd, uint32_t first, uint32_t count, const VkViewport* v)
{
    (void)cmd, (void)first, (void)count;
    rec.viewport = *v;
    rec.viewport_set = 1;
}
static void cmd_set_scissor(VkCommandBuffer cmd, uint32_t first, uint32_t count, const VkRect2D* r)
{
    (void)cmd, (void)first, (void)count;
    rec.scissor = *r;
    rec.scissor_set = 1;
}
static void cmd_set_depth_test(VkCommandBuffer cmd, VkBool32 on) { (void)cmd; rec.depth_test = on; }
static void cmd_set_depth_write(VkCommandBuffer cmd, VkBool32 on) { (void)cmd; rec.depth_write = on; }
static void cmd_set_depth_compare(VkCommandBuffer cmd, VkCompareOp op) { (void)cmd; rec.depth_compare = op; }
static void cmd_clear_attachments(VkCommandBuffer cmd, uint32_t count, const VkClearAttachment* attachments,
    uint32_t rect_count, const VkClearRect* rects)
{
    (void)cmd;
    clear_in_scope(attachments, (int)count, rects, (int)rect_count);
}
static void cmd_clear_color_image(VkCommandBuffer cmd, VkImage image, VkImageLayout layout,
    const VkClearColorValue* color, uint32_t count, const VkImageSubresourceRange* ranges)
{
    (void)cmd, (void)layout, (void)count, (void)ranges;
    clear_texture_value((mtl_image*)image, color, 1.0f);
}
static void cmd_clear_depth_image(VkCommandBuffer cmd, VkImage image, VkImageLayout layout,
    const VkClearDepthStencilValue* value, uint32_t count, const VkImageSubresourceRange* ranges)
{
    (void)cmd, (void)layout, (void)count, (void)ranges;
    VkClearColorValue zero = { { 0, 0, 0, 0 } };
    clear_texture_value((mtl_image*)image, &zero, value->depth);
}
static void cmd_copy_buffer_to_image(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, VkImageLayout layout,
    uint32_t count, const VkBufferImageCopy* regions)
{
    (void)cmd, (void)layout;
    mtl_buffer* b = (mtl_buffer*)buffer;
    mtl_image* m = (mtl_image*)image;
    id<MTLBlitCommandEncoder> e = blit_encoder();
    size_t texel = texel_size(m->format);
    for (uint32_t i = 0; i < count; i++) {
        const VkBufferImageCopy* r = &regions[i];
        uint32_t row = r->bufferRowLength ? r->bufferRowLength : r->imageExtent.width;
        uint32_t rows = r->bufferImageHeight ? r->bufferImageHeight : r->imageExtent.height;
        [e copyFromBuffer:b->buffer
                   sourceOffset:(NSUInteger)r->bufferOffset
              sourceBytesPerRow:row * texel
            sourceBytesPerImage:(NSUInteger)row * rows * texel
                     sourceSize:MTLSizeMake(r->imageExtent.width, r->imageExtent.height, r->imageExtent.depth ? r->imageExtent.depth : 1)
                      toTexture:m->texture
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake((NSUInteger)r->imageOffset.x, (NSUInteger)r->imageOffset.y,
                                    (NSUInteger)r->imageOffset.z)];
    }
}
static void cmd_copy_image_to_buffer(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, VkBuffer buffer,
    uint32_t count, const VkBufferImageCopy* regions)
{
    (void)cmd, (void)layout;
    mtl_buffer* b = (mtl_buffer*)buffer;
    mtl_image* m = (mtl_image*)image;
    id<MTLBlitCommandEncoder> e = blit_encoder();
    size_t texel = texel_size(m->format);
    for (uint32_t i = 0; i < count; i++) {
        const VkBufferImageCopy* r = &regions[i];
        uint32_t row = r->bufferRowLength ? r->bufferRowLength : r->imageExtent.width;
        uint32_t rows = r->bufferImageHeight ? r->bufferImageHeight : r->imageExtent.height;
        [e copyFromTexture:m->texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake((NSUInteger)r->imageOffset.x, (NSUInteger)r->imageOffset.y, 0)
                          sourceSize:MTLSizeMake(r->imageExtent.width, r->imageExtent.height, 1)
                            toBuffer:b->buffer
                   destinationOffset:(NSUInteger)r->bufferOffset
              destinationBytesPerRow:row * texel
            destinationBytesPerImage:(NSUInteger)row * rows * texel];
    }
}
static void cmd_copy_image(VkCommandBuffer cmd, VkImage from, VkImageLayout from_layout, VkImage to,
    VkImageLayout to_layout, uint32_t count, const VkImageCopy* regions)
{
    (void)cmd, (void)from_layout, (void)to_layout;
    mtl_image *a = (mtl_image*)from, *b = (mtl_image*)to;
    id<MTLBlitCommandEncoder> e = blit_encoder();
    for (uint32_t i = 0; i < count; i++) {
        const VkImageCopy* r = &regions[i];
        [e copyFromTexture:a->texture
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake((NSUInteger)r->srcOffset.x, (NSUInteger)r->srcOffset.y, (NSUInteger)r->srcOffset.z)
                    sourceSize:MTLSizeMake(r->extent.width, r->extent.height, r->extent.depth ? r->extent.depth : 1)
                     toTexture:b->texture
              destinationSlice:0
              destinationLevel:0
             destinationOrigin:MTLOriginMake((NSUInteger)r->dstOffset.x, (NSUInteger)r->dstOffset.y, (NSUInteger)r->dstOffset.z)];
    }
}
static void cmd_blit_image(VkCommandBuffer cmd, VkImage from, VkImageLayout from_layout, VkImage to,
    VkImageLayout to_layout, uint32_t count, const VkImageBlit* regions, VkFilter filter)
{
    (void)cmd, (void)from_layout, (void)to_layout;
    mtl_image *a = (mtl_image*)from, *b = (mtl_image*)to;
    for (uint32_t i = 0; i < count; i++) {
        const VkImageBlit* r = &regions[i];
        int sw = r->srcOffsets[1].x - r->srcOffsets[0].x, sh = r->srcOffsets[1].y - r->srcOffsets[0].y;
        int dw = r->dstOffsets[1].x - r->dstOffsets[0].x, dh = r->dstOffsets[1].y - r->dstOffsets[0].y;
        if (sw == dw && sh == dh && a->format == b->format) {
            VkImageCopy copy = { .srcOffset = r->srcOffsets[0], .dstOffset = r->dstOffsets[0],
                .extent = { (uint32_t)sw, (uint32_t)sh, 1 } };
            cmd_copy_image(cmd, from, from_layout, to, to_layout, 1, &copy);
            continue;
        }
        /* Scaled or converted (RGBA8 window image -> BGRA8 drawable): drawn. */
        if (rendering)
            vk_end_rendering();
        end_encoder();
        vk_cmd();
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = b->texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = [rec.cb renderCommandEncoderWithDescriptor:pass];
        VkFormat formats[6] = { b->format, 0, 0, 0, 0, 0 };
        id<MTLRenderPipelineState> state = utility_pipeline(formats, 1, 0, filter == VK_FILTER_LINEAR ? 1 : 2);
        if (state) {
            [e setRenderPipelineState:state];
            [e setFragmentTexture:a->texture atIndex:0];
            [e setFragmentSamplerState:((mtl_sampler*)(filter == VK_FILTER_LINEAR ? vk.linear : vk.nearest))->state
                               atIndex:0];
            float uv[4] = { (float)r->srcOffsets[0].x / a->texture.width, (float)r->srcOffsets[0].y / a->texture.height,
                (float)r->srcOffsets[1].x / a->texture.width, (float)r->srcOffsets[1].y / a->texture.height };
            VkRect2D rect = { { r->dstOffsets[0].x, r->dstOffsets[0].y }, { (uint32_t)dw, (uint32_t)dh } };
            utility_draw(e, (int)b->texture.width, (int)b->texture.height, &rect, uv);
        }
        [e endEncoding];
    }
}
static void cmd_copy_buffer(VkCommandBuffer cmd, VkBuffer from, VkBuffer to, uint32_t count, const VkBufferCopy* regions)
{
    (void)cmd;
    id<MTLBlitCommandEncoder> e = blit_encoder();
    for (uint32_t i = 0; i < count; i++)
        [e copyFromBuffer:((mtl_buffer*)from)->buffer
                 sourceOffset:(NSUInteger)regions[i].srcOffset
                     toBuffer:((mtl_buffer*)to)->buffer
            destinationOffset:(NSUInteger)regions[i].dstOffset
                         size:(NSUInteger)regions[i].size];
}
static void cmd_fill_buffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data)
{
    (void)cmd;
    mtl_buffer* b = (mtl_buffer*)buffer;
    if (size == VK_WHOLE_SIZE)
        size = b->size - offset;
    if ((data & 0xff) * 0x01010101u != data)
        fprintf(stderr, "[mtl] vkCmdFillBuffer with a non-byte pattern (%08x): filled with its low byte\n", data);
    [blit_encoder() fillBuffer:b->buffer range:NSMakeRange((NSUInteger)offset, (NSUInteger)size) value:(uint8_t)data];
}
static void cmd_update_buffer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
    const void* data)
{
    vk_slice s = vk_arena(size, 16);
    if (!s.data)
        return;
    memcpy(s.data, data, (size_t)size);
    VkBufferCopy copy = { s.offset, offset, size };
    cmd_copy_buffer(cmd, s.buffer, buffer, 1, &copy);
}
static void cmd_pipeline_barrier2(VkCommandBuffer cmd, const VkDependencyInfo* info)
{
    (void)cmd, (void)info;
    vk_barrier();
}
static void cmd_push_descriptor_set(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout,
    uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes)
{
    push_descriptor_set(cmd, point, layout, set, count, writes);
}

/* Objects created and destroyed directly. */
static VkResult create_pipeline_layout(VkDevice d, const VkPipelineLayoutCreateInfo* info,
    const VkAllocationCallbacks* a, VkPipelineLayout* out)
{
    (void)d, (void)a;
    mtl_layout* l = calloc(1, sizeof *l);
    for (uint32_t i = 0; i < info->pushConstantRangeCount; i++)
        if (info->pPushConstantRanges[i].offset + info->pPushConstantRanges[i].size > l->push_bytes)
            l->push_bytes = info->pPushConstantRanges[i].offset + info->pPushConstantRanges[i].size;
    *out = (VkPipelineLayout)l;
    return VK_SUCCESS;
}
static void destroy_pipeline_layout(VkDevice d, VkPipelineLayout l, const VkAllocationCallbacks* a)
{
    (void)d, (void)a;
    free((void*)l);
}
static void destroy_shader_module(VkDevice d, VkShaderModule m, const VkAllocationCallbacks* a)
{
    (void)d, (void)a;
    mtl_module* module = (mtl_module*)m;
    if (!module)
        return;
    [module->function release];
    free(module);
}
static void destroy_pipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks* a)
{
    (void)d, (void)a;
    mtl_pipeline* pipeline = (mtl_pipeline*)p;
    if (!pipeline)
        return;
    if (rec.graphics.pipeline == pipeline)
        rec.graphics.pipeline = NULL;
    if (rec.kernel.pipeline == pipeline)
        rec.kernel.pipeline = NULL;
    if (rec.applied == pipeline)
        rec.applied = NULL;
    [pipeline->render release];
    if (pipeline->compute)
        [pipeline->kernel release];
    free(pipeline);
}
static void destroy_descriptor_set_layout(VkDevice d, VkDescriptorSetLayout l, const VkAllocationCallbacks* a)
{
    (void)d, (void)a;
    free((void*)l);
}
static VkResult device_wait_idle(VkDevice d)
{
    (void)d;
    if (rec.cb) { /* what is recorded so far is part of the frame: committed when the frame is */
    }
    if (last_committed)
        [last_committed waitUntilCompleted];
    return VK_SUCCESS;
}
static VkResult create_semaphore(VkDevice d, const VkSemaphoreCreateInfo* info, const VkAllocationCallbacks* a,
    VkSemaphore* out)
{
    (void)d, (void)info, (void)a;
    static int dummy;
    *out = (VkSemaphore)&dummy;
    return VK_SUCCESS;
}
static void destroy_semaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a) { (void)d, (void)s, (void)a; }
static void physical_device_format_properties(VkPhysicalDevice gpu, VkFormat format, VkFormatProperties* out)
{
    (void)gpu, (void)format;
    /* Every format the renderer uses can be sampled, drawn into and (R16/R32 integers on Apple GPUs) written. */
    out->linearTilingFeatures = 0;
    out->optimalTilingFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
                               | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
                               | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    out->bufferFeatures = 0;
}
static void physical_device_features2(VkPhysicalDevice gpu, VkPhysicalDeviceFeatures2* out)
{
    (void)gpu;
    memset(&out->features, 0, sizeof out->features);
    out->features.shaderStorageImageExtendedFormats = VK_TRUE;
    out->features.independentBlend = VK_TRUE;
    out->features.fragmentStoresAndAtomics = VK_TRUE;
}

/* ---- presentation: a CAMetalLayer ---- */

static VkResult surface_capabilities(VkPhysicalDevice gpu, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* caps)
{
    (void)gpu;
    mtl_surface* s = (mtl_surface*)surface;
    memset(caps, 0, sizeof *caps);
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(s->window, &w, &h);
    caps->currentExtent = (VkExtent2D){ (uint32_t)w, (uint32_t)h };
    caps->minImageCount = 2;
    caps->maxImageCount = 3;
    caps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    return VK_SUCCESS;
}
static VkResult surface_formats(VkPhysicalDevice gpu, VkSurfaceKHR surface, uint32_t* count, VkSurfaceFormatKHR* out)
{
    (void)gpu, (void)surface;
    if (out && *count >= 1)
        out[0] = (VkSurfaceFormatKHR){ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    *count = 1;
    return VK_SUCCESS;
}
static VkResult surface_present_modes(VkPhysicalDevice gpu, VkSurfaceKHR surface, uint32_t* count,
    VkPresentModeKHR* out)
{
    (void)gpu, (void)surface;
    if (out && *count >= 2)
        out[0] = VK_PRESENT_MODE_FIFO_KHR, out[1] = VK_PRESENT_MODE_IMMEDIATE_KHR;
    *count = 2;
    return VK_SUCCESS;
}
static VkResult surface_support(VkPhysicalDevice gpu, uint32_t family, VkSurfaceKHR surface, VkBool32* out)
{
    (void)gpu, (void)family, (void)surface;
    *out = VK_TRUE;
    return VK_SUCCESS;
}
static void destroy_surface(VkInstance instance, VkSurfaceKHR surface, const VkAllocationCallbacks* a)
{
    (void)instance, (void)a;
    mtl_surface* s = (mtl_surface*)surface;
    if (!s)
        return;
    SDL_Metal_DestroyView(s->view);
    free(s);
}
/* vk_present.c (ROOD_RENDERER=metal): the window's layer as a surface. */
int mtl_create_surface(SDL_Window* window, VkSurfaceKHR* out)
{
    mtl_surface* s = calloc(1, sizeof *s);
    s->window = window;
    s->view = SDL_Metal_CreateView(window);
    if (!s->view) {
        free(s);
        return 0;
    }
    s->layer = (CAMetalLayer*)SDL_Metal_GetLayer(s->view);
    s->layer.device = device;
    s->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    s->layer.framebufferOnly = NO; /* drawn into by the window image's copy */
    *out = (VkSurfaceKHR)s;
    return 1;
}
static VkResult create_swapchain(VkDevice d, const VkSwapchainCreateInfoKHR* info, const VkAllocationCallbacks* a,
    VkSwapchainKHR* out)
{
    (void)d, (void)a;
    mtl_swapchain* sc = calloc(1, sizeof *sc);
    sc->surface = (mtl_surface*)info->surface;
    sc->count = info->minImageCount > 3 ? 3 : (int)info->minImageCount;
    if (sc->count < 2)
        sc->count = 2;
    sc->extent = info->imageExtent;
    sc->surface->layer.drawableSize = CGSizeMake(info->imageExtent.width, info->imageExtent.height);
    sc->surface->layer.maximumDrawableCount = (NSUInteger)(sc->count < 2 ? 2 : sc->count);
#if TARGET_OS_OSX
    sc->surface->layer.displaySyncEnabled = info->presentMode == VK_PRESENT_MODE_FIFO_KHR;
#endif
    for (int i = 0; i < sc->count; i++) {
        sc->images[i].format = VK_FORMAT_B8G8R8A8_UNORM;
        sc->images[i].samples = 1;
    }
    *out = (VkSwapchainKHR)sc;
    return VK_SUCCESS;
}
static void destroy_swapchain(VkDevice d, VkSwapchainKHR swapchain, const VkAllocationCallbacks* a)
{
    (void)d, (void)a;
    mtl_swapchain* sc = (mtl_swapchain*)swapchain;
    if (!sc)
        return;
    for (int i = 0; i < sc->count; i++)
        [sc->drawables[i] release];
    free(sc);
}
static VkResult swapchain_images(VkDevice d, VkSwapchainKHR swapchain, uint32_t* count, VkImage* out)
{
    (void)d;
    mtl_swapchain* sc = (mtl_swapchain*)swapchain;
    if (out) {
        uint32_t n = *count < (uint32_t)sc->count ? *count : (uint32_t)sc->count;
        for (uint32_t i = 0; i < n; i++)
            out[i] = (VkImage)&sc->images[i];
    }
    *count = (uint32_t)sc->count;
    return VK_SUCCESS;
}
static VkResult acquire_next_image(VkDevice d, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
    VkFence fence, uint32_t* index)
{
    (void)d, (void)timeout, (void)semaphore, (void)fence;
    mtl_swapchain* sc = (mtl_swapchain*)swapchain;
    CGSize size = sc->surface->layer.drawableSize;
    if ((uint32_t)size.width != sc->extent.width || (uint32_t)size.height != sc->extent.height)
        return VK_ERROR_OUT_OF_DATE_KHR;
    id<CAMetalDrawable> drawable;
    @autoreleasepool {
        drawable = [[sc->surface->layer nextDrawable] retain];
    }
    if (!drawable)
        return VK_TIMEOUT;
    int i = sc->next;
    sc->next = (sc->next + 1) % sc->count;
    [sc->drawables[i] release];
    sc->drawables[i] = drawable;
    sc->images[i].texture = drawable.texture;
    *index = (uint32_t)i;
    return VK_SUCCESS;
}
static VkResult queue_present(VkQueue q, const VkPresentInfoKHR* info)
{
    (void)q;
    for (uint32_t k = 0; k < info->swapchainCount; k++) {
        mtl_swapchain* sc = (mtl_swapchain*)info->pSwapchains[k];
        uint32_t i = info->pImageIndices[k];
        if (!sc->drawables[i])
            continue;
        /* After the frame's command buffer (committed first, same queue). */
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [cb presentDrawable:sc->drawables[i]];
        [cb commit];
        [sc->drawables[i] release];
        sc->drawables[i] = nil;
        sc->images[i].texture = nil;
    }
    return VK_SUCCESS;
}

/* ---- ray tracing (vk_rt.inc: ray queries over a TLAS of room and mover BLASes) ----
 * A Vulkan acceleration structure is a Metal one; its "device address" (an
 * instance's reference) is the handle itself. A TLAS build turns Vulkan's
 * instances into Metal's (user ids: gl_InstanceCustomIndex) and keeps the
 * BLASes it references, declared with it wherever it is bound. */
static MTLAccelerationStructureDescriptor* structure_descriptor(const VkAccelerationStructureBuildGeometryInfoKHR* info,
    uint32_t count, int bind, NSMutableArray* bottoms, mtl_buffer** instance_out, NSUInteger* instance_offset)
{
    const VkAccelerationStructureGeometryKHR* g = info->pGeometries ? info->pGeometries : *info->ppGeometries;
    if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) {
        MTLAccelerationStructureTriangleGeometryDescriptor* t = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        t.triangleCount = count;
        t.opaque = (g->flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0;
        t.vertexStride = (NSUInteger)g->geometry.triangles.vertexStride;
        t.vertexFormat = MTLAttributeFormatFloat3;
        t.indexType = g->geometry.triangles.indexType == VK_INDEX_TYPE_UINT16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        if (bind) {
            NSUInteger off = 0;
            mtl_buffer* v = buffer_at(g->geometry.triangles.vertexData.deviceAddress, &off);
            t.vertexBuffer = v ? v->buffer : nil;
            t.vertexBufferOffset = off;
            mtl_buffer* ix = buffer_at(g->geometry.triangles.indexData.deviceAddress, &off);
            t.indexBuffer = ix ? ix->buffer : nil;
            t.indexBufferOffset = off;
        }
        MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        d.geometryDescriptors = @[ t ];
        if (info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
            d.usage = MTLAccelerationStructureUsageRefit;
        return d;
    }
    MTLInstanceAccelerationStructureDescriptor* d = [MTLInstanceAccelerationStructureDescriptor descriptor];
    d.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
    d.instanceCount = count;
    if (bind && count) {
        /* Vulkan's instances (CPU-visible arena) -> Metal's, in a new arena slice. */
        NSUInteger off = 0;
        mtl_buffer* src = buffer_at(g->geometry.instances.data.deviceAddress, &off);
        const VkAccelerationStructureInstanceKHR* in = src ? (const void*)((char*)src->buffer.contents + off) : NULL;
        vk_slice slice = vk_arena((VkDeviceSize)count * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor), 16);
        MTLAccelerationStructureUserIDInstanceDescriptor* out = slice.data;
        for (uint32_t i = 0; in && out && i < count; i++) {
            mtl_structure* bottom = (mtl_structure*)(uintptr_t)in[i].accelerationStructureReference;
            NSUInteger index = [bottoms indexOfObjectIdenticalTo:bottom->structure];
            if (index == NSNotFound) {
                index = bottoms.count;
                [bottoms addObject:bottom->structure];
            }
            memset(&out[i], 0, sizeof out[i]);
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 3; r++)
                    out[i].transformationMatrix.columns[c].elements[r] = in[i].transform.matrix[r][c];
            out[i].mask = in[i].mask;
            out[i].userID = in[i].instanceCustomIndex;
            out[i].accelerationStructureIndex = (uint32_t)index;
            out[i].intersectionFunctionTableOffset = 0;
            out[i].options = ((in[i].flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)
                                     ? MTLAccelerationStructureInstanceOptionDisableTriangleCulling : 0)
                           | ((in[i].flags & VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR)
                                     ? MTLAccelerationStructureInstanceOptionOpaque : 0)
                           | ((in[i].flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR)
                                     ? MTLAccelerationStructureInstanceOptionNonOpaque : 0);
        }
        d.instanceDescriptorBuffer = ((mtl_buffer*)slice.buffer)->buffer;
        d.instanceDescriptorBufferOffset = (NSUInteger)slice.offset;
        d.instancedAccelerationStructures = bottoms;
        if (instance_out)
            *instance_out = (mtl_buffer*)slice.buffer, *instance_offset = (NSUInteger)slice.offset;
    }
    return d;
}
static void structure_sizes(VkDevice d, VkAccelerationStructureBuildTypeKHR type,
    const VkAccelerationStructureBuildGeometryInfoKHR* info, const uint32_t* counts,
    VkAccelerationStructureBuildSizesInfoKHR* sizes)
{
    (void)d, (void)type;
    @autoreleasepool {
        MTLAccelerationStructureDescriptor* desc = structure_descriptor(info, counts[0], 0, nil, NULL, NULL);
        MTLAccelerationStructureSizes s = [device accelerationStructureSizesWithDescriptor:desc];
        sizes->accelerationStructureSize = s.accelerationStructureSize;
        sizes->buildScratchSize = s.buildScratchBufferSize;
        sizes->updateScratchSize = s.refitScratchBufferSize;
    }
}
static VkResult create_structure(VkDevice d, const VkAccelerationStructureCreateInfoKHR* info,
    const VkAllocationCallbacks* a, VkAccelerationStructureKHR* out)
{
    (void)d, (void)a;
    mtl_structure* s = calloc(1, sizeof *s);
    s->structure = [device newAccelerationStructureWithSize:(NSUInteger)info->size];
    s->top = info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (!s->structure) {
        free(s);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = (VkAccelerationStructureKHR)s;
    return VK_SUCCESS;
}
static void destroy_structure(VkDevice d, VkAccelerationStructureKHR handle, const VkAllocationCallbacks* a)
{
    (void)d, (void)a;
    mtl_structure* s = (mtl_structure*)handle;
    if (!s)
        return;
    [s->structure release];
    [s->bottoms release];
    free(s);
}
static VkDeviceAddress structure_address(VkDevice d, const VkAccelerationStructureDeviceAddressInfoKHR* info)
{
    (void)d;
    return (VkDeviceAddress)(uintptr_t)info->accelerationStructure;
}
static void cmd_build_structures(VkCommandBuffer cmd, uint32_t count, const VkAccelerationStructureBuildGeometryInfoKHR* infos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* ranges)
{
    (void)cmd;
    if (rendering)
        vk_end_rendering();
    end_encoder();
    vk_cmd();
    id<MTLAccelerationStructureCommandEncoder> e = [rec.cb accelerationStructureCommandEncoder];
    for (uint32_t i = 0; i < count; i++) {
        const VkAccelerationStructureBuildGeometryInfoKHR* info = &infos[i];
        mtl_structure* dst = (mtl_structure*)info->dstAccelerationStructure;
        mtl_structure* src = (mtl_structure*)info->srcAccelerationStructure;
        NSMutableArray* bottoms = dst->top ? [[NSMutableArray alloc] init] : nil;
        MTLAccelerationStructureDescriptor* desc = structure_descriptor(info, ranges[i][0].primitiveCount, 1, bottoms,
            NULL, NULL);
        NSUInteger scratch_offset = 0;
        mtl_buffer* scratch = buffer_at(info->scratchData.deviceAddress, &scratch_offset);
        if (!scratch) {
            fprintf(stderr, "[mtl] an acceleration structure build without scratch memory\n");
            [bottoms release];
            continue;
        }
        if (dst->top) {
            for (id<MTLAccelerationStructure> bottom in bottoms)
                [e useResource:bottom usage:MTLResourceUsageRead];
            [dst->bottoms release];
            dst->bottoms = bottoms;
        }
        if (info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && src)
            [e refitAccelerationStructure:src->structure
                               descriptor:desc
                              destination:dst->structure
                            scratchBuffer:scratch->buffer
                      scratchBufferOffset:scratch_offset];
        else
            [e buildAccelerationStructure:dst->structure
                               descriptor:desc
                            scratchBuffer:scratch->buffer
                      scratchBufferOffset:scratch_offset];
    }
    [e endEncoding];
}

static void install_entry_points(void)
{
    vkCmdBindPipeline = cmd_bind_pipeline;
    vkCmdPushConstants = cmd_push_constants;
    vkCmdDraw = cmd_draw;
    vkCmdDrawIndexed = cmd_draw_indexed;
    vkCmdDispatch = cmd_dispatch;
    vkCmdSetViewport = cmd_set_viewport;
    vkCmdSetScissor = cmd_set_scissor;
    vkCmdSetDepthTestEnable = cmd_set_depth_test;
    vkCmdSetDepthWriteEnable = cmd_set_depth_write;
    vkCmdSetDepthCompareOp = cmd_set_depth_compare;
    vkCmdBindVertexBuffers = cmd_bind_vertex_buffers;
    vkCmdBindIndexBuffer = cmd_bind_index_buffer;
    vkCmdClearAttachments = cmd_clear_attachments;
    vkCmdClearColorImage = cmd_clear_color_image;
    vkCmdClearDepthStencilImage = cmd_clear_depth_image;
    vkCmdCopyBufferToImage = cmd_copy_buffer_to_image;
    vkCmdCopyImageToBuffer = cmd_copy_image_to_buffer;
    vkCmdCopyImage = cmd_copy_image;
    vkCmdBlitImage = cmd_blit_image;
    vkCmdCopyBuffer = cmd_copy_buffer;
    vkCmdFillBuffer = cmd_fill_buffer;
    vkCmdUpdateBuffer = cmd_update_buffer;
    vkCmdPipelineBarrier2 = cmd_pipeline_barrier2;
    vkCmdPushDescriptorSetKHR = cmd_push_descriptor_set;
    vkCreatePipelineLayout = create_pipeline_layout;
    vkDestroyPipelineLayout = destroy_pipeline_layout;
    vkDestroyShaderModule = destroy_shader_module;
    vkDestroyPipeline = destroy_pipeline;
    vkDestroyDescriptorSetLayout = destroy_descriptor_set_layout;
    vkDeviceWaitIdle = device_wait_idle;
    vkCreateSemaphore = create_semaphore;
    vkDestroySemaphore = destroy_semaphore;
    vkGetPhysicalDeviceFormatProperties = physical_device_format_properties;
    vkGetPhysicalDeviceFeatures2 = physical_device_features2;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR = surface_capabilities;
    vkGetPhysicalDeviceSurfaceFormatsKHR = surface_formats;
    vkGetPhysicalDeviceSurfacePresentModesKHR = surface_present_modes;
    vkGetPhysicalDeviceSurfaceSupportKHR = surface_support;
    vkDestroySurfaceKHR = destroy_surface;
    vkCreateSwapchainKHR = create_swapchain;
    vkDestroySwapchainKHR = destroy_swapchain;
    vkGetSwapchainImagesKHR = swapchain_images;
    vkAcquireNextImageKHR = acquire_next_image;
    vkQueuePresentKHR = queue_present;
    if (vk.ray_query) {
        vkGetAccelerationStructureBuildSizesKHR = structure_sizes;
        vkCreateAccelerationStructureKHR = create_structure;
        vkDestroyAccelerationStructureKHR = destroy_structure;
        vkGetAccelerationStructureDeviceAddressKHR = structure_address;
        vkCmdBuildAccelerationStructuresKHR = cmd_build_structures;
    }
}

/* ---- MetalFX (vk_present.c): the lit image magnified to the view ----
 * Lighting runs at the internal resolution (vk_present.c lit_internal); its
 * picture is enlarged to the window by MetalFX's spatial scaler instead of
 * a bilinear draw: edges and texels stay sharp on a Retina window. The
 * scaler writes a whole texture, so it goes to one of the view's size, then
 * to the destination's rectangle. 0: unavailable (the caller draws). */
int mtl_upscale(const vk_image* source, int src_w, int src_h, const vk_image* dest, int x, int y, int w, int h)
{
    static id<MTLFXSpatialScaler> scaler;
    static id<MTLTexture> output;
    static int in_w, in_h, out_w, out_h;
    static VkFormat in_format, out_format;
    static int unavailable = -1;
    if (unavailable < 0) {
        const char* v = getenv("ROOD_METALFX");
        unavailable = (v && *v == '0') || ![MTLFXSpatialScalerDescriptor supportsDevice:device];
        if (unavailable)
            fprintf(stderr, "[mtl] MetalFX off: the view is magnified bilinearly\n");
    }
    if (unavailable || !source->image || !dest->image || w <= 0 || h <= 0)
        return 0;
    if (!scaler || in_w != src_w || in_h != src_h || out_w != w || out_h != h || in_format != source->format
        || out_format != dest->format) {
        [scaler release];
        [output release];
        MTLFXSpatialScalerDescriptor* d = [[MTLFXSpatialScalerDescriptor new] autorelease];
        d.inputWidth = (NSUInteger)src_w, d.inputHeight = (NSUInteger)src_h;
        d.outputWidth = (NSUInteger)w, d.outputHeight = (NSUInteger)h;
        d.colorTextureFormat = pixel_format(source->format);
        d.outputTextureFormat = pixel_format(dest->format);
        d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual; /* display-referred by now */
        scaler = [d newSpatialScalerWithDevice:device];
        MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format(dest->format)
                                                                                     width:(NSUInteger)w
                                                                                    height:(NSUInteger)h
                                                                                 mipmapped:NO];
        t.storageMode = MTLStorageModePrivate;
        t.usage = scaler ? scaler.outputTextureUsage : MTLTextureUsageShaderWrite;
        output = scaler ? [device newTextureWithDescriptor:t] : nil;
        in_w = src_w, in_h = src_h, out_w = w, out_h = h, in_format = source->format, out_format = dest->format;
        if (!scaler || !output) {
            fprintf(stderr, "[mtl] MetalFX scaler %dx%d -> %dx%d unavailable\n", src_w, src_h, w, h);
            [scaler release], scaler = nil;
            unavailable = 1;
            return 0;
        }
        fprintf(stderr, "[mtl] MetalFX: %dx%d -> %dx%d\n", src_w, src_h, w, h);
    }
    if (rendering)
        vk_end_rendering();
    end_encoder();
    vk_cmd();
    scaler.colorTexture = ((mtl_image*)source->image)->texture;
    scaler.inputContentWidth = (NSUInteger)src_w;
    scaler.inputContentHeight = (NSUInteger)src_h;
    scaler.outputTexture = output;
    [scaler encodeToCommandBuffer:rec.cb];
    [blit_encoder() copyFromTexture:output
                        sourceSlice:0
                        sourceLevel:0
                       sourceOrigin:MTLOriginMake(0, 0, 0)
                         sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
                          toTexture:((mtl_image*)dest->image)->texture
                   destinationSlice:0
                   destinationLevel:0
                  destinationOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)];
    end_encoder();
    return 1;
}

/* ---- the developer overlay and modern menus (vk_overlay.cpp) ---- */

uint64_t mtl_overlay_texture_create(int width, int height)
{
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:(NSUInteger)width
                                                                                height:(NSUInteger)height
                                                                             mipmapped:NO];
    d.storageMode = MTLStorageModeShared; /* written from the CPU (replaceRegion) */
    d.usage = MTLTextureUsageShaderRead;
    return (uint64_t)(uintptr_t)[device newTextureWithDescriptor:d];
}
void mtl_overlay_texture_update(uint64_t texture, int x, int y, int w, int h, const void* rgba, int pitch)
{
    id<MTLTexture> t = (id<MTLTexture>)(uintptr_t)texture;
    /* A frame in flight may still sample it: replaced only once the GPU is done with it. */
    vkDeviceWaitIdle(vk.device);
    [t replaceRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h) mipmapLevel:0
           withBytes:rgba bytesPerRow:(NSUInteger)pitch];
}
void mtl_overlay_texture_destroy(uint64_t texture)
{
    vkDeviceWaitIdle(vk.device);
    [(id<MTLTexture>)(uintptr_t)texture release];
}
typedef struct {
    float clip[4];
    uint64_t texture;
    unsigned index_offset, index_count, vertex_offset;
} mtl_overlay_cmd;
void mtl_overlay_draw(const void* vertices, int vertex_count, const uint16_t* indices, int index_count,
    const mtl_overlay_cmd* cmds, int cmd_count, const float display[4], int fb_w, int fb_h)
{
    id<MTLRenderCommandEncoder> e = render_encoder();
    if (!e || !vertex_count || !index_count)
        return;
    static id<MTLRenderPipelineState> states[2]; /* RGBA8, RGBA16F targets */
    VkFormat format = rec.attachment_formats[0];
    int which = format == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!states[which]) {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor new] autorelease];
        d.vertexFunction = utility_function("overlay_vertex");
        d.fragmentFunction = utility_function("overlay_fragment");
        d.colorAttachments[0].pixelFormat = pixel_format(format);
        d.colorAttachments[0].blendingEnabled = YES;
        d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        if (rec.attachment_formats[5] != VK_FORMAT_UNDEFINED)
            d.depthAttachmentPixelFormat = pixel_format(rec.attachment_formats[5]);
        NSError* error = nil;
        states[which] = [device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!states[which]) {
            fprintf(stderr, "[mtl] overlay pipeline: %s\n", error.localizedDescription.UTF8String);
            return;
        }
    }
    vk_slice v = vk_arena((VkDeviceSize)vertex_count * 20, 16), i = vk_arena((VkDeviceSize)index_count * 2, 16);
    if (!v.data || !i.data)
        return;
    memcpy(v.data, vertices, (size_t)vertex_count * 20);
    memcpy(i.data, indices, (size_t)index_count * 2);
    [e setRenderPipelineState:states[which]];
    [e setDepthStencilState:depth_state(0, 0, VK_COMPARE_OP_ALWAYS)];
    [e setViewport:(MTLViewport){ 0, 0, (double)fb_w, (double)fb_h, 0, 1 }];
    [e setVertexBuffer:((mtl_buffer*)v.buffer)->buffer offset:(NSUInteger)v.offset atIndex:0];
    [e setVertexBytes:display length:16 atIndex:1];
    [e setFragmentSamplerState:((mtl_sampler*)vk.linear)->state atIndex:0];
    int width = rec.attachments.width, height = rec.attachments.height;
    for (int k = 0; k < cmd_count; k++) {
        const mtl_overlay_cmd* c = &cmds[k];
        int x0 = (int)c->clip[0], y0 = (int)c->clip[1], x1 = (int)c->clip[2], y1 = (int)c->clip[3];
        x0 = x0 < 0 ? 0 : x0, y0 = y0 < 0 ? 0 : y0;
        x1 = x1 > width ? width : x1, y1 = y1 > height ? height : y1;
        if (x1 <= x0 || y1 <= y0 || !c->texture || !c->index_count)
            continue;
        [e setScissorRect:(MTLScissorRect){ (NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0) }];
        [e setFragmentTexture:(id<MTLTexture>)(uintptr_t)c->texture atIndex:0];
        [e drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                      indexCount:c->index_count
                       indexType:MTLIndexTypeUInt16
                     indexBuffer:((mtl_buffer*)i.buffer)->buffer
               indexBufferOffset:(NSUInteger)i.offset + c->index_offset * 2u
                   instanceCount:1
                      baseVertex:(NSInteger)c->vertex_offset
                    baseInstance:0];
    }
    restore_render_state();
}

/* ---- readback ---- */

int vk_read_image(const vk_image* image, VkImageAspectFlags aspect, int x, int y, int w, int h, void* out,
    size_t texel_bytes)
{
    (void)aspect;
    if (!image->image || w <= 0 || h <= 0)
        return 0;
    vk_buffer staging;
    if (!vk_buffer_create(&staging, (VkDeviceSize)w * h * texel_bytes, 0, 2))
        return 0;
    mtl_image* m = (mtl_image*)image->image;
    id<MTLBlitCommandEncoder> e = blit_encoder();
    [e copyFromTexture:m->texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake((NSUInteger)x, (NSUInteger)y, 0)
                      sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
                        toBuffer:((mtl_buffer*)staging.buffer)->buffer
               destinationOffset:0
          destinationBytesPerRow:(NSUInteger)w * texel_bytes
        destinationBytesPerImage:(NSUInteger)w * h * texel_bytes];
    vk_flush_wait();
    memcpy(out, staging.mapped, (size_t)w * h * texel_bytes);
    vk_buffer_destroy(&staging);
    return 1;
}
