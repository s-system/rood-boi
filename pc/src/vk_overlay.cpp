// Developer overlay on Vulkan: Dear ImGui's upstream Vulkan backend
// (imgui_impl_vulkan, MIT) drawing into the presenter's window image with
// dynamic rendering. dev_overlay.cpp builds the UI as on OpenGL.
#ifndef IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#define IMGUI_IMPL_VULKAN_NO_PROTOTYPES /* as build_boot.py compiles imgui_impl_vulkan.cpp */
#endif
#include "vk_core.h"
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include <stdio.h>
#include <string.h>

static bool ready;

static PFN_vkVoidFunction load(const char* name, void*)
{
    /* Instance-level entry points (physical device, surface) come from the
     * instance: asking the device for them is invalid usage. */
    bool instance_level = !strncmp(name, "vkGetPhysicalDevice", 19) || !strncmp(name, "vkEnumeratePhysicalDevices", 26)
                       || !strcmp(name, "vkDestroySurfaceKHR");
    PFN_vkVoidFunction f = !instance_level && vkGetDeviceProcAddr ? vkGetDeviceProcAddr(vk.device, name) : nullptr;
    return f ? f : vk_get_instance_proc_addr(vk.instance, name);
}

#ifdef ROOD_METAL
/* Under Metal (mtl_core.m) there is no Vulkan device for the upstream
 * backend: a small renderer of its own. Textures follow Dear ImGui's
 * protocol (ImGuiBackendFlags_RendererHasTextures: the font atlas and the
 * menus' images are ImTextureData it asks us to create, update and destroy);
 * the draw lists are drawn by mtl_core.m into the open rendering scope. */
extern "C" {
uint64_t mtl_overlay_texture_create(int width, int height);
void mtl_overlay_texture_update(uint64_t texture, int x, int y, int w, int h, const void* rgba, int pitch);
void mtl_overlay_texture_destroy(uint64_t texture);
typedef struct {
    float clip[4]; /* framebuffer pixels: x0, y0, x1, y1 */
    uint64_t texture;
    unsigned index_offset, index_count, vertex_offset;
} mtl_overlay_cmd;
void mtl_overlay_draw(const void* vertices, int vertex_count, const uint16_t* indices, int index_count,
    const mtl_overlay_cmd* cmds, int cmd_count, const float display[4], int fb_w, int fb_h);
}
static void update_texture(ImTextureData* tex)
{
    if (tex->Status == ImTextureStatus_WantCreate) {
        uint64_t t = mtl_overlay_texture_create(tex->Width, tex->Height);
        tex->SetTexID((ImTextureID)t);
        tex->Status = ImTextureStatus_WantUpdates;
        tex->Updates.resize(0);
        tex->UpdateRect = { 0, 0, (unsigned short)tex->Width, (unsigned short)tex->Height };
        tex->Updates.push_back(tex->UpdateRect);
    }
    if (tex->Status == ImTextureStatus_WantUpdates) {
        for (const ImTextureRect& r : tex->Updates) {
            /* Alpha8 atlases become RGBA (white, coverage in alpha), as other backends do. */
            ImVector<unsigned int> rgba;
            const void* pixels = tex->GetPixelsAt(r.x, r.y);
            int pitch = tex->GetPitch();
            if (tex->Format == ImTextureFormat_Alpha8) {
                rgba.resize(r.w * r.h);
                for (int y = 0; y < r.h; y++)
                    for (int x = 0; x < r.w; x++)
                        rgba[y * r.w + x] = 0x00ffffffu | (unsigned)((const unsigned char*)tex->GetPixelsAt(r.x + x, r.y + y))[0] << 24;
                pixels = rgba.Data, pitch = r.w * 4;
            }
            mtl_overlay_texture_update((uint64_t)tex->GetTexID(), r.x, r.y, r.w, r.h, pixels, pitch);
        }
        tex->SetStatus(ImTextureStatus_OK);
    }
    if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 2) {
        mtl_overlay_texture_destroy((uint64_t)tex->GetTexID());
        tex->SetTexID(ImTextureID_Invalid);
        tex->SetStatus(ImTextureStatus_Destroyed);
    }
}
#endif

extern "C" int pc_vk_overlay_init(void)
{
#ifdef ROOD_METAL
    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "rood_metal";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures;
    ImGui::GetPlatformIO().Renderer_TextureMaxWidth = ImGui::GetPlatformIO().Renderer_TextureMaxHeight = 16384;
    ready = true;
    return 1;
#endif
    if (!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_3, load)) {
        fprintf(stderr, "[overlay] Vulkan backend functions unavailable\n");
        return 0;
    }
    static const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    ImGui_ImplVulkan_InitInfo info = {};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = vk.instance;
    info.PhysicalDevice = vk.gpu;
    info.Device = vk.device;
    info.QueueFamily = vk.queue_family;
    info.Queue = vk.queue;
    info.PipelineCache = vk.pipeline_cache; /* persisted between runs (vk_core.c) */
    info.DescriptorPoolSize = 64;
    info.MinImageCount = 2;
    info.ImageCount = VK_FRAMES + 1;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &format;
    if (!ImGui_ImplVulkan_Init(&info)) {
        fprintf(stderr, "[overlay] ImGui Vulkan backend failed\n");
        return 0;
    }
    ready = true;
    return 1;
}

/* Before an engine reload (dev_overlay.cpp): the backend's own objects, while the device lives. */
extern "C" void pc_vk_overlay_shutdown(void)
{
    if (!ready)
        return;
    vkDeviceWaitIdle(vk.device);
#ifdef ROOD_METAL
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures)
        if (tex->GetTexID() != ImTextureID_Invalid) {
            mtl_overlay_texture_destroy((uint64_t)tex->GetTexID());
            tex->SetTexID(ImTextureID_Invalid);
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    ImGui::GetIO().BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
    ImGui::GetIO().BackendRendererName = nullptr;
    ready = false;
    return;
#endif
    ImGui_ImplVulkan_Shutdown();
    ready = false;
}

extern "C" void pc_vk_overlay_new_frame(void)
{
#ifndef ROOD_METAL
    if (ready)
        ImGui_ImplVulkan_NewFrame();
#endif
}

/* The UI built this frame (ImGui::Render already ran), inside a rendering
 * scope on the window image. */
extern "C" void pc_vk_overlay_render(VkCommandBuffer cmd)
{
    if (!ready || !ImGui::GetCurrentContext())
        return;
    ImDrawData* data = ImGui::GetDrawData();
    if (!data || !data->Valid)
        return;
#ifdef ROOD_METAL
    (void)cmd;
    if (data->Textures)
        for (ImTextureData* tex : *data->Textures)
            if (tex->Status != ImTextureStatus_OK)
                update_texture(tex);
    int fb_w = (int)(data->DisplaySize.x * data->FramebufferScale.x), fb_h = (int)(data->DisplaySize.y * data->FramebufferScale.y);
    if (fb_w <= 0 || fb_h <= 0 || !data->TotalVtxCount)
        return;
    /* Every list's vertices and indices in one array each, commands with offsets into them. */
    static ImVector<ImDrawVert> vertices;
    static ImVector<ImDrawIdx> indices;
    static ImVector<mtl_overlay_cmd> cmds;
    vertices.resize(0), indices.resize(0), cmds.resize(0);
    ImVec2 off = data->DisplayPos, scale = data->FramebufferScale;
    for (const ImDrawList* list : data->CmdLists) {
        unsigned vbase = (unsigned)vertices.Size, ibase = (unsigned)indices.Size;
        vertices.reserve(vertices.Size + list->VtxBuffer.Size);
        for (const ImDrawVert& v : list->VtxBuffer)
            vertices.push_back(v);
        for (const ImDrawIdx& i : list->IdxBuffer)
            indices.push_back(i);
        for (const ImDrawCmd& c : list->CmdBuffer) {
            if (c.UserCallback) {
                if (c.UserCallback != ImDrawCallback_ResetRenderState)
                    c.UserCallback(list, &c);
                continue;
            }
            mtl_overlay_cmd m;
            m.clip[0] = (c.ClipRect.x - off.x) * scale.x, m.clip[1] = (c.ClipRect.y - off.y) * scale.y;
            m.clip[2] = (c.ClipRect.z - off.x) * scale.x, m.clip[3] = (c.ClipRect.w - off.y) * scale.y;
            m.texture = (uint64_t)c.GetTexID();
            m.index_offset = ibase + c.IdxOffset, m.index_count = c.ElemCount, m.vertex_offset = vbase + c.VtxOffset;
            cmds.push_back(m);
        }
    }
    float display[4] = { off.x, off.y, data->DisplaySize.x, data->DisplaySize.y };
    mtl_overlay_draw(vertices.Data, vertices.Size, indices.Data, indices.Size, cmds.Data, cmds.Size, display, fb_w, fb_h);
#else
    VkViewport viewport = { 0, 0, data->DisplaySize.x, data->DisplaySize.y, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    ImGui_ImplVulkan_RenderDrawData(data, cmd);
#endif
}
