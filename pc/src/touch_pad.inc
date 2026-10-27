/* The touch pad (included by dev_overlay.cpp): a PS1 pad drawn over the
 * picture for touch screens without a gamepad (the iPhone; ROOD_TOUCH=1
 * shows it elsewhere, ROOD_TOUCH=0 hides it). The D-pad is a stick: a
 * finger that lands on its side of the screen steers from where it landed
 * (eight directions); the face buttons, L1/L2/R1/R2, Select and Start are
 * pressed while a finger is on them. Its buttons join the gamepad's
 * (boot_window.c, pc_touch_pad_buttons); it hides while a gamepad is
 * connected. Drawn on Dear ImGui's foreground list, every frame. */
#include <SDL3/SDL.h>
#include <float.h>
#include <math.h>

namespace touch_pad {
/* libetc pad bits (boot_controller.h): L2 R2 L1 R1 Triangle Circle Cross Square Select L3 R3 Start Up Right Down Left */
enum { L2 = 0, R2, L1, R1, TRIANGLE, CIRCLE, CROSS, SQUARE, SELECT, L3, R3, START, UP, RIGHT, DOWN, LEFT };
struct Button {
    int bit;
    float x, y, r; /* centre and radius, in units of the screen's short side (see layout) */
    const char* label;
    bool round;
};
static const int MAX_FINGERS = 10;
static struct {
    SDL_FingerID id;
    int control; /* -1 the stick, else a button's index; -2 free */
    float x0, y0, x, y; /* the stick's origin and the finger (window pixels) */
} fingers[MAX_FINGERS];
static unsigned held;
static int enabled = -1;
static float view_w = 1, view_h = 1;

static bool on()
{
    if (enabled < 0) {
        const char* v = getenv("ROOD_TOUCH");
#ifdef ROOD_IOS
        enabled = !(v && *v == '0');
#else
        enabled = v && *v == '1';
#endif
        for (auto& f : fingers)
            f.control = -2;
    }
    return enabled && !SDL_HasGamepad();
}
/* Positions: x from the left (negative: from the right), y from the bottom
 * (negative: from the top), in units of u = the short side / 9. */
static const Button buttons[] = {
    { TRIANGLE, -2.1f, 3.2f, 0.62f, "", true },
    { CIRCLE, -0.95f, 2.05f, 0.62f, "", true },
    { CROSS, -2.1f, 0.9f, 0.62f, "", true },
    { SQUARE, -3.25f, 2.05f, 0.62f, "", true },
    { L2, 1.3f, -0.7f, 0.55f, "L2", false },
    { L1, 1.3f, -1.95f, 0.55f, "L1", false },
    { R2, -1.3f, -0.7f, 0.55f, "R2", false },
    { R1, -1.3f, -1.95f, 0.55f, "R1", false },
    { SELECT, -100.0f, 0.55f, 0.42f, "SELECT", false }, /* (x: beside the centre, see centre_x) */
    { START, 100.0f, 0.55f, 0.42f, "START", false },
};
static const int BUTTON_COUNT = (int)(sizeof buttons / sizeof buttons[0]);
static float unit() { return (view_w < view_h ? view_w : view_h) / 9.0f; }
static ImVec2 centre(const Button& b)
{
    float u = unit();
    float x = b.x == -100.0f ? view_w * 0.5f - 1.6f * u : b.x == 100.0f ? view_w * 0.5f + 1.6f * u
            : b.x < 0 ? view_w + b.x * u : b.x * u;
    float y = b.y < 0 ? -b.y * u : view_h - b.y * u;
    return ImVec2(x, y);
}
static ImVec2 stick_home() { float u = unit(); return ImVec2(2.3f * u, view_h - 2.3f * u); }
static int button_at(float x, float y)
{
    float u = unit();
    for (int i = 0; i < BUTTON_COUNT; i++) {
        ImVec2 c = centre(buttons[i]);
        float r = buttons[i].r * u * (buttons[i].round ? 1.35f : 1.6f); /* a little slack around each */
        if ((x - c.x) * (x - c.x) + (y - c.y) * (y - c.y) <= r * r)
            return i;
    }
    return -2;
}
static unsigned stick_bits(float dx, float dy)
{
    float u = unit(), length = sqrtf(dx * dx + dy * dy);
    if (length < 0.35f * u)
        return 0;
    float angle = atan2f(dy, dx); /* screen y down */
    int sector = (int)floorf((angle + (float)M_PI / 8.0f) / ((float)M_PI / 4.0f)) & 7; /* 0 right, 2 down, 4 left, 6 up */
    static const unsigned bits[8] = { 1u << RIGHT, (1u << RIGHT) | (1u << DOWN), 1u << DOWN, (1u << DOWN) | (1u << LEFT),
        1u << LEFT, (1u << LEFT) | (1u << UP), 1u << UP, (1u << UP) | (1u << RIGHT) };
    return bits[sector];
}
static void update()
{
    unsigned bits = 0;
    for (auto& f : fingers) {
        if (f.control == -1)
            bits |= stick_bits(f.x - f.x0, f.y - f.y0);
        else if (f.control >= 0)
            bits |= 1u << buttons[f.control].bit;
    }
    held = bits;
}
} // namespace touch_pad

/* boot_window.c: an SDL event (fingers only are taken). */
extern "C" void pc_touch_pad_event(const SDL_Event* e)
{
    using namespace touch_pad;
    if (e->type != SDL_EVENT_FINGER_DOWN && e->type != SDL_EVENT_FINGER_MOTION && e->type != SDL_EVENT_FINGER_UP
        && e->type != SDL_EVENT_FINGER_CANCELED)
        return;
    if (!on()) {
        held = 0;
        return;
    }
    float x = e->tfinger.x * view_w, y = e->tfinger.y * view_h;
    int slot = -1;
    for (int i = 0; i < MAX_FINGERS; i++)
        if (fingers[i].control != -2 && fingers[i].id == e->tfinger.fingerID)
            slot = i;
    if (e->type == SDL_EVENT_FINGER_DOWN) {
        if (slot < 0)
            for (int i = 0; i < MAX_FINGERS && slot < 0; i++)
                if (fingers[i].control == -2)
                    slot = i;
        if (slot < 0)
            return;
        int b = button_at(x, y);
        fingers[slot].id = e->tfinger.fingerID;
        if (b >= 0)
            fingers[slot].control = b;
        else if (x < view_w * 0.45f && y > view_h * 0.35f) /* the stick: from where the finger lands */
            fingers[slot].control = -1, fingers[slot].x0 = x, fingers[slot].y0 = y;
        else
            return; /* (elsewhere: nothing) */
        fingers[slot].x = x, fingers[slot].y = y;
    } else if (slot >= 0) {
        if (e->type == SDL_EVENT_FINGER_MOTION) {
            fingers[slot].x = x, fingers[slot].y = y;
            if (fingers[slot].control >= 0) { /* sliding onto another button presses that one */
                int b = button_at(x, y);
                if (b >= 0)
                    fingers[slot].control = b;
            }
        } else
            fingers[slot].control = -2;
    }
    update();
}
extern "C" unsigned pc_touch_pad_buttons(void) { return touch_pad::on() ? touch_pad::held : 0; }

/* The pad over everything (pc_overlay_frame, before ImGui::Render). */
static void touch_pad_draw()
{
    using namespace touch_pad;
    ImGuiIO& io = ImGui::GetIO();
    view_w = io.DisplaySize.x, view_h = io.DisplaySize.y;
    if (!on())
        return;
    ImDrawList* d = ImGui::GetForegroundDrawList();
    float u = unit();
    const ImU32 fill = IM_COL32(255, 255, 255, 38), line = IM_COL32(255, 255, 255, 110),
                pressed = IM_COL32(255, 255, 255, 120), text = IM_COL32(255, 255, 255, 190);
    /* The stick: its ring at home, or under the finger that holds it. */
    ImVec2 home = stick_home(), knob = home;
    for (auto& f : fingers)
        if (f.control == -1) {
            home = ImVec2(f.x0, f.y0);
            float dx = f.x - f.x0, dy = f.y - f.y0, l = sqrtf(dx * dx + dy * dy), m = 1.1f * u;
            knob = l > m ? ImVec2(home.x + dx * m / l, home.y + dy * m / l) : ImVec2(f.x, f.y);
        }
    d->AddCircleFilled(home, 1.5f * u, fill, 48);
    d->AddCircle(home, 1.5f * u, line, 48, 2.0f);
    d->AddCircleFilled(knob, 0.6f * u, (held & 0xF000u) ? pressed : line, 32);
    for (int i = 0; i < BUTTON_COUNT; i++) {
        const Button& b = buttons[i];
        ImVec2 c = centre(b);
        float r = b.r * u;
        bool down = held & (1u << b.bit);
        if (b.round) {
            d->AddCircleFilled(c, r, down ? pressed : fill, 32);
            d->AddCircle(c, r, line, 32, 2.0f);
        } else {
            float w = b.bit == SELECT || b.bit == START ? 2.3f * r : 1.9f * r, h = r;
            d->AddRectFilled(ImVec2(c.x - w, c.y - h), ImVec2(c.x + w, c.y + h), down ? pressed : fill, h);
            d->AddRect(ImVec2(c.x - w, c.y - h), ImVec2(c.x + w, c.y + h), line, h, 0, 2.0f);
        }
        if (b.round) { /* the PlayStation symbols, drawn (fonts may lack them) */
            float k = 0.45f * r, w = 0.09f * r + 1.5f;
            if (b.bit == TRIANGLE)
                d->AddTriangle(ImVec2(c.x, c.y - k * 1.1f), ImVec2(c.x + k, c.y + k * 0.75f), ImVec2(c.x - k, c.y + k * 0.75f),
                    IM_COL32(64, 220, 170, 220), w);
            else if (b.bit == CIRCLE)
                d->AddCircle(c, k, IM_COL32(255, 100, 110, 220), 32, w);
            else if (b.bit == CROSS) {
                d->AddLine(ImVec2(c.x - k, c.y - k), ImVec2(c.x + k, c.y + k), IM_COL32(120, 170, 255, 220), w);
                d->AddLine(ImVec2(c.x + k, c.y - k), ImVec2(c.x - k, c.y + k), IM_COL32(120, 170, 255, 220), w);
            } else
                d->AddRect(ImVec2(c.x - k * 0.85f, c.y - k * 0.85f), ImVec2(c.x + k * 0.85f, c.y + k * 0.85f),
                    IM_COL32(240, 140, 220, 220), 0.0f, 0, w);
            continue;
        }
        float size = 0.62f * r * (b.bit == SELECT || b.bit == START ? 0.75f : 1.0f);
        ImFont* font = ImGui::GetFont();
        ImVec2 t = font->CalcTextSizeA(size * 1.6f, FLT_MAX, 0.0f, b.label);
        d->AddText(font, size * 1.6f, ImVec2(c.x - t.x * 0.5f, c.y - t.y * 0.5f), text, b.label);
    }
}
