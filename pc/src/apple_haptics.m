/* The phone's haptics for the game's DualShock motors (boot_gamepad.c, when
 * no gamepad is connected; iPhone only, build_macos.py --ios). Core Haptics:
 * one continuous event, its intensity following the large motor (0-255) and
 * the small one (on: a firmer, sharper buzz), changed live through dynamic
 * parameters; stopped while both are off. */
#import <Foundation/Foundation.h>
#import <CoreHaptics/CoreHaptics.h>
#include <stdio.h>

static CHHapticEngine* engine;
static id<CHHapticAdvancedPatternPlayer> player;
static int running, unsupported;

static int ready(void)
{
    if (unsupported)
        return 0;
    if (engine)
        return 1;
    if (!CHHapticEngine.capabilitiesForHardware.supportsHaptics) {
        unsupported = 1;
        return 0;
    }
    NSError* error = nil;
    engine = [[CHHapticEngine alloc] initAndReturnError:&error];
    if (!engine) {
        fprintf(stderr, "[haptics] no engine: %s\n", error.localizedDescription.UTF8String);
        unsupported = 1;
        return 0;
    }
    engine.playsHapticsOnly = YES;
    engine.autoShutdownEnabled = YES;
    engine.stoppedHandler = ^(CHHapticEngineStoppedReason reason) { (void)reason; running = 0; player = nil; };
    engine.resetHandler = ^{ running = 0; player = nil; };
    fprintf(stderr, "[haptics] Core Haptics ready (the game's vibration)\n");
    return 1;
}
static int start(void)
{
    NSError* error = nil;
    if (![engine startAndReturnError:&error])
        return 0;
    CHHapticEventParameter* intensity = [[CHHapticEventParameter alloc]
        initWithParameterID:CHHapticEventParameterIDHapticIntensity value:1.0f];
    CHHapticEventParameter* sharpness = [[CHHapticEventParameter alloc]
        initWithParameterID:CHHapticEventParameterIDHapticSharpness value:0.3f];
    CHHapticEvent* event = [[CHHapticEvent alloc] initWithEventType:CHHapticEventTypeHapticContinuous
                                                         parameters:@[ intensity, sharpness ]
                                                       relativeTime:0
                                                           duration:30.0];
    CHHapticPattern* pattern = [[CHHapticPattern alloc] initWithEvents:@[ event ] parameters:@[] error:&error];
    player = pattern ? [engine createAdvancedPlayerWithPattern:pattern error:&error] : nil;
    if (!player)
        return 0;
    player.loopEnabled = YES;
    if (![player startAtTime:CHHapticTimeImmediate error:&error])
        return 0;
    running = 1;
    return 1;
}
void pc_haptics_motors(int small, int large)
{
    if (!ready())
        return;
    float strength = large / 255.0f;
    if (small && strength < 0.6f)
        strength = 0.6f;
    NSError* error = nil;
    if (strength <= 0.0f) {
        if (running && player)
            [player stopAtTime:CHHapticTimeImmediate error:&error];
        running = 0;
        return;
    }
    if (!running && !start())
        return;
    CHHapticDynamicParameter* intensity = [[CHHapticDynamicParameter alloc]
        initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:strength relativeTime:0];
    CHHapticDynamicParameter* sharpness = [[CHHapticDynamicParameter alloc]
        initWithParameterID:CHHapticDynamicParameterIDHapticSharpnessControl value:small ? 0.4f : -0.2f relativeTime:0];
    [player sendParameters:@[ intensity, sharpness ] atTime:CHHapticTimeImmediate error:&error];
}
