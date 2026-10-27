#!/bin/sh
# Play the native macOS build (pc/tools/build_macos.py) from the repository
# root: a window, Vulkan through MoltenVK (Homebrew's molten-vk and
# vulkan-loader) when present, else Apple's OpenGL. Extra arguments go to the
# game (--skip-movies...); ROOD_RENDERER=gl forces OpenGL.
cd "$(dirname "$0")/../.." || exit 1
loader=/opt/homebrew/lib/libvulkan.dylib
if [ -z "$ROOD_RENDERER" ] && [ -e "$loader" ]; then
    export ROOD_RENDERER=vulkan SDL_VULKAN_LIBRARY="$loader"
fi
exec pc/build/macos/rood --window "$@"
