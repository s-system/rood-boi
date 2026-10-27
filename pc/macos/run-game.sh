#!/bin/sh
# Play the native macOS build (pc/tools/build_macos.py) from the repository
# root: a window, the renderer on Metal (pc/src/mtl_core.m; ROOD_RENDERER=gl
# forces Apple's OpenGL). Extra arguments go to the game (--skip-movies...).
cd "$(dirname "$0")/../.." || exit 1
export ROOD_RENDERER="${ROOD_RENDERER:-metal}"
# Settings saved across runs, as pc/run-game.sh keeps them.
mkdir -p pc/user
export ROOD_SETTINGS="${ROOD_SETTINGS:-pc/user/settings.cfg}"
exec pc/build/macos/rood --window "$@"
