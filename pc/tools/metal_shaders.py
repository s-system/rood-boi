#!/usr/bin/env python3
"""The Vulkan renderer's shaders for the Metal backend (pc/src/mtl_core.m):
each SPIR-V module of a generated vk_shaders.h (pc/tools/vk_shaders.py)
translated to MSL by SPIRV-Cross, compiled into one metallib, and a C table
(mtl_shaders.c) the backend looks shaders up in.

    python3 pc/tools/metal_shaders.py [--header pc/build/boot64-gb/vk_shaders.h]
        [--out pc/build/macos/metal] [--sdk macosx|iphoneos]

Needs spirv-cross (Homebrew) and Xcode's metal toolchain.

Resources: the backend keeps Vulkan's model (one push descriptor set, push
constants). Set 0 becomes a Metal 3 argument buffer (tier 2: every resource
is an 8-byte id or address, in the order SPIRV-Cross declares them), so the
binding -> argument id map must be exact: before translation each resource
variable is renamed rb<binding> in the SPIR-V (an OpName), which SPIRV-Cross
keeps (a combined image sampler's sampler is rb<binding>Smplr). The table
gives, per shader: its function, the buffer indices of the argument buffer
and of the push constants, every argument (binding, id, kind) and a compute
shader's workgroup size. A shader is found by its SPIR-V's size and hash
(vk_shaders.h's arrays are static: one copy per translation unit).
Vertex shaders flip y (Vulkan's clip space is y-down, Metal's y-up).
"""
import argparse
import hashlib
import pathlib
import re
import struct
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE_NAMES = {"vert": "vertex", "frag": "fragment", "comp": "kernel"}
OP_NAME, OP_EXECUTION_MODE, OP_VARIABLE, OP_DECORATE = 5, 16, 59, 71
DECORATION_BINDING, DECORATION_SET = 33, 34
EXECUTION_LOCAL_SIZE = 17
# Storage classes holding descriptors.
RESOURCE_CLASSES = {0, 2, 12}  # UniformConstant, Uniform, StorageBuffer
KINDS = {"texture": 1, "sampler": 2, "buffer": 3, "structure": 4, "atomic": 5}
# Helpers SPIRV-Cross defines as plain (not inline) functions: one copy per
# module would collide in the metallib, so each module gets its own name.
SHARED_HELPERS = ["spvMakeIntersectionParams"]


def read_header(path):
    """{name_stage: [words]} from vk_shaders.h."""
    text = path.read_text()
    shaders = {}
    for m in re.finditer(r"vk_spv_(\w+)_(vert|frag|comp|geom)\[\]\s*=\s*\{([^}]*)\}", text):
        shaders[(m.group(1), m.group(2))] = [int(w, 0) for w in re.findall(r"0x[0-9a-fA-F]+|\d+", m.group(3))]
    return shaders


def instructions(words):
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        yield i, op, words[i + 1:i + count]
        i += count


def name_words(target, text):
    data = text.encode() + b"\0"
    data += b"\0" * (-len(data) % 4)
    body = [target] + list(struct.unpack(f"<{len(data) // 4}I", data))
    return [(len(body) + 1) << 16 | OP_NAME] + body


def rename_resources(words):
    """The module with every set 0 resource variable named rb<binding>, its
    bindings and its LocalSize (compute)."""
    bindings, sets, variables, local = {}, {}, set(), (1, 1, 1)
    first_decoration = None
    for at, op, operands in instructions(words):
        if op == OP_DECORATE:
            if first_decoration is None:
                first_decoration = at
            if operands[1] == DECORATION_BINDING:
                bindings[operands[0]] = operands[2]
            elif operands[1] == DECORATION_SET:
                sets[operands[0]] = operands[2]
        elif op == OP_VARIABLE and operands[2] in RESOURCE_CLASSES:
            variables.add(operands[1])
        elif op == OP_EXECUTION_MODE and operands[1] == EXECUTION_LOCAL_SIZE:
            local = tuple(operands[2:5])
    named = {v: b for v, b in bindings.items() if v in variables and sets.get(v, 0) == 0}
    out = words[:5]
    for at, op, operands in instructions(words):
        if at == first_decoration:
            for variable, binding in sorted(named.items()):
                out += name_words(variable, f"rb{binding}")
        if op == OP_NAME and operands[0] in named:
            continue
        out += [words[at]] + list(operands)
    if first_decoration is None:
        raise ValueError("no decorations")
    return out, sorted(set(named.values())), local


def translate(name, stage, words, work, sdk):
    """MSL for one module: (function name, source)."""
    function = f"{name}_{stage}"
    spv = work / f"{function}.spv"
    spv.write_bytes(struct.pack(f"<{len(words)}I", *words))
    args = ["spirv-cross", "--msl", "--msl-version", "30000", "--msl-argument-buffers", "--msl-argument-buffer-tier",
            "1", "--rename-entry-point", "main", function, STAGE_NAMES[stage].replace("kernel", "comp")
            .replace("vertex", "vert").replace("fragment", "frag"), str(spv)]
    if stage == "vert":
        args.append("--flip-vert-y")
    if sdk == "iphoneos":
        args.append("--msl-ios")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{function}: {result.stderr.strip()}")
    source = result.stdout
    for helper in SHARED_HELPERS:
        source = re.sub(rf"\b{helper}\b", f"{helper}_{function}", source)
    return function, source


def layout(function, source):
    """(set buffer index, push buffer index, [(binding, id, kind)]) from the
    MSL: the argument buffer struct's members and the entry's parameters."""
    arguments = []
    struct_ = re.search(r"struct spvDescriptorSetBuffer0\s*\{(.*?)\};", source, re.S)
    if struct_:
        for line in struct_.group(1).strip().splitlines():
            m = re.search(r"\brb(\d+)(Smplr|_atomic)?\s*\[\[id\((\d+)\)\]\]", line)
            if not m:
                raise RuntimeError(f"{function}: unexpected argument {line.strip()}")
            if m.group(2) == "_atomic":  # an image's atomics: a buffer over the same memory (r32 linear texture)
                kind = "atomic"
            elif m.group(2) or line.strip().startswith("sampler"):
                kind = "sampler"
            elif "acceleration_structure" in line:
                kind = "structure"
            elif re.match(r"\s*(texture|depth)", line):
                kind = "texture"
            else:
                kind = "buffer"
            arguments.append((int(m.group(1)), int(m.group(3)), KINDS[kind]))
    entry = re.search(rf"\b{function}\((.*?)\)\s*\{{", source, re.S)
    if not entry:
        raise RuntimeError(f"{function}: no entry point")
    set_index = push_index = -1
    for param in entry.group(1).split(","):
        m = re.search(r"\[\[buffer\((\d+)\)\]\]", param)
        if not m:
            continue
        if "spvDescriptorSetBuffer0" in param:
            set_index = int(m.group(1))
        elif push_index < 0:
            push_index = int(m.group(1))
        else:
            raise RuntimeError(f"{function}: more than one other buffer")
    return set_index, push_index, arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--header", default=str(ROOT / "pc/build/boot64-gb/vk_shaders.h"))
    parser.add_argument("--out", default=str(ROOT / "pc/build/macos/metal"))
    parser.add_argument("--sdk", default="macosx", choices=["macosx", "iphoneos"])
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    work = out / "msl"
    work.mkdir(parents=True, exist_ok=True)
    entries, airs, failed = [], [], []
    for (name, stage), words in sorted(read_header(pathlib.Path(args.header)).items()):
        if stage == "geom":  # Metal has none: native_pull.vert replaces native.geom
            continue
        try:
            renamed, _, local = rename_resources(words)
            function, source = translate(name, stage, renamed, work, args.sdk)
            set_index, push_index, arguments = layout(function, source)
        except (RuntimeError, ValueError) as error:
            failed.append(str(error))
            continue
        metal = work / f"{function}.metal"
        metal.write_text(source)
        air = work / f"{function}.air"
        if not air.exists() or air.stat().st_mtime < metal.stat().st_mtime or air.read_bytes() == b"":
            result = subprocess.run(["xcrun", "-sdk", args.sdk, "metal", "-c", "-O3", str(metal), "-o", str(air)],
                                    capture_output=True, text=True)
            if result.returncode:
                failed.append(f"{function}: {result.stderr.strip()[:2000]}")
                continue
        airs.append(air)
        blob = struct.pack(f"<{len(words)}I", *words)
        digest = int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")
        entries.append((len(blob), digest, function, set_index, push_index, arguments, local))
    library = out / "rood.metallib"
    result = subprocess.run(["xcrun", "-sdk", args.sdk, "metallib", *map(str, airs), "-o", str(library)],
                            capture_output=True, text=True)
    if result.returncode:
        failed.append(f"metallib: {result.stderr.strip()}")
    lines = ["/* Generated by pc/tools/metal_shaders.py: the Metal backend's shader table. */",
             '#include "mtl_shaders.h"', ""]
    for size, digest, function, set_index, push_index, arguments, local in entries:
        if arguments:
            lines.append(f"static const mtl_argument {function}_arguments[] = {{ "
                         + ", ".join(f"{{ {b}, {i}, {k} }}" for b, i, k in arguments) + " };")
    lines += ["", "const mtl_shader_info mtl_shaders[] = {"]
    for size, digest, function, set_index, push_index, arguments, local in entries:
        table = f"{function}_arguments" if arguments else "0"
        lines.append(f'    {{ {size}u, 0x{digest:016x}ull, "{function}", {set_index}, {push_index}, {table}, '
                     f"{len(arguments)}, {{ {local[0]}, {local[1]}, {local[2]} }} }},")
    lines += ["};", f"const int mtl_shader_count = {len(entries)};", ""]
    (out / "mtl_shaders.c").write_text("\n".join(lines))
    print(f"{len(entries)} shaders -> {library}")
    for f in failed:
        print("failed:", f, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
