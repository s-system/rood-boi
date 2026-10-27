#!/usr/bin/env python3
"""Rewrite a preprocessed game unit (.i) for a guest base other than zero.

The 64-bit port keeps PS1 addresses in 32-bit integers and turns them back
into pointers by zero-extension, which needs guest memory mapped at its PS1
addresses. Where the low 4 GB cannot be mapped (macOS and iOS on arm64), the
runtime maps guest memory at ROOD_GUEST_BASE + address instead: the low 32
bits of a guest pointer are still its PS1 address, and only the integer to
pointer direction changes.

This pass finds every integer to pointer conversion in the unit with clang's
JSON AST (the same input pc/tools/lp64_audit.py reads) and wraps its operand
in __rood_guest_ptr(), which adds the base to a 32-bit address, maps a code
token back to its function (psx_types.h), and leaves zero (null) and full
64-bit host pointers alone. Pointer to integer
truncations need no change.

Usage: guest_base.py BASE UNIT.i [OUT.i]   (BASE e.g. 0x7f00000000)
"""
import json
import subprocess
import sys
from pathlib import Path

TARGET = "x86_64-linux-gnu"
PRELUDE = """\
extern unsigned long pc_psx_code_native(unsigned int token);
static __inline__ __attribute__((always_inline, unused)) unsigned long __rood_guest_ptr(unsigned long value)
{
    if ((value >> 32) || !value)
        return value;
    if ((unsigned int)value >= 0xF0000000u) /* a code token: psx_types.h */
        return pc_psx_code_native((unsigned int)value);
    return value | %(base)#xUL;
}
"""


def ast(data, extra=(), target=TARGET):
    result = subprocess.run(
        ["clang", "-target", target, "-fsyntax-only", "-w", "-ferror-limit=0", "-fgnuc-version=4.2.1",
         "-fno-builtin", *extra, "-Xclang", "-ast-dump=json", "-x", "c", "-"],
        input=data, capture_output=True)
    if not result.stdout:
        raise SystemExit(result.stderr.decode(errors="replace"))
    return json.loads(result.stdout)


def end_offset(range_):
    end = range_.get("end", {})
    end = end.get("expansionLoc", end)
    if "offset" not in end:
        return None
    return end["offset"] + end.get("tokLen", 0)


def begin_offset(range_):
    begin = range_.get("begin", {})
    begin = begin.get("expansionLoc", begin)
    return begin.get("offset")


def is_null_constant(node):
    while node.get("kind") in ("ImplicitCastExpr", "ParenExpr", "CStyleCastExpr"):
        node = (node.get("inner") or [{}])[0]
    return node.get("kind") == "IntegerLiteral" and node.get("value") == "0"


WIDE = {"long", "unsigned long", "long int", "unsigned long int", "long long", "unsigned long long",
        "long long int", "unsigned long long int", "long unsigned int", "long long unsigned int",
        "__int128", "unsigned __int128"}


def is_wide(node):
    """A 64-bit operand may hold a full host pointer (kept for reference: the
    conversion is the same either way)."""
    t = node.get("type", {})
    name = t.get("desugaredQualType", t.get("qualType", ""))
    name = name.replace("const ", "").replace("volatile ", "").strip()
    return name in WIDE


def conversions(root):
    """(begin, end, wide, implicit) for the operands of integer to pointer
    conversions; an implicit one keeps a pointer type (void*)."""
    found = []
    stack = [root]
    while stack:
        node = stack.pop()
        inner = node.get("inner") or []
        if node.get("castKind") == "IntegralToPointer" and inner and not is_null_constant(inner[0]):
            r = inner[0].get("range", {})
            b, e = begin_offset(r), end_offset(r)
            if b is not None and e is not None and b < e:
                found.append((b, e, is_wide(inner[0]), node.get("kind") == "ImplicitCastExpr"))
        stack.extend(inner)
    return found


def rewrite(data, base, target=TARGET):
    spans = conversions(ast(data, target=target))
    # Insert from the end so earlier offsets stay valid; nested spans are
    # fine because each edit only adds text at two points. At one offset the
    # closing text of an inner span goes before the opening of the next.
    # The operand converts to unsigned long as C converts it to a pointer:
    # a signed int sign-extends ((void*)-1 stays MAP_FAILED), an unsigned
    # one zero-extends; only values whose high 32 bits are zero get the base.
    opening = lambda wide, implicit: ("(void*)" if implicit else "") + "__rood_guest_ptr((unsigned long)("
    # Among edits at one offset, the one applied last ends up leftmost:
    # outer spans open left of inner ones and close right of them.
    edits = sorted([(b, 1, -e, opening(w, i)) for b, e, w, i in spans] + [(e, 0, -b, "))") for b, e, _, _ in spans],
                   reverse=True)
    edits = [(offset, text) for offset, _, _, text in edits]
    out = bytearray(data)
    for offset, text in edits:
        out[offset:offset] = text.encode()
    prelude = (PRELUDE % {"base": base}).encode()
    return prelude + bytes(out), len(spans)


def main():
    base = int(sys.argv[1], 0)
    unit = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else unit
    data, count = rewrite(unit.read_bytes(), base)
    out.write_bytes(data)
    print(f"{unit.name}: {count} conversions", file=sys.stderr)


if __name__ == "__main__":
    main()
