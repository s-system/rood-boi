#!/usr/bin/env python3
"""The iPhone app: build_macos.py --ios's executable in a signed bundle.

    python3 pc/tools/package_ios.py --shell PATH/TO/Shell.app [--out pc/build/ios/VagrantStory.app]
        [--install UDID]

The shell is an app Xcode built and signed for this team (any bundle id,
with the entitlements the port needs: com.apple.developer.kernel.
extended-virtual-addressing for the guest window at 0x7F00000000, and
increased-memory-limit), carrying its provisioning profile. Its executable
is replaced by pc/build/ios/rood, rood.metallib goes beside it, and the
game's read-only data goes in game/ (the disc image, its extracted files,
the lighting, assets, plugins and fonts: boot_runtime.c ios_files links
them into Documents/game, the working directory). The bundle is then
signed again with the shell's identity and entitlements. The disc and its
files come from this checkout (ignored by git) and only go to the device.
"""
import argparse
import pathlib
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ["disks", "data", "lighting", "pc/assets", "pc/plugins", "pc/third_party/fonts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shell", required=True)
    parser.add_argument("--out", default=str(ROOT / "pc/build/ios/VagrantStory.app"))
    parser.add_argument("--install", metavar="UDID", help="install it on this device (xcrun devicectl)")
    args = parser.parse_args()
    shell, app = pathlib.Path(args.shell), pathlib.Path(args.out)
    executable = plistlib.loads((shell / "Info.plist").read_bytes())["CFBundleExecutable"]
    if app.exists():
        for child in app.iterdir():  # keep game/ (big): rsync updates it
            if child.name != "game":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    app.mkdir(parents=True, exist_ok=True)
    for child in shell.iterdir():
        if child.name in (executable, "_CodeSignature"):
            continue
        (shutil.copytree if child.is_dir() else shutil.copy2)(child, app / child.name)
    shutil.copy2(ROOT / "pc/build/ios/rood", app / executable)
    shutil.copy2(ROOT / "pc/build/ios/rood.metallib", app / "rood.metallib")
    for item in DATA:
        source = ROOT / item
        if not source.exists():
            sys.exit(f"missing {source}")
        target = app / "game" / item
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", "--copy-links", f"{source}/", f"{target}/"], check=True)
    # The shell's identity and entitlements.
    info = subprocess.run(["codesign", "-dvv", str(shell)], capture_output=True, text=True).stderr
    identity = re.search(r"^Authority=(.+)$", info, re.M).group(1)
    entitlements = subprocess.run(["codesign", "-d", "--entitlements", ":-", str(shell)], capture_output=True).stdout
    with tempfile.NamedTemporaryFile(suffix=".plist", delete=False) as f:
        f.write(entitlements)
    subprocess.run(["codesign", "--force", "--sign", identity, "--entitlements", f.name, "--timestamp=none",
                    "--generate-entitlement-der", str(app)], check=True)
    print(f"{app} signed by {identity}")
    if args.install:
        subprocess.run(["xcrun", "devicectl", "device", "install", "app", "--device", args.install, str(app)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
