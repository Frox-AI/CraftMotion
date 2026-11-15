#!/usr/bin/env python3
"""Fetch the public-domain / CC0 source material for the SAMUIFX-preset edit.

Everything downloaded here is either public domain (pre-1978 US cartoons whose
copyright was not renewed, NASA footage) or CC0 (public-domain dedication).

    python edits/fetch_assets.py --dest /tmp/pd_assets

Sources
-------
Cartoons  Internet Archive "classic_cartoons" collection (Fleischer, 1941-42).
          Superman: The Mechanical Monsters (1941) and Electric Earthquake (1942)
          are US public domain (copyright not renewed).
Footage   NASA Image & Video Library — US government work, public domain.
Music     "Midnight Club" by Alec Penn Radio — CC0 1.0 from the Internet Archive.
Presets   SAMUIFX "AE Inspired 5 Anime Edit Presets" (Alight Motion XML) as shared
          by the channel in the video description.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1rB9gAjZRKAaUQmomvx8WeNKIW-AIpUHs"

CARTOONS = {
    "superman1.mp4": ("superman_the_mechanical_monsters",
                      "superman_the_mechanical_monsters_512kb.mp4"),
    "superman2.mp4": ("superman_electric_earthquake",
                      "superman_electric_earthquake_512kb.mp4"),
}
NASA = {
    "solarflare.mp4": "GSFC_20160426_SDO_m12224_SolarFlare",
    "blackhole.mp4": "GSFC_20181002_SMBH_m13043_Simulation",
}
MUSIC = ("alecpennradio-phonk", "Midnight Club.mp3", "phonk.mp3")

# 20 s / 720x480 / 30 fps cuts used by samuifx_edit.py: (source, offset seconds, name).
# Offsets were picked by scanning frame brightness for usable footage; re-cutting
# from the same source at the same offset reproduces the edit exactly.
CUTS = [
    ("superman1.mp4", 300, "cart_mm.mp4", "silhouette in a doorway"),
    ("superman1.mp4", 175, "cart_mm2.mp4", "HOUSE OF JEWELS newspaper"),
    ("superman1.mp4", 505, "cart_mm3.mp4", "glowing jewel exhibit"),
    ("superman2.mp4", 275, "cart_ee.mp4", "machinery hall"),
    ("superman2.mp4", 295, "cart_ee2.mp4", "scientist at the control panel"),
    ("superman2.mp4", 55, "cart_ee3.mp4", "white-out flash"),
]


def _download(url: str, dest: str, label: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  ✓ {label} (cached)")
        return True
    print(f"  ↓ {label}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CraftMotion/1.1"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as out:
            shutil.copyfileobj(r, out)
        size = os.path.getsize(dest)
        print(f"    {size / 1e6:.1f} MB → {dest}")
        return size > 10_000
    except Exception as exc:  # noqa: BLE001
        print(f"    failed: {exc}")
        return False


def fetch_sources(dest: str) -> dict:
    os.makedirs(dest, exist_ok=True)
    got = {}
    for name, (ident, file) in CARTOONS.items():
        url = f"https://archive.org/download/{ident}/{urllib.parse.quote(file)}"
        got[name] = _download(url, os.path.join(dest, name), f"cartoon {ident}")
    for name, ident in NASA.items():
        for variant in ("~medium.mp4", "~small.mp4", "~mobile.mp4", "~orig.mp4"):
            url = (f"https://images-assets.nasa.gov/video/{ident}/{ident}{variant}")
            got.setdefault(name, False)
            if _download(url, os.path.join(dest, name), f"NASA {ident}{variant}"):
                got[name] = True
                break
    ident, file, local = MUSIC
    url = f"https://archive.org/download/{ident}/{urllib.parse.quote(file)}"
    got[local] = _download(url, os.path.join(dest, local), f"music {file}")
    return got


def cut_clips(dest: str) -> dict:
    """Produce the short 720x480 proxies the edit timeline references."""
    got = {}
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        ff = "ffmpeg"
    for src_name, offset, out_name, label in CUTS:
        src = os.path.join(dest, src_name)
        out = os.path.join(dest, out_name)
        if not os.path.exists(src):
            print(f"  ! {out_name}: {src_name} missing")
            got[out_name] = False
            continue
        if os.path.exists(out) and os.path.getsize(out) > 100_000:
            print(f"  ✓ {out_name} (cached)")
            got[out_name] = True
            continue
        print(f"  ✂ {out_name} ← {src_name} @{offset}s ({label})")
        proc = subprocess.run([
            ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(offset), "-i", src,
            "-t", "20", "-vf",
            "scale=720:480:force_original_aspect_ratio=decrease,pad=720:480:-1:-1,fps=30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-an", out], capture_output=True)
        ok = proc.returncode == 0 and os.path.getsize(out) > 100_000
        got[out_name] = ok
        if not ok:
            print("    failed:", proc.stderr.decode("utf-8", "ignore")[-300:])
    return got


def fetch_presets(dest: str) -> list:
    """Alight Motion XML presets shared by SAMUIFX (Google Drive folder)."""
    try:
        import gdown  # type: ignore
    except ImportError:
        print("  ! gdown not installed — run: pip install gdown")
        return []
    out_dir = os.path.join(dest, "samuifx")
    if os.path.isdir(out_dir) and any(f.endswith(".xml") for _, _, fs in os.walk(out_dir) for f in fs):
        print("  ✓ presets (cached)")
        return sorted(os.path.join(b, f) for b, _, fs in os.walk(out_dir)
                      for f in fs if f.endswith(".xml"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"  ↓ presets → {out_dir}")
    gdown.download_folder(DRIVE_FOLDER, output=out_dir, quiet=True, use_cookies=False)
    return sorted(os.path.join(b, f) for b, _, fs in os.walk(out_dir)
                  for f in fs if f.endswith(".xml"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="/tmp/pd_assets")
    ap.add_argument("--skip-presets", action="store_true")
    ap.add_argument("--skip-cuts", action="store_true",
                    help="do not (re)build the 720x480 cart_*.mp4 proxies")
    a = ap.parse_args(argv)
    print(f"CraftMotion · fetching edit assets into {a.dest}")
    got = fetch_sources(a.dest)
    if not a.skip_cuts:
        got.update(cut_clips(a.dest))
    presets = [] if a.skip_presets else fetch_presets(a.dest)
    print("\nsummary")
    for name, ok in got.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"  ✓ {len(presets)} Alight Motion XML presets")
    for p in presets:
        print(f"      {os.path.basename(p)}")
    return 0 if all(got.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
