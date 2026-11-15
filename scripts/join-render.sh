#!/usr/bin/env bash
# Join the two half-renders into signal_noise.mp4 and rebuild the storyboard.
#
#   bash scripts/join-render.sh [parts-dir]
#
# The renderer is single-threaded and this sandbox has 2 cores, so the edit is
# rendered as 0–15.5 s and 15.5–31 s in parallel and joined here. Both halves
# come from the same encoder settings, so the join is a stream copy — no
# re-encode, no generation loss.
set -euo pipefail

DIR="${1:-edits/render}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/signal_noise.mp4"
FF="$(python3 -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"

[ -f "$DIR/signal_noise_p1.mp4" ] || { echo "missing $DIR/signal_noise_p1.mp4"; exit 1; }
[ -f "$DIR/signal_noise_p2.mp4" ] || { echo "missing $DIR/signal_noise_p2.mp4"; exit 1; }

printf "file '%s'\n" "$(basename "$DIR/signal_noise_p1.mp4")" \
                    "$(basename "$DIR/signal_noise_p2.mp4")" > "$DIR/parts.txt"

# video: concat segments. audio: concat the same way, then mux once.
"$FF" -hide_banner -loglevel error -y -f concat -safe 0 -i "$DIR/parts.txt" \
      -c copy "$DIR/_video.mp4"
"$FF" -hide_banner -loglevel error -y -f concat -safe 0 -i "$DIR/parts.txt" \
      -vn -c:a copy "$DIR/_audio.m4a"
# -t 31 exactly: concatenating audio adds a frame of padding at the join, which
# would otherwise report 31.02 s.
"$FF" -hide_banner -loglevel error -y -i "$DIR/_video.mp4" -i "$DIR/_audio.m4a" \
      -map 0:v:0 -map 1:a:0 -c:v copy -c:a copy -t 31 "$OUT"
rm -f "$DIR/_video.mp4" "$DIR/_audio.m4a" "$DIR/parts.txt"

# storyboard straight from the finished file
python3 - "$OUT" "$DIR/storyboard.png" <<'PY'
import subprocess, sys, os
from PIL import Image
import numpy as np
import imageio_ffmpeg

mp4, dst = sys.argv[1], sys.argv[2]
ff = imageio_ffmpeg.get_ffmpeg_exe()
ts = [1, 4, 8, 12, 17, 21, 26, 29]
tiles = []
for t in ts:
    p = f"/tmp/_sb_{t}.png"
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t),
                    "-i", mp4, "-frames:v", "1", p], check=True)
    im = Image.open(p).convert("RGB")
    a = np.asarray(im.convert("L"), float)
    print(f"  {t:>2}s  mean {a.mean():5.1f}  std {a.std():5.1f}")
    tiles.append(im)
    os.remove(p)
tiles = [t.resize((270, 338)) for t in tiles]
sheet = Image.new("RGB", (4 * 270, 2 * 338), (16, 16, 20))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 4) * 270, (i // 4) * 338))
sheet.save(dst)
print(f"· storyboard → {dst}")
PY

python3 - "$OUT" <<'PY'
import sys
sys.path.insert(0, ".")
from craftmotion.core.media import probe_media
info = probe_media(sys.argv[1])
print("·", info)
assert abs(info["duration_ms"] - 31000) <= 40, f"expected 31000 ms, got {info['duration_ms']}"
assert info["has_audio"], "no audio stream"
print("✓ 31.00 s, audio present")
PY
