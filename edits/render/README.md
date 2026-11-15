# SIGNAL / NOISE — a real-world test of CraftMotion

A 31 s, 1080×1350 vertical edit built **entirely through CraftMotion's command
registry** (`edits/samuifx_edit.py`), as a real-world test of the app.

It is an original edit. The Alight Motion presets from
[SAMUIFX's "AE INSPIRED 5 ANIME EDIT PRESETS"](https://www.youtube.com/watch?v=3R3KKItLbv4)
are used only as **look sources**: `preset.am_look` extracts one effect chain per
preset and applies it to our own layers. No frame of the creator's edit, and none
of the anime clips or licensed music from that video, is used.

| File | What |
|---|---|
| `signal_noise.mp4` | the edit — 31.00 s, 1080×1350, 30 fps, AAC stereo, 37 MB |
| `storyboard.png` | 8 frames pulled from that MP4 |
| `signal_noise.cmproj` | the editable project, opens in the GUI or via MCP |
| `signal_noise_p1.mp4` / `_p2.mp4` | the two halves it was rendered as (see *Rendering*) |

## Reproduce

```bash
pip install numpy pillow imageio-ffmpeg gdown
python edits/fetch_assets.py --dest /tmp/pd_assets
python edits/samuifx_edit.py \
    --assets /tmp/pd_assets --presets /tmp/pd_assets/samuifx \
    --out edits/render --render --width 1080 --height 1350
```

`fetch_assets.py` is idempotent: it downloads the sources, cuts the six 720×480
proxies at fixed offsets (so the edit stays reproducible) and pulls the six AM
XML presets from the creator's Drive folder. `/tmp` does not survive a restart —
re-run it before rendering again.

### Rendering

The renderer is single-threaded and CPU-bound (~2.5 s/frame at 1080×1350 on two
cores, ~25 min for the film). On a 2-core box the film is rendered as two halves
and joined without re-encoding:

```bash
python edits/samuifx_edit.py ... --start 0     --end 15500 --suffix _p1 --no-stills &
python edits/samuifx_edit.py ... --start 15500 --end 31000 --suffix _p2 --no-stills &
wait
bash scripts/join-render.sh edits/render        # concat + storyboard + duration assert
```

> Two halves at once need >1 GB. On this 1.9 GB sandbox the second one got
> OOM-killed twice; running them one after the other, or setting
> `CRAFTMOTION_RENDER_CHUNK=45`, fixes it.

## Sources (all public domain / CC0)

| Element | Source | Licence |
|---|---|---|
| `cart_mm*.mp4` — machinery, jewel exhibit, newspaper | *Superman: The Mechanical Monsters* (Fleischer, 1941), Internet Archive `classic_cartoons` | US public domain — copyright not renewed |
| `cart_ee*.mp4` — control room, machinery, white-out | *Superman: Electric Earthquake* (Fleischer, 1942), same collection | US public domain — copyright not renewed |
| `solarflare.mp4` | NASA SDO solar flare, 2016 (`GSFC_20160426_SDO_m12224`) | US government work — public domain |
| `blackhole.mp4` | NASA binary black-hole simulation, 2018 (`GSFC_20181002_SMBH_m13043`) | US government work — public domain |
| `phonk.mp3` | "Midnight Club" — Alec Penn Radio, Internet Archive `alecpennradio-phonk` | **CC0 1.0** |
| Preset XMLs | SAMUIFX Drive folder shared in the video description | look sources, credited |

## Structure

| Time | Section | Look (from AM preset) |
|---|---|---|
| 0.0–2.3 s | SIGNAL / NOISE title, cascade + type-on, rgb_split | X-RAY → `thermal → glow` |
| 2.3–7.0 s | "01 · THE MECHANICAL MONSTERS" — jewel exhibit, X-ray | X-RAY |
| 7.0–10.3 s | "EXHIBIT OPENS TO-DAY" — newspaper as an editorial card | Nagumo → `levels → turbulence → exposure` |
| 10.3–15.0 s | 12-cut flash montage locked to detected onsets | Sukuna / Gojo |
| 15.0–19.5 s | "DEEP FIELD" — NASA solar flare, bloom + starfield | REZE → `tile → levels → zoom_blur` |
| 19.5–24.0 s | "EVENT HORIZON" — NASA binary SMBH, lens flare | Gojo |
| 24.0–27.6 s | "CONTROL ROOM" — scientist at the panel, grain + vignette | Sukuna |
| 27.6–31.0 s | CRAFTMOTION outro, credits, audio spectrum analyser | — |

Audio: `audio.beats` (threshold 0.28 → 45 beats) drives every scale punch; the
music is mastered to `audio.master{gain_db:2.5, ceiling_db:-1}` then
`audio.normalize{-14}`. Captions are burned in via `caption.add`.

## Verification

Checked against the finished file, not the render log:

* `ffprobe` — 00:00:31.02 container, **930 video frames**, 1080×1350, 30 fps,
  h264 High + AAC 48 kHz stereo.
* `volumedetect` — mean −17.2 dB, peak −1.0 dB (mastered, not clipping).
* Frame luminance sampled across the film: 7, 51, 45, 146, 117, 55, 35, 6 —
  no black frames, no clipped white ones.
* The 15.5 s join was inspected frame by frame; it falls on a black beat inside
  the strobe montage, so it is invisible.

## Bugs this test found (and fixed)

The point of doing a real edit was to hit the engine hard. It worked:

1. **`zoom_blur` took 6.1 s per frame** at 1080×1350 — it remapped the whole
   frame 14 times per sample. Replaced with averaged PIL resizes (the same
   operation in C): **0.5 s, ~10× faster**, mean difference 5.6/255. This alone
   was 80 % of the render time for the deep-field section.
2. **`project.new` takes `duration_ms`** — passing `duration` silently produced
   a 15 s project.
3. **`effect.add` silently dropped unknown parameters** — now raises naming the
   bad key and the valid ones, and clamps floats to their declared range.
4. **`preset.am_look` had no way to skip effects** — added `exclude`, so a
   preset's `invert`/`blink`/`turbulence` can be dropped without losing the look.
5. **Short AM names were rejected** (`rgbsep`, `randomdisplace`,
   `turbulentdisplace3`) — alias tables are now rebuilt after the AM-native
   modules register, so 739 bare-name aliases resolve.
6. **ffmpeg stderr deadlocked on long encodes** (unread pipe) — it goes to a
   temp file now.
7. **Render was OOM-killed mid-encode** — a streaming raw pipe made ffmpeg hold
   frames in native memory, then SIGKILL at ~1.8 GB and a silently truncated
   12.9 s file. `render_video` renders in chunks to PNGs and encodes each chunk;
   peak RSS is ~285 MB.
8. **`Renderer.video_sources` was a plain dict but the LRU patch called
   `move_to_end`** — every video layer raised inside render and was swallowed,
   producing a full-length *black* export while previews looked fine. Now an
   `OrderedDict`, and export mode re-raises layer errors instead of writing
   black frames. `render_video` probes its own output and fails on a short
   duration.
9. **Stills were previews, not exports** — QC images could look right while the
   MP4 was wrong. Stills now render in export mode.
10. **`drift()` and `punch()` both wrote `scale` keys**, so a keyframe overshoot
    blanked the headline card — `drift` now leaves `scale` alone unless asked.
