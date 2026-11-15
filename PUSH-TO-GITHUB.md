# Ship it: zip → GitHub → Windows `.exe`

This folder is a complete, git-ready copy of CraftMotion v1.1.0 (source, tests,
PyInstaller spec, WiX installer, GitHub Actions workflow, and the SIGNAL / NOISE
edit + its MP4).

The build has to happen on a Windows runner — PyInstaller cannot cross-compile
from Linux — so GitHub Actions is the cheap way to get the `.exe` without a
Windows machine of your own.

---

## 1. Create the repo and push

### With the GitHub CLI (fastest)

```bash
# one-time: install https://cli.github.com/ then
gh auth login

cd craftmotion
git init -b main
git add .
git commit -m "CraftMotion 1.1.0 + SIGNAL / NOISE edit"
gh repo create craftmotion --private --source . --remote origin --push
```

Or just run the helper, which does all of the above and then starts the build:

```bash
bash scripts/push-to-github.sh craftmotion        # name of the repo to create
```

### Without `gh` (plain git)

1. Create an empty repo at <https://github.com/new> — **no** README, no
   .gitignore (this folder already has one).
2. Then:

```bash
cd craftmotion
git init -b main
git add .
git commit -m "CraftMotion 1.1.0 + SIGNAL / NOISE edit"
git remote add origin https://github.com/<you>/craftmotion.git
git push -u origin main
```

GitHub will ask for a **personal access token** as the password (Settings →
Developer settings → Personal access tokens → *classic*, scope `repo`). Your
account password will not work over HTTPS.

---

## 2. Get the `.exe`

Pushing to `main` already starts the workflow (it is also `workflow_dispatch`).

* Open **Actions → Build Windows app (.exe + .msi)**.
* First run takes ~6–10 min (pip + PyInstaller + WiX).
* When it goes green, open the run and download the artifact:

| Artifact | Contents |
|---|---|
| **CraftMotion-Windows** | `CraftMotion.exe` — portable, single file, ffmpeg bundled |
| **CraftMotion-MSI** | `CraftMotion-1.1.0-x64.msi` — installer, Start-menu + desktop shortcuts |

From the CLI instead:

```bash
gh run watch                    # follow the build
gh run download --name CraftMotion-Windows -D out
```

### If the run fails

The workflow runs `pytest tests -q` before building, so a red X is almost always
a test failure, not the build. Open the failing step's log; the suite is fast
(~5 s) and runs cross-platform.

The MSI steps are `continue-on-error: true` — WiX problems never block the
`.exe`.

---

## 3. Run it

`CraftMotion.exe` is a single file: double-click for the GUI, or use it headless.

```powershell
.\CraftMotion.exe                      # GUI
.\CraftMotion.exe doctor               # environment check
.\CraftMotion.exe mcp                  # MCP server on stdio (agent mode)
.\CraftMotion.exe --help
```

The MSI installs the same exe to `Program Files\CraftMotion` with shortcuts.

---

## 4. The edit

`edits/render/signal_noise.mp4` — 31 s, 1080×1350, 30 fps, AAC stereo.
`edits/render/storyboard.png` — 8 frames pulled from that MP4.
`edits/render/README.md` — sources, licences and what the test found.

Re-render it (needs the public-domain assets, ~185 MB, fetched on demand):

```bash
pip install numpy pillow imageio-ffmpeg gdown
python edits/fetch_assets.py --dest /tmp/pd_assets
python edits/samuifx_edit.py --assets /tmp/pd_assets \
    --presets /tmp/pd_assets/samuifx --out edits/render \
    --render --width 1080 --height 1350
```

On a 2-core box that is ~25 min (the renderer is single-threaded and CPU-bound);
a normal desktop is several times faster.
