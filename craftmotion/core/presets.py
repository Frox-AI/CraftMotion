"""Built-in look presets (effect stacks) available to humans and agents alike."""
from __future__ import annotations

BUILTIN_LOOKS = {
    "cinematic_teal": {
        "label": "Cinematic Teal & Orange",
        "effects": [
            ("color_balance", {"shadows": 0.5, "midtones": 0.15, "highlights": 0.35}),
            ("brightness_contrast", {"brightness": -0.02, "contrast": 0.18}),
            ("saturation", {"saturation": 0.12, "vibrance": 0.25}),
            ("vignette", {"amount": 0.35, "feather": 0.55}),
            ("grain", {"amount": 0.06}),
        ],
    },
    "warm_film": {
        "label": "Warm Film",
        "effects": [
            ("temperature", {"temp": 0.25, "tint": 0.05}),
            ("curves", {"lift": 0.03, "gamma": 1.05, "gain": 1.02}),
            ("glow", {"radius": 18, "strength": 0.35, "threshold": 0.6}),
            ("grain", {"amount": 0.08}),
            ("vignette", {"amount": 0.3}),
        ],
    },
    "bw_noir": {
        "label": "B&W Noir",
        "effects": [
            ("greyscale", {"mix": 1.0}),
            ("brightness_contrast", {"contrast": 0.35}),
            ("vignette", {"amount": 0.5, "feather": 0.4}),
            ("grain", {"amount": 0.1}),
        ],
    },
    "vhs_retro": {
        "label": "VHS Retro",
        "effects": [
            ("vhs", {"wobble": 1.4, "noise": 0.2, "color_bleed": 8}),
            ("scanlines", {"spacing": 4, "opacity": 0.25}),
            ("chromatic", {"shift": 3, "angle": 0}),
            ("saturation", {"saturation": 0.25}),
        ],
    },
    "dreamy_glow": {
        "label": "Dreamy Glow",
        "effects": [
            ("bloom", {"radius": 30, "strength": 0.85, "threshold": 0.6}),
            ("blur", {"radius": 1.2}),
            ("saturation", {"saturation": -0.1, "vibrance": 0.2}),
        ],
    },
    "viral_shake": {
        "label": "Viral Shake",
        "effects": [
            ("jitter", {"amount": 14, "speed": 24, "rotation": 0.8}),
            ("zoom_blur", {"strength": 0.12}),
            ("chromatic", {"shift": 2.5}),
        ],
    },
    "glitch_pop": {
        "label": "Glitch Pop",
        "effects": [
            ("glitch", {"amount": 0.55, "block": 14}),
            ("chromatic", {"shift": 5}),
            ("scanlines", {"spacing": 3, "opacity": 0.15}),
        ],
    },
    "clean_crisp": {
        "label": "Clean & Crisp",
        "effects": [
            ("sharpen", {"amount": 0.45, "radius": 1.6}),
            ("brightness_contrast", {"contrast": 0.1}),
            ("saturation", {"vibrance": 0.12}),
        ],
    },
    "old_photo": {
        "label": "Old Photo",
        "effects": [
            ("sepia", {"amount": 0.8}),
            ("grain", {"amount": 0.15}),
            ("vignette", {"amount": 0.55, "feather": 0.35}),
            ("brightness_contrast", {"contrast": -0.1, "brightness": 0.04}),
        ],
    },
    "neon_night": {
        "label": "Neon Night",
        "effects": [
            ("color_balance", {"shadows": 0.65, "highlights": 0.3}),
            ("glow", {"radius": 14, "strength": 0.9, "threshold": 0.45}),
            ("saturation", {"saturation": 0.3, "vibrance": 0.4}),
            ("vignette", {"amount": 0.4}),
        ],
    },
}
