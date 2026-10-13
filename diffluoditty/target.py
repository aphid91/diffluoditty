"""Loading target images as premultiplied RGBA tensors."""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

NOTO_EMOJI_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/png/128/emoji_u{code}.png"

EMOJI = {
    "gecko": "\U0001F98E",
    "butterfly": "\U0001F98B",
    "fish": "\U0001F420",
    "smiley": "\U0001F600",
}


def load_image(path_or_url: str, max_size: int = 40) -> torch.Tensor:
    """Load an RGBA image, shrink it to fit in ``max_size``, premultiply alpha.

    Returns a (4, H, W) float tensor in [0, 1].
    """
    if str(path_or_url).startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as r:
            img = Image.open(io.BytesIO(r.read()))
    else:
        img = Image.open(Path(path_or_url))
    img = img.convert("RGBA")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_emoji(emoji: str = "gecko", max_size: int = 40) -> torch.Tensor:
    """Fetch an emoji from the Noto emoji repo, e.g. ``load_emoji("gecko")`` or ``load_emoji("🦎")``."""
    emoji = EMOJI.get(emoji, emoji)
    code = "_".join(f"{ord(c):x}" for c in emoji if c != "️")
    return load_image(NOTO_EMOJI_URL.format(code=code), max_size)


def pad_target(target: torch.Tensor, padding: int = 16) -> torch.Tensor:
    """Surround the target with empty space so the pattern has room to grow (and be damaged)."""
    return F.pad(target, (padding, padding, padding, padding))
