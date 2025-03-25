# ============================================================================
# Utility Functions
# ============================================================================

import einops
import numpy as np


def _normalize(x, min_val, max_val):
    """Normalize values from [min_val, max_val] to [0, 1]."""
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    """Unnormalize values from [0, 1] to [min_val, max_val]."""
    return x * (max_val - min_val) + min_val


def _parse_image(image) -> np.ndarray:
    """Convert image to standard format (H,W,C) with uint8 type."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image
