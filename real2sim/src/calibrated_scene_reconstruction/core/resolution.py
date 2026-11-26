"""Pixel-size arithmetic shared by scene rendering and physics visualization."""


def scaled_even_size(width, height, scale):
    """Scale each axis and round to even pixels, preserving the original tie rounding."""
    return 2 * round(width * scale / 2), 2 * round(height * scale / 2)
