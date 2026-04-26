"""
Shared PDF utilities for all_type_parser modules.
"""


def is_not_watermark(obj: dict) -> bool:
    """
    Return True if this PDF object should be KEPT.
    Filters out NIHR 'Pre-submission' watermark characters, which are
    rendered in light gray (all RGB channels ~0.827) at very large font sizes.
    """
    if obj.get("object_type") != "char":
        return True
    color = obj.get("non_stroking_color")
    if isinstance(color, (list, tuple)) and len(color) == 3:
        r, g, b = color
        if abs(r - g) < 0.05 and abs(g - b) < 0.05 and r > 0.7:
            return False
    return True
