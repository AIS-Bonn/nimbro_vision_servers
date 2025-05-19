import re
import numpy as np
from typing import Optional
from skimage.draw import polygon as sk_polygon

PATTERN_MAP = {
    "<CAPTION>":                         r"<CAPTION>",
    "<DETAILED_CAPTION>":                r"<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>":           r"<MORE_DETAILED_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>":     r"<CAPTION_TO_PHRASE_GROUNDING>.+",
    "<OD>":                              r"<OD>",
    "<DENSE_REGION_CAPTION>":            r"<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>":                 r"<REGION_PROPOSAL>",
    "<REFERRING_EXPRESSION_SEGMENTATION>": r"<REFERRING_EXPRESSION_SEGMENTATION>.+",
    "<REGION_TO_SEGMENTATION>":          r"<REGION_TO_SEGMENTATION>(?:<loc_\d+>){4}",
    "<OPEN_VOCABULARY_DETECTION>":       r"<OPEN_VOCABULARY_DETECTION>.+",
    "<REGION_TO_CATEGORY>":              r"<REGION_TO_CATEGORY>(?:<loc_\d+>){4}",
    "<REGION_TO_DESCRIPTION>":           r"<REGION_TO_DESCRIPTION>(?:<loc_\d+>){4}",
    "<OCR>":                             r"<OCR>",
    "<OCR_WITH_REGION>":                 r"<OCR_WITH_REGION>",
}

PATTERNS = {
    tag: re.compile(rf"^\s*{body}\s*$", re.I | re.S)
    for tag, body in PATTERN_MAP.items()
}

def get_prompt_type(text: str) -> Optional[str]:
    """
    Return the literal tag (e.g. "<CAPTION>", "<OD>", …) that the
    supplied `text` conforms to, or None if no pattern matches.

    >>> get_task_tag("<CAPTION>")
    '<CAPTION>'
    >>> get_task_tag("<REGION_TO_CATEGORY><loc_1><loc_2><loc_3><loc_4>")
    '<REGION_TO_CATEGORY>'
    >>> get_task_tag("invalid")
    None
    """
    for tag, pattern in PATTERNS.items():
        if pattern.fullmatch(text):
            return tag
    return None

def bbox_to_loc(bbox_xyxy, image_h, image_w):
    x1, y1, x2, y2 = bbox_xyxy
    locs = [int(1000*x1/image_w), int(1000*y1/image_h), int(1000*x2/image_w), int(1000*y2/image_h)]
    locs = [max(0, min(loc, 999)) for loc in locs]
    return ''.join([f"<loc_{str(loc)}>" for loc in locs])

def clean_string(s):
    s = re.sub(r'\\s', '', s)
    s = re.sub(r'<pad>', '', s)
    s = re.sub(r'<loc_\d+>', '', s)
    return s.rstrip()

def polygons_to_bbox_and_mask(list_of_polygons):
    """
    Convert a list of flat polygon coordinate lists into one (xmin, ymin, xmax, ymax)
    bbox and one binary mask covering the union of all the polygons.

    Parameters
    ----------
    list_of_polygons : list of lists (or arrays) of float
        Each polygon is [x1, y1, x2, y2, ..., xN, yN].

    Returns
    -------
    bbox_xyxy : list of int
        [xmin, ymin, xmax, ymax] for the union of all input polygons.
    mask : 2D numpy.ndarray, dtype bool
        Binary mask of shape (ymax–ymin+1, xmax–xmin+1), True inside any polygon.
    """
    if not list_of_polygons:
        raise ValueError("Need at least one polygon")

    # ---- 1) compute global bbox over all polys ----
    all_xs = np.hstack([np.array(poly[0::2], dtype=np.float32) for poly in list_of_polygons])
    all_ys = np.hstack([np.array(poly[1::2], dtype=np.float32) for poly in list_of_polygons])

    xmin, ymin = np.floor([all_xs.min(), all_ys.min()]).astype(int)
    xmax, ymax = np.ceil ([all_xs.max(), all_ys.max()]).astype(int)

    width  = xmax - xmin + 1
    height = ymax - ymin + 1

    # ---- 2) make empty mask ----
    mask = np.zeros((height, width), dtype=bool)

    # ---- 3) rasterize each polygon into the same mask ----
    for poly in list_of_polygons:
        xs = np.array(poly[0::2], dtype=np.float32) - xmin
        ys = np.array(poly[1::2], dtype=np.float32) - ymin

        rr, cc = sk_polygon(ys, xs, shape=mask.shape)
        mask[rr, cc] = True

    bbox_xyxy = [xmin.item(), ymin.item(), xmax.item() + 1, ymax.item() + 1]
    return bbox_xyxy, mask

def quad_to_xyxy(quad):
    """
    Convert a 4-point quad [x1, y1, x2, y2, x3, y3, x4, y4]
    into an axis-aligned bbox (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    quad : list or array of float
        [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns
    -------
    bbox_xyxy : list of int
        [xmin, ymin, xmax, ymax], with mins floored and maxes ceiled.
    """
    xs = np.array(quad[0::2], dtype=float)
    ys = np.array(quad[1::2], dtype=float)

    xmin = int(np.floor(xs.min()))
    ymin = int(np.floor(ys.min()))
    xmax = int(np.ceil (xs.max()))
    ymax = int(np.ceil (ys.max()))

    return xmin, ymin, xmax, ymax