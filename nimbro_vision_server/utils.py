import io
import sys
import base64
import logging
import requests
import itertools
from pathlib import Path
from typing import List, Tuple, Literal

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw, ImageFont

# ── Misc. Utils ────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configures the root logger. Returns a named logger for the package.
    """
    # Basic config on the root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    # Create & return a module‐specific logger
    return logging.getLogger("nimbro_vision_server")

def encode_image(img_pil_or_arr, channels='rgb'):
    """
    Encode a PIL Image or NumPy array to a base64‐encoded PNG string.

    Parameters:
        img_pil_or_arr: PIL.Image.Image or np.ndarray
            The input image. If a NumPy array, it must be either:
              - 2D (H×W) for grayscale
              - 3D (H×W×3) for color
        channels: str, one of {'rgb', 'bgr'}
            Specifies the channel ordering of the input array. Ignored for PIL images.
            - 'rgb': array is already in R, G, B order
            - 'bgr': array is in B, G, R order and will be converted to RGB

    Returns:
        str: Base64‐encoded PNG data (ASCII string, without data URI prefix).
    """
    # 1. Convert array to PIL Image if needed
    if isinstance(img_pil_or_arr, np.ndarray):
        arr = img_pil_or_arr
        # Validate array shape
        if arr.ndim == 2:
            mode = 'L'
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = 'RGB'
            if channels.lower() == 'bgr':
                # swap B and R
                arr = arr[..., ::-1]
        else:
            raise ValueError(f"Unsupported array shape {arr.shape}; must be H×W or H×W×3")
        img = Image.fromarray(arr, mode=mode)
    elif isinstance(img_pil_or_arr, Image.Image):
        img = img_pil_or_arr
    else:
        raise TypeError(f"Unsupported type: {type(img_pil_or_arr)}")

    # 2. Ensure image is in RGB or L mode for PNG
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    # 3. Save to in-memory buffer as PNG
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # 4. Base64‐encode and return ASCII string
    b64_bytes = base64.b64encode(buffer.read())
    return b64_bytes.decode('ascii')

def encode_mask(mask):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError("Mask must be of ndarray with dtype=bool")
    mask_uint8 = mask.astype(np.uint8) * 255
    success, mask_png = cv2.imencode('.png', mask_uint8)
    if not success:
        raise RuntimeError("Failed to encode mask to PNG")
    return base64.b64encode(mask_png).decode('utf-8')

def decode_mask(mask):
    mask_bytes = base64.b64decode(mask)
    mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_np = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
    if mask_np is None:
        raise RuntimeError("Failed to decode PNG from base64")
    return mask_np > 0                                                                                                                                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                            
def load_image_b64(path: str | Path, output_format: str = "PNG") -> str:                                                                                                                                                                    
    """                                                                                                                                                                                                                                     
    Load an image from *path* and return a base-64–encoded string.                                                                                                                                                                          
                                                                                                                                                                                                                                            
    Parameters                                                                                                                                                                                                                              
    ----------                                                                                                                                                                                                                              
    path : str | pathlib.Path                                                                                                                                                                                                               
        Path to the image on disk.                                                                                                                                                                                                          
    output_format : str, default "PNG"                                                                                                                                                                                                      
        Format to re-encode the image in memory before encoding to base-64.                                                                                                                                                                 
        PNG is the default because it’s loss-less, but you can use                                                                                                                                                                          
        "JPEG", "WEBP", etc.                                                                                                                                                                                                                
                                                                                                                                                                                                                                            
    Returns                                                                                                                                                                                                                                 
    -------                                                                                                                                                                                                                                 
    str                                                                                                                                                                                                                                     
        Base-64 representation of the image (no data-URI prefix).                                                                                                                                                                           
                                                                                                                                                                                                                                            
    Raises                                                                                                                                                                                                                                  
    ------                                                                                                                                                                                                                                  
    FileNotFoundError                                                                                                                                                                                                                       
        If *path* does not exist.                                                                                                                                                                                                           
    """                                                                                                                                                                                                                                     
    path = Path(path)                                                                                                                                                                                                                       
    if not path.is_file():                                                                                                                                                                                                                  
        raise FileNotFoundError(f"Image not found: {path}")                                                                                                                                                                                 
                                                                                                                                                                                                                                            
    with Image.open(path) as im:                                                                                                                                                                                                            
        im = im.convert("RGB")          # standardize channels                                                                                                                                                                              
        buffer = io.BytesIO()                                                                                                                                                                                                               
        im.save(buffer, format=output_format)                                                                                                                                                                                               
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def load_url_b64(
    url: str,
    output_format: Literal["PNG", "JPEG", "WEBP"] = "PNG",
    timeout: int = 10,
) -> str:
    """
    Download an image at *url* and return a base-64–encoded string.

    Parameters
    ----------
    url : str
        Direct (raw) URL to the image.
    output_format : {"PNG", "JPEG", "WEBP"}, default "PNG"
        Format to re-encode the image in memory before converting to base-64.
        PNG is loss-less; pick JPEG/WEBP if size matters more than fidelity.
    timeout : int, default 10
        Seconds to wait for the HTTP response before aborting.

    Returns
    -------
    str
        Base-64 representation of the image (no data-URI prefix).

    Raises
    ------
    requests.HTTPError
        If the server responds with an error status (4xx / 5xx).
    requests.Timeout
        If the request exceeds *timeout* seconds.
    PIL.UnidentifiedImageError
        If the response body is not a valid image.
    """
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()                 # 4xx / 5xx → exception
        with Image.open(resp.raw) as im:
            im = im.convert("RGB")              # normalise channels
            buf = io.BytesIO()
            im.save(buf, format=output_format)
            return base64.b64encode(buf.getvalue()).decode("utf-8")



# ── Visualization Utils ────────────────────────────────────────────────────────────

def _text_dims(draw: ImageDraw.ImageDraw, text: str, font):
    """Return (width, height) of *text* in the current Pillow version."""
    if hasattr(draw, "textbbox"):                     # Pillow ≥ 10
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    else:                                             # Pillow < 10
        return draw.textsize(text, font=font)


def visualize_mmgroundingdino(
        payload: dict,
        output: dict,
        score_threshold: float = 0.0,
        figsize=(8, 8),
        palette=((243, 195, 0), (135, 86, 146), (243, 132, 0),
                 (161, 202, 241), (190, 0, 50), (194, 178, 128),
                 (132, 132, 130), (0, 136, 86), (230, 143, 172),
                 (0, 103, 165), (249, 147, 121), (96, 78, 151),
                 (246, 166, 0), (179, 68, 108), (220, 211, 0),
                 (136, 45, 23), (141, 182, 0), (101, 69, 34),
                 (226, 88, 34), (43, 61, 38)),
        font: ImageFont.ImageFont | None = None
    ):
    """
    Draw boxes & labels for mmgroundingdino.
    """
    if font is None:                      
        font = ImageFont.load_default()

    images_b64     = payload["images"]
    detections_all = output["artifact"]["detections"]

    colour_cycle = itertools.cycle(palette)
    prompt2color = {}

    annotated_imgs = []
    for img_b64, detections in zip(images_b64, detections_all):

        img   = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        draw  = ImageDraw.Draw(img, "RGBA")
        w_img, h_img = img.size

        for det in (d for d in detections if d["confidence"] >= score_threshold):
            prompt = det["prompt"]
            colour = prompt2color.setdefault(prompt, next(colour_cycle))

            x1, y1, x2, y2 = map(int, det["box_xyxy"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            draw.rectangle([x1, y1, x2, y2], outline=colour + (255,), width=3)

            label = f"{prompt} {det['confidence']:.2f}"
            text_w, text_h = _text_dims(draw, label, font)
            draw.rectangle([x1, y1, x1 + text_w + 6, y1 + text_h + 4],
                           fill=colour + (160,))
            draw.text((x1 + 3, y1 + 2), label, font=font, fill=(0, 0, 0, 255))

        annotated_imgs.append(img)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return annotated_imgs

def visualize_sam2_prompts(
        payload: dict,
        marker_size: int = 200,
        linewidth: int = 2,
        figsize: tuple = (10, 8),
        color_pos: str = "lime",
        color_neg: str = "red",
        label_offset: int = 10
    ):
    """
    Visualise point / bbox prompts contained in *payload*.

    Parameters
    ----------
    payload : dict
        {
          'image'  : <base-64 string>,
          'prompts': [ {object_id, bbox?, points?, labels?}, ... ]
        }

    • Positive points (label == 1) → green star, ID above
    • Negative points (label == 0) → red   star, ID above
    • Bounding boxes               → green outline, ID above
    """
    # --- unpack --------------------------------------------------------------
    img_b64  = payload.get("image")
    prompts  = payload.get("prompts", None)

    if img_b64 is None:
        raise ValueError("payload is missing an 'image' key")
    if not prompts:
        raise ValueError("'prompts' list is empty")

    # decode base-64 → PIL.Image
    image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

    # --- drawing setup -------------------------------------------------------
    txt_outline = [pe.withStroke(linewidth=2, foreground="black")]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    # --- iterate over every prompt ------------------------------------------
    for prm in prompts:
        oid = prm["object_id"]

        # bounding box
        if prm.get("bbox") is not None:
            x1, y1, x2, y2 = map(float, prm["bbox"])
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=linewidth,
                edgecolor=color_pos,
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - label_offset,
                f"ID {oid}",
                fontsize=12,
                color="white",
                path_effects=txt_outline,
                va="bottom"
            )

        # points
        if prm.get("points"):
            pts    = np.asarray(prm["points"], dtype=float)
            labels = np.asarray(prm.get("labels", [1] * len(pts)), dtype=int)

            # positive
            pos_pts = pts[labels == 1]
            if pos_pts.size:
                ax.scatter(
                    pos_pts[:, 0], pos_pts[:, 1],
                    color=color_pos,
                    marker="*",
                    s=marker_size,
                    edgecolor="white",
                    linewidth=1.25,
                    zorder=3
                )
                for x, y in pos_pts:
                    ax.text(
                        x, y - label_offset,
                        str(oid),
                        fontsize=max(marker_size // 25, 10),
                        color="white",
                        ha="center", va="bottom",
                        path_effects=txt_outline,
                        zorder=4
                    )

            # negative
            neg_pts = pts[labels == 0]
            if neg_pts.size:
                ax.scatter(
                    neg_pts[:, 0], neg_pts[:, 1],
                    color=color_neg,
                    marker="*",
                    s=marker_size,
                    edgecolor="white",
                    linewidth=1.25,
                    zorder=3
                )
                for x, y in neg_pts:
                    ax.text(
                        x, y - label_offset,
                        str(oid),
                        fontsize=max(marker_size // 25, 10),
                        color="white",
                        ha="center", va="bottom",
                        path_effects=txt_outline,
                        zorder=4
                    )

    plt.tight_layout()
    plt.show()
    return fig, ax

def visualize_sam2(
        payload: dict,
        output:  dict,
        linewidth: int         = 2,
        fontsize:  int         = 12,
        mask_alpha: float      = 0.40,
        draw_contour: bool     = True,
        contour_alpha: float   = 0.80,
        contour_thickness: int = 1,
        palette=((243, 195,   0), (135,  86, 146), (243, 132,   0),
                 (161, 202, 241), (190,   0,  50), (194, 178, 128),
                 (132, 132, 130), (  0, 136,  86), (230, 143, 172),
                 (  0, 103, 165), (249, 147, 121), ( 96,  78, 151),
                 (246, 166,   0), (179,  68, 108), (220, 211,   0),
                 (136,  45,  23), (141, 182,   0), (101,  69,  34),
                 (226,  88,  34), ( 43,  61,  38))
    ) -> List[Tuple[plt.Figure, plt.Axes]]:

    imgs_b64 = payload.get("images", [payload.get("image")])
    track_sets = output["artifact"]["tracks"]
    if len(imgs_b64) != len(track_sets):
        raise ValueError("Number of images and track groups differ")

    palette = np.asarray(palette, dtype=float) / 255.0
    n_col   = len(palette)
    figs_axes = []

    for img_b64, tracks in zip(imgs_b64, track_sets):

        rgb  = np.array(Image.open(io.BytesIO(base64.b64decode(img_b64))))
        vis  = rgb.astype(float) / 255.0                       # float 0-1

        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr["box_xyxy"])
            h, w = y2 - y1, x2 - x1
            if h < 0 or w < 0:
                continue
            tid  = tr.get("track_id", 0)
            col_rgb = palette[tid % n_col]                     # float RGB
            col_bgr = tuple(int(c*255) for c in col_rgb[::-1]) # OpenCV BGR

            # decode binary mask
            mask = decode_mask(tr["mask"])

            # -------- mask fill ------------------------------------------
            alpha = (mask * mask_alpha)[..., None]
            region = vis[y1:y2, x1:x2]
            vis[y1:y2, x1:x2] = (1 - alpha) * region + alpha * col_rgb

            # -------- contour (OpenCV) ------------------------------------
            if draw_contour:
                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                blank = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.drawContours(blank, contours, -1, col_bgr,
                                 thickness=contour_thickness)

                blank_rgb = blank[..., ::-1].astype(float) / 255.0
                contour_mask = (blank_rgb.sum(-1, keepdims=True) > 0).astype(float)

                region = vis[y1:y2, x1:x2]
                vis[y1:y2, x1:x2] = ((1 - contour_mask*contour_alpha) * region +
                                     (contour_mask*contour_alpha)  * blank_rgb)

        # ---------------------------- plot -------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow((vis * 255).astype(np.uint8))
        ax.axis("off")

        for tr in tracks:
            x1, y1, x2, y2 = tr["box_xyxy"]
            tid  = tr.get("track_id", 0)
            col_rgb = palette[tid % n_col]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=linewidth,
                                     edgecolor=col_rgb,
                                     facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"ID {tid}", color="white",
                    fontsize=fontsize, va="bottom",
                    bbox=dict(boxstyle="square,pad=0.1",
                              fc=col_rgb, ec="none", alpha=0.8))

        plt.tight_layout()
        plt.show()
        figs_axes.append((fig, ax))

    return figs_axes

def visualize_kosmos2(
        payload: dict,
        output: dict,
        figsize=(8, 8),
        palette=((243, 195, 0), (135, 86, 146), (243, 132, 0),
                 (161, 202, 241), (190, 0, 50), (194, 178, 128),
                 (132, 132, 130), (0, 136, 86), (230, 143, 172),
                 (0, 103, 165), (249, 147, 121), (96, 78, 151),
                 (246, 166, 0), (179, 68, 108), (220, 211, 0),
                 (136, 45, 23), (141, 182, 0), (101, 69, 34),
                 (226, 88, 34), (43, 61, 38)),
        font: ImageFont.ImageFont | None = None
    ):
    """
    Draw boxes & labels for mmgroundingdino.
    """
    if font is None:                      
        font = ImageFont.load_default()

    images_b64     = payload["images"]
    detections_all = output["artifact"]["detections"]
    captions   = output["artifact"]["captions"]

    colour_cycle = itertools.cycle(palette)
    prompt2color = {}

    annotated_imgs = []
    for img_b64, caption, detections in zip(images_b64, captions, detections_all):

        img   = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        draw  = ImageDraw.Draw(img, "RGBA")
        w_img, h_img = img.size

        for det in detections:
            label = det["label"]
            colour = prompt2color.setdefault(label, next(colour_cycle))

            x1, y1, x2, y2 = map(int, det["box_xyxy"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            draw.rectangle([x1, y1, x2, y2], outline=colour + (255,), width=3)

            label = f"{label}"
            text_w, text_h = _text_dims(draw, label, font)
            draw.rectangle([x1, y1, x1 + text_w + 6, y1 + text_h + 4],
                           fill=colour + (160,))
            draw.text((x1 + 3, y1 + 2), label, font=font, fill=(0, 0, 0, 255))

        annotated_imgs.append(img)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.title(caption)
        plt.tight_layout()
        plt.show()

    return annotated_imgs

def visualize_florence2(
        payload: dict,
        output:  dict,
        linewidth: int         = 2,
        fontsize:  int         = 12,
        mask_alpha: float      = 0.40,
        draw_contour: bool     = True,
        contour_alpha: float   = 0.80,
        contour_thickness: int = 1,
        palette=((243, 195,   0), (135,  86, 146), (243, 132,   0),
                 (161, 202, 241), (190,   0,  50), (194, 178, 128),
                 (132, 132, 130), (  0, 136,  86), (230, 143, 172),
                 (  0, 103, 165), (249, 147, 121), ( 96,  78, 151),
                 (246, 166,   0), (179,  68, 108), (220, 211,   0),
                 (136,  45,  23), (141, 182,   0), (101,  69,  34),
                 (226,  88,  34), ( 43,  61,  38))
    ) -> List[Tuple[plt.Figure, plt.Axes]]:
    """
    Visualise detections/captions produced by Florence-2.

    Parameters
    ----------
    payload : dict
        The request payload containing base-64 encoded image(s).
    output : dict
        The model response with one list of detections and one caption per image.

    Returns
    -------
    list[(Figure, Axes)]
        A list of (figure, axes) pairs – one per input image.
    """

    # ------------------------------------------------------------------ setup
    imgs_b64 = payload["images"]
    if not isinstance(imgs_b64, list):
        imgs_b64 = [imgs_b64]
    detections_ls = output["artifact"]["detections"]
    captions      = output["artifact"]["captions"]

    if len(imgs_b64) != len(detections_ls):
        raise ValueError("Number of images and detections differ")
    if len(imgs_b64) != len(captions):
        raise ValueError("Number of images and captions differ")

    palette = np.asarray(palette, dtype=float) / 255.0
    n_col   = len(palette)
    figs_axes = []

    for img_idx, (img_b64, dets, caption) in enumerate(zip(
            imgs_b64, detections_ls, captions)):

        # ---------- image -----------------------------------------------------------------
        rgb  = np.array(Image.open(io.BytesIO(base64.b64decode(img_b64))))
        vis  = rgb.astype(float) / 255.0                       # float 0-1

        # ---------- draw masks first ------------------------------------------------------
        for det_idx, det in enumerate(dets):
            if "mask" not in det:
                continue
            x1, y1, x2, y2 = map(int, det["box_xyxy"])
            h, w = y2 - y1, x2 - x1
            if h <= 0 or w <= 0:
                continue

            # colour for this detection
            col_rgb = palette[det_idx % n_col]
            col_bgr = tuple(int(c * 255) for c in col_rgb[::-1])

            mask = decode_mask(det["mask"])

            # blend mask
            alpha  = (mask * mask_alpha)[..., None]
            region = vis[y1:y2, x1:x2]
            vis[y1:y2, x1:x2] = (1 - alpha) * region + alpha * col_rgb

            # optional contour
            if draw_contour:
                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.drawContours(blank, contours, -1, col_bgr,
                                 thickness=contour_thickness)

                blank_rgb = blank[..., ::-1].astype(float) / 255.0
                contour_mask = (blank_rgb.sum(-1, keepdims=True) > 0)
                region = vis[y1:y2, x1:x2]
                vis[y1:y2, x1:x2] = ((1 - contour_mask*contour_alpha) * region +
                                     (contour_mask*contour_alpha)  * blank_rgb)

        # ---------- figure/axes -----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow((vis * 255).astype(np.uint8))
        ax.axis("off")

        # optional title from caption
        if caption:
            ax.set_title(caption, fontsize=fontsize + 2, pad=12)

        # ---------- draw boxes + labels ---------------------------------------------------
        for det_idx, det in enumerate(dets):
            if "box_xyxy" not in det:
                continue
            x1, y1, x2, y2 = map(int, det["box_xyxy"])
            col_rgb = palette[det_idx % n_col]

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=linewidth,
                                     edgecolor=col_rgb,
                                     facecolor="none")
            ax.add_patch(rect)

            label = det.get("label")
            if label:
                text_bg = dict(boxstyle="square,pad=0.1",
                               fc=col_rgb, ec="none", alpha=0.8)
                ax.text(x1, y1 - 5, label,
                        color="white", fontsize=fontsize, va="bottom",
                        bbox=text_bg)

        plt.tight_layout()
        plt.show()
        figs_axes.append((fig, ax))

    return figs_axes