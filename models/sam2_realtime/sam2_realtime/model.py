import os
import io
import base64
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2_camera_predictor
from sam2.utils.misc import mask_to_box

from nimbro_vision_server.model_base import BaseModel
from nimbro_vision_server.utils import setup_logging, encode_mask

logger = setup_logging()

class Model(BaseModel):
    """
    Wrapper for the SAM2-realtime model.
    Implements the BaseModel interface so it can be loaded, invoked,
    and unloaded by the server framework.
    """

    @classmethod
    def get_available_flavors(cls) -> List[str]:
        # Define the supported model sizes or variants
        return ["tiny", "small", "base", "large"]

    @classmethod
    def get_name(cls) -> str:
        # Return the name of the model family
        return "sam2_realtime"

    def __init__(self):
        self.default_compile_model = False
        self.device = "cuda:0"
        self.color_channel_mode = "BGR"
        self.model = None
        self.flavor = None
        self.is_initialized = False

    def get_status(self):
        if self.flavor is None:
            return None
        else:
            return {'flavor': self.flavor, 'tracker_initialized': self.is_initialized}

    def load(self, payload: Dict[str, Any]) -> None:
        """
        Load weights and configuration for the model.
        Expects payload to contain:
          - flavor: one of get_available_flavors()
          - compile_model: whether to compile the model
        """
        flavor = payload.get("flavor")
        if flavor not in self.get_available_flavors():
            raise ValueError(f"Unknown flavor '{flavor}', valid flavors: {self.get_available_flavors()}")
        compile_model = payload.get("compile_model", self.default_compile_model)

        if flavor == 'tiny':
            config_file = 'sam2.1_hiera_t.yaml'
            weights_file = 'sam2.1_hiera_tiny.pt'
        elif flavor == 'small':
            config_file = 'sam2.1_hiera_s.yaml'
            weights_file = 'sam2.1_hiera_small.pt'
        elif flavor == 'base':
            config_file = 'sam2.1_hiera_b+.yaml'
            weights_file = 'sam2.1_hiera_base_plus.pt'
        elif flavor == 'large':
            config_file = 'sam2.1_hiera_l.yaml'
            weights_file = 'sam2.1_hiera_large.pt'
        else:
            raise NotImplementedError
        
        weight_path = os.path.join('/cache/sam2_realtime/', weights_file)
        config_path = os.path.join('configs/sam2.1', config_file)
        self.model = build_sam2_camera_predictor(config_path, weight_path, device=self.device, vos_optimized=compile_model)
        return True

    def unload(self) -> None:
        """
        Free GPU memory and remove references
        """
        if self.model is not None:
            del self.model
            self.model = None
            self.flavor = None
            self.is_initialized = False
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
        return True
    
    def reset(self) -> None:
        if self.model is not None and self.is_initialized:
            self.model.reset_state()
            self.is_initialized = False
        return True  

    def update(self, inference_inputs) -> None:
        """
        Initialize the model with new tracks or update existing tracks.
        Inference inputs contains:
          - cv2_img: H x W x 3 image
          - prompts: list of bbox_prompts and point_prompts
            -> bbox_prompt: dict with keys 'object_id' and 'bbox'
            -> point_prompt: dict with keys 'object_id', 'points' and 'labels'
          - reset_tracks: whether to reset the model state
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        cv2_img, prompts, reset_tracks = inference_inputs['cv2_img'], inference_inputs['prompts'], inference_inputs['reset_tracks']
        # TODO: currently force reset_tracks to True, see below
        reset_tracks = True
        if self.is_initialized and reset_tracks:
            self.reset()
            self.is_initialized = False
        # start tracks from scratch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            if not self.is_initialized:
                self.model.load_first_frame(cv2_img)
                self.is_initialized = True
                for prompt in prompts:
                    if 'bbox' in prompt:
                        _, out_obj_ids, out_mask_logits = self.model.add_new_prompt(frame_idx=0, obj_id=prompt['object_id'], bbox=prompt['bbox'])
                    elif 'points' in prompt:
                        _, out_obj_ids, out_mask_logits = self.model.add_new_prompt(frame_idx=0, obj_id=prompt['object_id'], points=prompt['points'], labels=prompt['labels'])
                    else:
                        raise ValueError("Invalid prompt format")
            # update existing tracks
            else:
                self.model.add_conditioning_frame(cv2_img)
                for prompt in prompts:
                    if 'bbox' in prompt:
                        # TODO: currently if_new_target=True will raise a NotImplementedError - seems to be WIP
                        _, out_obj_ids, out_mask_logits = self.model.add_new_prompt_during_track(obj_id=prompt['object_id'], bbox=prompt['bbox'], clear_old_points=False, if_new_target=True)
                    elif 'points' in prompt:
                        _, out_obj_ids, out_mask_logits = self.model.add_new_prompt_during_track(obj_id=prompt['object_id'], points=prompt['points'], labels=prompt['labels'], clear_old_points=False, if_new_target=True)
                    else:
                        raise ValueError("Invalid prompt format")
        track_ids = torch.atleast_1d(torch.tensor(out_obj_ids).int())
        masks = (out_mask_logits > 0.0) # num_objs x 1 x H x W
        masks[torch.isnan(masks)] = 0
        boxes = mask_to_box(masks).int()
        masks, boxes = masks.squeeze(1), boxes.squeeze(1)
        result = {'track_ids': track_ids, 'masks': masks, 'boxes': boxes}
        return [result]
    
    def decode_image(self, b64: str) -> np.ndarray:
        """
        Decode a base64-encoded image string into a numpy array.
        """
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        if self.color_channel_mode == "RGB":
            cv2_img = np.array(img)
        else:
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return cv2_img

    def preprocess(self, payload: Dict[str, Any], mode='infer') -> Any:
        """
        Convert incoming JSON (e.g. base64 image and parameters) for the model.
        Mode specifies which endpoint provides the payload ('infer', 'init', 'update').
        Expects payload to contain (for 'infer' mode):
          - images: base64-encoded image string or list of image strings
        """
        if mode == 'infer':
            images = payload.get("images")
            if images is None:
                raise ValueError("Missing 'images' in payload")
            # if necessary, convert to list of images
            if not isinstance(images, list):
                if isinstance(images, str):
                    images = [images]
                else:
                    raise ValueError("Invalid 'images' in payload")
            decoded_images = [self.decode_image(b64) for b64 in images]
            return {'cv2_images': decoded_images}
        elif mode == 'update':
            image = payload.get("image")
            if image is None:
                raise ValueError("Missing 'image' in payload")
            prompts = payload.get("prompts")
            if prompts is None:
                raise ValueError("Missing 'prompts' in payload")
            if not isinstance(prompts, list):
                if isinstance(prompts, dict):
                    prompts = [prompts]
                else:
                    raise ValueError("Invalid 'prompts' in payload")
            processed_prompts = []
            for prompt in prompts:
                processed_prompt = prompt.copy()
                if not 'object_id' in prompt:
                    raise ValueError("Missing 'object_id' in prompt")
                if not 'bbox' in prompt and not ('points' in prompt and 'labels' in prompt):
                    raise ValueError("Missing 'bbox' or 'points' and 'labels' in prompt")
                if 'bbox' in prompt:
                    bbox = prompt['bbox'] # [x1, y1, x2, y2]
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise ValueError("Invalid 'bbox' in prompt")
                    processed_prompt['bbox'] = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                if 'points' in prompt:
                    points = prompt['points']
                    if not isinstance(points, list) or len(points) == 0:
                        raise ValueError("Invalid 'points' in prompt")
                    points = np.array(points, dtype=np.float32)
                    if points.shape[1] != 2:
                        raise ValueError("Invalid 'points' in prompt")
                    processed_prompt['points'] = points
                if 'labels' in prompt:
                    labels = prompt['labels']
                    if not isinstance(labels, list) or len(labels) == 0:
                        raise ValueError("Invalid 'labels' in prompt")
                    labels = np.array(labels, dtype=np.int32)
                    if labels.shape[0] != points.shape[0]:
                        raise ValueError("Invalid 'labels' in prompt")
                    processed_prompt['labels'] = labels
                processed_prompts.append(processed_prompt)
            reset_tracks = payload.get("reset_tracks", False)
            cv2_img = self.decode_image(image)
            return {'cv2_img': cv2_img, 'prompts': processed_prompts, 'reset_tracks': reset_tracks}
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        
    def infer(self, inference_inputs) -> Any:
        """
        Run the forward pass.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        results = []
        if not self.is_initialized:
            return results
        for cv2_img in inference_inputs['cv2_images']:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                out_obj_ids, out_mask_logits = self.model.track(cv2_img)
            track_ids = torch.atleast_1d(torch.tensor(out_obj_ids).int())
            masks = (out_mask_logits > 0.0) # num_objs x 1 x H x W
            masks[torch.isnan(masks)] = 0
            boxes = mask_to_box(masks).int()
            masks, boxes = masks.squeeze(1), boxes.squeeze(1)
            result = {'track_ids': track_ids, 'masks': masks, 'boxes': boxes}
            results.append(result)
        return results

    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert raw outputs into JSON-serializable results.
        """

        results, metadata = [], []
        for output_for_img in outputs:
            result_for_img, metadata_for_img = [], None
            for track_id, mask, box in zip(output_for_img["track_ids"], output_for_img["masks"], output_for_img["boxes"]):
                if metadata_for_img is None:
                    metadata_for_img = {
                        "image_width": mask.shape[1],
                        "image_height": mask.shape[0],
                    }
                x1, y1, x2, y2 = map(int, box)
                if x1 < 0 or x1 > mask.shape[1] or x2 < 0 or x2 > mask.shape[1] or y1 < 0 or y1 > mask.shape[0] or y2 < 0 or y2 > mask.shape[0]:
                    logger.warning(f"Track discarded with invalid bounding box coordinates: {box}.")
                    continue
                cropped_mask = mask[y1:y2, x1:x2]
                cropped_mask = cropped_mask.cpu().numpy().astype(np.bool_)
                mask_base64 = encode_mask(cropped_mask)
                result_for_img.append({
                    "box_xyxy": box.cpu().tolist(),
                    "mask": mask_base64,
                    "track_id": int(track_id),
                })
            results.append(result_for_img)
            metadata.append(metadata_for_img)

        return {"artifact": {"tracks": results, "model": "sam2_realtime", "metadata": metadata}}
