import gc
import io
import base64
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
import torch

from transformers import AutoProcessor, AutoModelForVision2Seq

from nimbro_vision_server.model_base import BaseModel
from nimbro_vision_server.utils import setup_logging

from models.kosmos2.kosmos2.utils import PATTERNS, get_prompt_type

logger = setup_logging()

class Model(BaseModel):
    """
    Wrapper for the KOSMOS-2 Model.
    Implements the BaseModel interface so it can be loaded, invoked,
    and unloaded by the server framework.
    """

    @classmethod
    def get_available_flavors(cls) -> List[str]:
        # Define the supported model sizes or variants
        return ["patch14-224"]

    @classmethod
    def get_name(cls) -> str:
        # Return the name of the model family
        return "kosmos2"

    def __init__(self):
        self.default_prompt = '<grounding> An image of'
        self.device = "cuda:0"
        self.color_channel_mode = "RGB"
        self.default_max_new_tokens = 64
        self.default_max_batch_size = 16
        self.model = None
        self.flavor = None

    def get_status(self):
        if self.flavor is None:
            return None
        else:
            return {'flavor': self.flavor}

    def load(self, payload: Dict[str, Any]) -> None:
        """
        Load weights and configuration for the model.
        Expects payload to contain:
          - flavor: one of get_available_flavors()
        """
        flavor = payload.get("flavor")
        if flavor not in self.get_available_flavors():
            raise ValueError(f"Unknown flavor '{flavor}', valid flavors: {self.get_available_flavors()}")

        if flavor == 'patch14-224':
            model_dir = '/cache/kosmos2/patch14-224'
        else:
            raise NotImplementedError
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
        self.flavor = flavor
        return True

    def unload(self) -> None:
        """
        Free GPU memory and remove references
        """
        if self.model is not None:
            del self.model, self.processor
            self.model = None
            self.processor = None
            self.flavor = None
            gc.collect()
            torch.cuda.empty_cache()
        return True
    
    def decode_image(self, b64: str) -> Image:
        """
        Decode a base64-encoded image string into a PIL image.
        """
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        if self.color_channel_mode == "RGB":
            cv2_img = np.array(img)
        else:
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return Image.fromarray(cv2_img, 'RGB' if self.color_channel_mode == "RGB" else 'BGR')
    

    def preprocess(self, payload: Dict[str, Any]) -> Any:
        """
        Convert incoming JSON for the model.
        Expects payload to contain:
          - images: base64-encoded image string or list of image strings
          - [optional] prompts: list of prompts (strings) for each image or a single prompt for all images
          - [optional] inference_parameters: dict with inference parameters (max_new_tokens, max_batch_size)
        """
        # unpack the payload
        images = payload.get("images")
        if images is None:
            raise ValueError("Missing 'images' in payload")
        # if necessary, convert to list of images
        if not isinstance(images, list):
            if isinstance(images, str):
                images = [images]
            else:
                raise ValueError("Invalid 'images' in payload")
        prompts = payload.get("prompts")
        if prompts is None:
            prompts = [self.default_prompt]
        if not isinstance(prompts, list):
            if isinstance(prompts, str):
                prompts = [prompts]
            else:
                raise ValueError("Invalid 'prompts' in payload")
        prompts = prompts * len(images) if len(prompts) == 1 and len(images) > 1 else prompts
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise ValueError("Invalid 'prompts' in payload")
            prompt_type = get_prompt_type(prompt)
            if prompt_type is None:
                raise ValueError(f"Invalid prompt format: {prompt}. Accepted formats: {list(PATTERNS.keys())}")
        if len(images) != len(prompts):
            raise ValueError("Mismatch between number of images and prompts")
        inference_parameters = payload.get("inference_parameters")
        if inference_parameters is None:
            inference_parameters = {}
        if not isinstance(inference_parameters, dict):
            raise ValueError("Invalid 'inference_parameters' in payload")
        standardized_params = inference_parameters.copy()
        if not 'max_new_tokens' in inference_parameters:
            standardized_params['max_new_tokens'] = self.default_max_new_tokens
        if not isinstance(standardized_params['max_new_tokens'], int):
            raise ValueError(f"Invalid 'max_new_tokens' in inference_parameters")
        if not 'max_batch_size' in inference_parameters:
            standardized_params['max_batch_size'] = self.default_max_batch_size
        if not isinstance(standardized_params['max_batch_size'], int):
            raise ValueError(f"Invalid 'max_batch_size' in inference_parameters")
        inference_parameters = standardized_params
        # decode images
        decoded_images = [self.decode_image(image) for image in images]
        return {'pil_images': decoded_images, 'prompts': prompts, 'inference_parameters': inference_parameters}

    def infer(self, inference_inputs) -> Any:
        """
        Run the forward pass.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        results = []
        images, prompts, inference_parameters = inference_inputs['pil_images'], inference_inputs['prompts'], inference_inputs['inference_parameters']
        for start in range(0, len(prompts), inference_parameters['max_batch_size']):
            end = min(start + inference_parameters['max_batch_size'], len(prompts))
            image_minibatch, prompt_minibatch = images[start:end], prompts[start:end]
            inputs = self.processor(text=prompt_minibatch, images=image_minibatch, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                                pixel_values=inputs["pixel_values"],
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                image_embeds=None,
                                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                                use_cache=True,
                                max_new_tokens=inference_parameters['max_new_tokens'],
                            )
            for generated_text, image in zip(self.processor.batch_decode(generated_ids, skip_special_tokens=True), image_minibatch):
                caption, entities = self.processor.post_process_generation(generated_text)
                h, w =  image.height, image.width
                labels, boxes_xyxy = [], []
                for label, _, bboxes in entities:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        bbox_xyxy = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
                        boxes_xyxy.append(bbox_xyxy)
                        labels.append(label)
                result = {'boxes_xyxy': boxes_xyxy,
                          'labels': labels,
                          'caption': caption}
                results.append(result)
        return results
    
    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert raw outputs into JSON-serializable results.
        """
        detections, captions = [], []
        for output_for_img in outputs:
            detections_for_img = []
            for box, label in zip(output_for_img["boxes_xyxy"], output_for_img["labels"]):
                detections_for_img.append({
                    "box_xyxy": box,
                    "label": label
                })
            captions.append(output_for_img["caption"])
            detections.append(detections_for_img)
        return {"artifact": {"detections": detections, "captions": captions, "model": "kosmos2"}}
        
