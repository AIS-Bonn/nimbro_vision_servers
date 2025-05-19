import gc
import io
import base64
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
import torch

from transformers import AutoModel

from nimbro_vision_server.model_base import BaseModel
from nimbro_vision_server.utils import setup_logging, decode_mask

from models.dam.dam.utils import KeywordsStoppingCriteria, tokenizer_image_token

logger = setup_logging()

class Model(BaseModel):
    """
    Wrapper for the Describe Anything Model.
    Implements the BaseModel interface so it can be loaded, invoked,
    and unloaded by the server framework.
    """

    @classmethod
    def get_available_flavors(cls) -> List[str]:
        # Define the supported model sizes or variants
        return ["3B"]

    @classmethod
    def get_name(cls) -> str:
        # Return the name of the model family
        return "dam"

    def __init__(self):
        self.default_temperature = 0.2
        self.default_top_p = 0.5
        self.default_num_beams = 1
        self.default_max_new_tokens = 512
        self.default_max_batch_size = 16
        self.default_query = 'Describe the masked region in detail.'
        self.device = "cuda:0"
        self.color_channel_mode = "RGB"
        self.query_base = '<image>\n'
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

        if flavor == '3B':
            model_dir = '/cache/dam/3B'
        else:
            raise NotImplementedError
        
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype="torch.float16", local_files_only=True, use_safetensors=True).to(self.device)
        self.model = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
        self.flavor = flavor
        return True

    def unload(self) -> None:
        """
        Free GPU memory and remove references
        """
        if self.model is not None:
            del self.model
            self.model = None
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
    
    def convert_mask(self, b64: str, bbox: list, img_h: int, img_w: int) -> Image:
        """
        Convert a base64-encoded bbox region mask string into a PIL image.
        """
        mask_arr = decode_mask(b64)
        x1, y1, x2, y2 = bbox
        full_mask = np.zeros((img_h, img_w), dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_arr
        return Image.fromarray(full_mask.astype(np.uint8) * 255)


    def preprocess(self, payload: Dict[str, Any]) -> Any:
        """
        Convert incoming JSON for the model.
        Expects payload to contain:
          - images: base64-encoded image string or list of image strings
          - prompts: dict containing 'bbox' and 'mask' prompt (one image and one prompt) or list of such dicts (one image with multiple prompts) or list of lists of dicts (multiple images with multiple prompts)
          - [optional] queries: list of queries (strings) for each image or a single query for all images
          - [optional] inference_parameters: dict with inference parameters (temperature, top_p, num_beams, max_new_tokens) or list of such dicts per image
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
        queries = payload.get("queries")
        if queries is None:
            queries = [self.default_query]
        if not isinstance(queries, list):
            if isinstance(queries, str):
                queries = [queries]
            else:
                raise ValueError("Invalid 'queries' in payload")
        queries = queries * len(images) if len(queries) == 1 and len(images) > 1 else queries
        if len(images) != len(queries):
            raise ValueError("Mismatch between number of images and queries")
        prompts = payload.get("prompts")
        if prompts is None:
            raise ValueError("Missing 'prompts' in payload")
        # if necessary, convert to list of list of prompts
        if isinstance(prompts, dict):
            prompts = [[prompts]]
        elif isinstance(prompts, list):
            if isinstance(prompts[0], dict):
                prompts = [prompts]
            elif isinstance(prompts[0], list):
                pass
            else:
                raise ValueError("Invalid 'prompts' in payload")
        else:
            raise ValueError("Invalid 'prompts' in payload")
        if len(images) != len(prompts):
            raise ValueError("Mismatch between number of images and prompts")
        inference_parameters = payload.get("inference_parameters")
        if inference_parameters is None:
            inference_parameters = [{}]
        elif isinstance(inference_parameters, dict):
            inference_parameters = [inference_parameters]
        if len(inference_parameters) == 1 and len(images) > 1:
            inference_parameters = inference_parameters * len(images)
        elif len(inference_parameters) != len(images):
            raise ValueError("Mismatch between number of images and inference_parameters")
        # standardize the inference_parameters
        for i, params in enumerate(inference_parameters):
            standardized_params = params.copy()
            if not 'temperature' in params:
                standardized_params['temperature'] = self.default_temperature
            if not isinstance(standardized_params['temperature'], float):
                raise ValueError(f"Invalid 'temperature' in inference_parameters[{i}]")
            if not 'top_p' in params:
                standardized_params['top_p'] = self.default_top_p
            if not isinstance(standardized_params['top_p'], float):
                raise ValueError(f"Invalid 'top_p' in inference_parameters[{i}]")
            if not 'num_beams' in params:
                standardized_params['num_beams'] = self.default_num_beams
            if not isinstance(standardized_params['num_beams'], int):
                raise ValueError(f"Invalid 'num_beams' in inference_parameters[{i}]")
            if not 'max_new_tokens' in params:
                standardized_params['max_new_tokens'] = self.default_max_new_tokens
            if not isinstance(standardized_params['max_new_tokens'], int):
                raise ValueError(f"Invalid 'max_new_tokens' in inference_parameters[{i}]")
            if not 'max_batch_size' in params:
                standardized_params['max_batch_size'] = self.default_max_batch_size
            if not isinstance(standardized_params['max_batch_size'], int):
                raise ValueError(f"Invalid 'max_batch_size' in inference_parameters[{i}]")
            inference_parameters[i] = standardized_params
        # prepend base query
        for i, query in enumerate(queries):
            queries[i] = self.query_base + query
        # decode images and prompts
        decoded_images, decoded_prompts = [], []
        for i, (image, prompts) in enumerate(zip(images, prompts)):
            decoded_image = self.decode_image(image)
            decoded_images.append(decoded_image)
            img_w, img_h = decoded_image.size
            decoded_prompts_per_image = []
            for prompt in prompts:
                if 'bbox' in prompt and 'mask' in prompt:
                    bbox = [int(coord) for coord in prompt['bbox']]
                    mask = prompt['mask']
                    decoded_mask = self.convert_mask(mask, bbox, img_h, img_w)
                    decoded_prompts_per_image.append(decoded_mask)
                else:
                    raise ValueError(f"Invalid prompt format for image {i}")
            decoded_prompts.append(decoded_prompts_per_image)

        return {'pil_images': decoded_images, 'pil_masks': decoded_prompts, 'queries': queries, 'inference_parameters': inference_parameters}

    def infer(self, inference_inputs) -> Any:
        """
        Run the forward pass.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        results = []
        for image,  masks, query, inference_args in zip(inference_inputs['pil_images'], inference_inputs['pil_masks'], inference_inputs['queries'], inference_inputs['inference_parameters']):
            results_per_image = []
            prompt, conv = self.model.get_prompt(query)
            input_ids = tokenizer_image_token(prompt, self.model.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)
            crop_mode, crop_mode2 = self.model.prompt_mode.split("+")
            for start in range(0, len(masks), inference_args['max_batch_size']):
                end = min(start + inference_args['max_batch_size'], len(masks))
                masks_batch = masks[start:end]
                image_batch = [image] * len(masks_batch)
                image_tensors = [self.model.get_image_tensor(image_pil, mask_pil, crop_mode=crop_mode, crop_mode2=crop_mode2) for image_pil, mask_pil in zip(image_batch, masks_batch)]
                stopping_criteria = KeywordsStoppingCriteria([conv.sep2], self.model.tokenizer, input_ids)
                inp_ids_batch = torch.cat([input_ids] * len(masks_batch), dim=0)
                generation_kwargs = {
                    "input_ids":    inp_ids_batch,
                    "images":       image_tensors,
                    "do_sample":    inference_args["temperature"] > 0.0,
                    "use_cache":    True,
                    "stopping_criteria": [ stopping_criteria ], 
                    "streamer":     None,
                    "temperature":  inference_args["temperature"],
                    "top_p":        inference_args["top_p"],
                    "num_beams":    inference_args["num_beams"],
                    "max_new_tokens": inference_args["max_new_tokens"],
                }
                with torch.inference_mode():
                    output_ids = self.model.model.generate(**generation_kwargs)
                outputs = self.model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                results_per_image.extend(outputs)
            results.append(results_per_image)
        return results

    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert raw outputs into JSON-serializable results.
        """
        return {"artifact": {"descriptions": outputs, "model": "dam"}}
