import gc
import io
import base64
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
import torch

from transformers import AutoProcessor, AutoModelForCausalLM

from nimbro_vision_server.model_base import BaseModel
from nimbro_vision_server.utils import setup_logging, encode_mask

from models.florence2.florence2.utils import PATTERNS, get_prompt_type, bbox_to_loc, clean_string, polygons_to_bbox_and_mask, quad_to_xyxy

logger = setup_logging()

class Model(BaseModel):
    """
    Wrapper for the Florence-2 Model.
    Implements the BaseModel interface so it can be loaded, invoked,
    and unloaded by the server framework.
    """

    @classmethod
    def get_available_flavors(cls) -> List[str]:
        # Define the supported model sizes or variants
        return ["base", "large", "base_ft", "large_ft"]

    @classmethod
    def get_name(cls) -> str:
        # Return the name of the model family
        return "florence2"

    def __init__(self):
        self.default_task_prompt = "<DENSE_REGION_CAPTION>"
        self.default_prompt_args = None
        self.device = "cuda:0"
        self.color_channel_mode = "RGB"
        self.default_max_new_tokens = 1024
        self.default_num_beams = 3
        self.default_max_batch_size = 8
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

        if flavor == 'base':
            model_dir = '/cache/florence2/Florence-2-base'
        elif flavor == 'large':
            model_dir = '/cache/florence2/Florence-2-large'
        elif flavor == 'base_ft':
            model_dir = '/cache/florence2/Florence-2-base-ft'
        elif flavor == 'large_ft':
            model_dir = '/cache/florence2/Florence-2-large-ft'
        else:
            raise NotImplementedError
        
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True, torch_dtype='auto').eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
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
    
    def build_prompt(self, task_prompt: str, image: Image, prompt_args: Dict[str, Any] = None) -> str:
        """
        Build the prompt for the model.
        Expects task_prompt to be a string and the optional prompt_args to be a dictionary.
        """
        if not isinstance(task_prompt, str):
            raise ValueError("Invalid task_prompt in payload")
        # determine the type of the task
        prompt_type = task_prompt if task_prompt in PATTERNS else get_prompt_type(task_prompt)
        if prompt_type is None:
            raise ValueError(f"Invalid prompt format: {task_prompt}. Accepted formats: {list(PATTERNS.keys())}")
        if prompt_args is None:
            prompt_args = {}
        # verify additional arguments and construct the prompt
        if prompt_type == "<CAPTION_TO_PHRASE_GROUNDING>":
            caption = prompt_args.get("caption")
            if caption is None:
                raise ValueError("Missing 'caption' in prompt_args for <CAPTION_TO_PHRASE_GROUNDING>")
            if not isinstance(caption, str) or len(caption)==0:
                raise ValueError("Invalid 'caption' in prompt_args for <CAPTION_TO_PHRASE_GROUNDING>")
            prompt = f"{task_prompt}{caption}"
        elif prompt_type in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<OPEN_VOCABULARY_DETECTION>"]:
            phrase = prompt_args.get("phrase")
            if phrase is None:
                raise ValueError(f"Missing 'phrase' in prompt_args for {prompt_type}")
            if not isinstance(phrase, str) or len(phrase)==0:
                raise ValueError(f"Invalid 'phrase' in prompt_args for {prompt_type}")
            prompt = f"{task_prompt}{phrase}"
        elif prompt_type in ["<REGION_TO_SEGMENTATION>", "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"]:
            bbox_xyxy = prompt_args.get("bbox_xyxy")
            if bbox_xyxy is None:
                raise ValueError(f"Missing 'bbox_xyxy' in prompt_args for {prompt_type}")
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
                raise ValueError(f"Invalid 'bbox_xyxy' in prompt_args for {prompt_type}")
            # convert to locs
            locs = bbox_to_loc(bbox_xyxy, image_h=image.height, image_w=image.width)
            prompt = f"{task_prompt}{locs}"
        else:
            prompt = task_prompt
        # final check that the prompt is valid
        prompt_type = get_prompt_type(prompt)
        if prompt_type is None:
            raise ValueError(f"Invalid prompt format: {prompt}. Accepted formats: {list(PATTERNS.keys())}")
        return prompt
    
    def preprocess(self, payload: Dict[str, Any]) -> Any:
        """
        Convert incoming JSON for the model.
        Expects payload to contain:
          - images: base64-encoded image string or list of image strings
          - [optional] prompts: list of prompts (dicts {'task_prompt': <TASK_PROMPT>, 'prompt_args': {additional arguments necessary for the task}}) for each image or a single prompt for all images
          - [optional] inference_parameters: dict with inference parameters (max_new_tokens, num_beams, max_batch_size)
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
            prompts = [{'task_prompt': self.default_task_prompt, 'prompt_args': self.default_prompt_args}]
        if not isinstance(prompts, list):
            if isinstance(prompts, dict):
                prompts = [prompts]
            elif isinstance(prompts, str):
                prompts = [{'task_prompt': prompts, 'prompt_args': None}]
            else:
                raise ValueError("Invalid 'prompts' in payload")
        prompts = prompts * len(images) if len(prompts) == 1 and len(images) > 1 else prompts
        standardized_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                if isinstance(prompt, str):
                    standardized_prompts.append({'task_prompt': prompt, 'prompt_args': None})
                else:
                    raise ValueError("Invalid 'prompts' in payload")
            else:
                standardized_prompts.append(prompt)
        if len(images) != len(standardized_prompts):
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
        if not 'num_beams' in inference_parameters:
            standardized_params['num_beams'] = self.default_num_beams
        if not isinstance(standardized_params['num_beams'], int):
            raise ValueError(f"Invalid 'num_beams' in inference_parameters")
        if not 'max_batch_size' in inference_parameters:
            standardized_params['max_batch_size'] = self.default_max_batch_size
        if not isinstance(standardized_params['max_batch_size'], int):
            raise ValueError(f"Invalid 'max_batch_size' in inference_parameters")
        inference_parameters = standardized_params
        # decode images
        decoded_images = [self.decode_image(image) for image in images]
        # build prompts
        converted_prompts = []
        for image, prompt in zip(decoded_images, standardized_prompts):
            task_prompt = prompt.get("task_prompt")
            if task_prompt is None:
                raise ValueError("Missing 'task_prompt' in prompt")
            prompt_args = prompt.get("prompt_args")
            converted_prompts.append(self.build_prompt(task_prompt, image, prompt_args))
        return {'pil_images': decoded_images, 'prompts': converted_prompts, 'inference_parameters': inference_parameters}

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
            inputs = self.processor(text=prompt_minibatch, images=image_minibatch, return_tensors="pt", padding=True).to(self.device, torch.float16)
            generated_ids = self.model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=inference_parameters['max_new_tokens'],
                                early_stopping=False,
                                do_sample=False,
                                num_beams=inference_parameters['num_beams'],
                            )
            
            for generated_text, prompt, image in zip(self.processor.batch_decode(generated_ids, skip_special_tokens=False), prompt_minibatch, image_minibatch):
                # parsed answer is a dict {'PROMPT_TYPE': PROMPT_TYPE_SPECIFIC_OUTPUT}
                parsed_answer = self.processor.post_process_generation(
                                    generated_text,
                                    task=get_prompt_type(prompt), 
                                    image_size=(image.width, image.height)
                                )
                results.append((prompt, parsed_answer))
        return results
    
    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert raw outputs into JSON-serializable results.
        """
        detections, captions = [], []
        for prompt, parsed_answer in outputs:
            task_type = list(parsed_answer.keys())[0]
            task_output = parsed_answer[task_type]
            if task_type in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>", "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"]:
                # caption only
                captions.append(clean_string(task_output))
                detections.append([])
            elif task_type in ["<CAPTION_TO_PHRASE_GROUNDING>", "<OD>", "<DENSE_REGION_CAPTION>"]:
                bboxes_xyxy = task_output['bboxes']
                labels = task_output['labels']
                detections_for_img = []
                for bbox, label in zip(bboxes_xyxy, labels):
                    detections_for_img.append({
                        "box_xyxy": [int(i) for i in bbox],
                        "label": label
                    })
                detections.append(detections_for_img)
                captions.append(None)
            elif task_type in ["<OPEN_VOCABULARY_DETECTION>"]:
                bboxes_xyxy = task_output['bboxes']
                labels = task_output['bboxes_labels']
                polygons = task_output['polygons']
                polygons_labels = task_output['polygons_labels']
                polygons_bboxes = []
                for polygon in polygons:
                    bbox_xyxy, _ = polygons_to_bbox_and_mask(polygon)
                    polygons_bboxes.append(bbox_xyxy)
                bboxes_xyxy = bboxes_xyxy + polygons_bboxes
                labels = labels + polygons_labels
                detections_for_img = []
                for bbox, label in zip(bboxes_xyxy, labels):
                    detections_for_img.append({
                        "box_xyxy": [int(i) for i in bbox],
                        "label": label
                    })
                detections.append(detections_for_img)
                captions.append(None)
            elif task_type in ["<REGION_PROPOSAL>"]:
                bboxes_xyxy = task_output['bboxes']
                detections_for_img = []
                for bbox in bboxes_xyxy:
                    detections_for_img.append({
                        "box_xyxy": [int(i) for i in bbox],
                    })
                detections.append(detections_for_img)
                captions.append(None)
            elif task_type in ["<REGION_TO_SEGMENTATION>"]:
                polygons = task_output['polygons']
                bboxes_xyxy, masks = [], []
                for polygon in polygons:
                    bbox_xyxy, mask = polygons_to_bbox_and_mask(polygon)
                    bboxes_xyxy.append(bbox_xyxy)
                    masks.append(encode_mask(mask))
                detections_for_img = []
                for bbox, mask in zip(bboxes_xyxy, masks):
                    detections_for_img.append({
                        "box_xyxy": bbox,
                        "mask": mask,
                    })
                detections.append(detections_for_img)
                captions.append(None)
            elif task_type in ["<REFERRING_EXPRESSION_SEGMENTATION>"]:
                label = prompt[len(task_type):]
                polygons = task_output['polygons']
                bboxes_xyxy, masks = [], []
                for polygon in polygons:
                    bbox_xyxy, mask = polygons_to_bbox_and_mask(polygon)
                    logger.info(f"bbox_xyxy: {bbox_xyxy}, mask: {mask.shape}")
                    bboxes_xyxy.append(bbox_xyxy)
                    mask_base64 = encode_mask(mask)
                    masks.append(mask_base64)
                detections_for_img = []
                for bbox, mask in zip(bboxes_xyxy, masks):
                    detections_for_img.append({
                        "box_xyxy": bbox,
                        "mask": mask,
                        "label": label
                    })
                detections.append(detections_for_img)
                captions.append(None)
            elif task_type in ["<OCR_WITH_REGION>"]:
                quad_boxes = task_output['quad_boxes']
                boxes_xyxy = [quad_to_xyxy(quad) for quad in quad_boxes]
                labels = task_output['labels']
                labels = [clean_string(label) for label in labels]
                detections_for_img = []
                for box, label in zip(boxes_xyxy, labels):
                    detections_for_img.append({
                        "box_xyxy": box,
                        "label": label
                    })
                detections.append(detections_for_img)
                captions.append(None)
            else: 
                raise ValueError(f"Unknown task type: {task_type}")
                                
        return {"artifact": {"detections": detections, "captions": captions, "model": "florence2"}}
        
