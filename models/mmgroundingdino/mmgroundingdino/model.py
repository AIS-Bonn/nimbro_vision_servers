import os
import io
import base64
from math import ceil
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms

from mmdet.apis import DetInferencer

from nimbro_vision_server.model_base import BaseModel
from nimbro_vision_server.utils import setup_logging

logger = setup_logging()

class Model(BaseModel):
    """
    Wrapper for the MMGroundingDINO model.
    Implements the BaseModel interface so it can be loaded, invoked,
    and unloaded by the server framework.
    """

    @classmethod
    def get_available_flavors(cls) -> List[str]:
        # Define the supported model sizes or variants
        return ["tiny", "base", "large", "large_zeroshot", "llmdet_tiny", "llmdet_base", "llmdet_large"]
    
    @classmethod
    def get_name(cls) -> str:
        # Return the name of the model family
        return "mmgroundingdino"

    def __init__(self):
        self.default_min_confidence = 0.05
        self.default_nms_iou = None
        self.default_overdetect_factor = None
        self.device = "cuda:0"
        self.max_count_per_prompt = 1000
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

        if flavor == 'tiny':
            config_file = 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
            weights_file = 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
        elif flavor == 'llmdet_tiny':
            config_file = 'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py'
            weights_file = 'tiny.pth'
        elif flavor == 'base':
            config_file = 'grounding_dino_swin-b_pretrain_all.py'
            weights_file = 'grounding_dino_swin-b_pretrain_all-f9818a7c.pth'
        elif flavor == 'llmdet_base':
            config_file = 'grounding_dino_swin-b_pretrain_all.py'
            weights_file = 'base.pth'
        elif flavor == 'large':
            config_file = 'grounding_dino_swin-l_pretrain_all.py'
            weights_file = 'grounding_dino_swin-l_pretrain_all-56d69e78.pth'
        elif flavor == 'llmdet_large':
            config_file = 'grounding_dino_swin-l_pretrain_all.py'
            weights_file = 'large.pth'
        elif flavor == 'large_zeroshot':
            config_file = 'grounding_dino_swin-l_pretrain_obj365_goldg.py'
            weights_file = 'grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth'
        else:
            raise NotImplementedError
        
        cfg_path = os.path.join('/repos/mmdetection/configs/mm_grounding_dino/', config_file)
        weight_path = os.path.join('/cache/mmgroundingdino/', weights_file)
        self.model = DetInferencer(model=cfg_path, weights=weight_path, device=self.device, show_progress=False)
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
            torch.cuda.empty_cache()
        return True
    
    def process_prompts(self, prompts: List[str]
                    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Given a list of prompts (possibly with duplicates), returns:
        - unique_prompts: prompts in original order, deduplicated
        - duplicate_counts: mapping from prompt → count (only for count > 1)
        """
        counts = Counter(prompts)
        prompt_counts = {p: c for p, c in counts.items()}

        seen = set()
        unique_prompts = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)

        return unique_prompts, prompt_counts

    def preprocess(self, payload: Dict[str, Any]) -> Any:
        """
        Convert incoming JSON (e.g. base64 image and parameters) for the model.
        Expects payload to contain:
          - images: base64-encoded image string or list of image strings
          - inference_parameters: dict with inference parameters (prompts, min_confidence, nms_iou, overdetect_factor) or list of such dicts per image
        """
        # unpack the payload
        inference_parameters = payload.get("inference_parameters")
        if inference_parameters is None:
            raise ValueError("Missing 'inference_parameters' in payload")
        images = payload.get("images")
        if images is None:
            raise ValueError("Missing 'images' in payload")
        # if necessary, convert to list of images
        if not isinstance(images, list):
            if isinstance(images, str):
                images = [images]
            else:
                raise ValueError("Invalid 'images' in payload")
        # if necessary, convert to list of inference_parameters
        if not isinstance(inference_parameters, list):
            if isinstance(inference_parameters, dict):
                inference_parameters = [inference_parameters]
            else:
                raise ValueError("Invalid 'inference_parameters' in payload")
        if len(images) == len(inference_parameters):
            pass
        elif len(images) == 1 and len(inference_parameters) > 1:
            # deal with this case later when the image is decoded
            pass
        elif len(images) > 1 and len(inference_parameters) == 1:
            # duplicate the inference_parameters for each image
            inference_parameters = inference_parameters * len(images)
        else:
            raise ValueError("Mismatch between number of images and inference_parameters")
        # standardize the inference_parameters
        for i, params in enumerate(inference_parameters):
            standardized_params = params.copy()
            if not 'prompts' in params:
                raise ValueError(f"Missing 'prompts' in inference_parameters[{i}]")
            if not isinstance(params['prompts'], list):
                if isinstance(params['prompts'], str):
                    standardized_params['prompts'] = [params['prompts']]
                else:
                    raise ValueError(f"Invalid 'prompts' in inference_parameters[{i}]")
            if not 'min_confidence' in params:
                standardized_params['min_confidence'] = self.default_min_confidence
            if not isinstance(standardized_params['min_confidence'], float):
                raise ValueError(f"Invalid 'min_confidence' in inference_parameters[{i}]")
            if not 'nms_iou' in params:
                standardized_params['nms_iou'] = self.default_nms_iou
            if not standardized_params['nms_iou'] is None and not isinstance(standardized_params['nms_iou'], float):
                raise ValueError(f"Invalid 'nms_iou' in inference_parameters[{i}]")
            if not 'overdetect_factor' in params:
                standardized_params['overdetect_factor'] = self.default_overdetect_factor
            if not standardized_params['overdetect_factor'] is None and not isinstance(standardized_params['overdetect_factor'], float):
                raise ValueError(f"Invalid 'overdetect_factor' in inference_parameters[{i}]")
            # deduplicate prompts and count duplicates
            unique_prompts, prompt_counts = self.process_prompts(standardized_params['prompts'])
            standardized_params['prompts'] = unique_prompts
            standardized_params['prompt_counts'] = prompt_counts
            inference_parameters[i] = standardized_params
        decoded_images = []
        for b64 in images:
            data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(data))
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            decoded_images.append(cv2_img)
        # go back to the case where we have one image and multiple inference_parameters
        if len(decoded_images) == 1 and len(inference_parameters) > 1:
            decoded_images = decoded_images * len(inference_parameters)
        return {'cv2_images_bgr': decoded_images, 'inference_parameters': inference_parameters}
    
    def count_prompt_chars(self, prompts) -> List[int]: 
        return [len(s) for s in prompts]
    
    def determine_bin_width(self, prompts):
        prompt_counts = self.count_prompt_chars(prompts)
        n = len(prompt_counts)
        prefix_sum = [0]*(n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + prompt_counts[i]
        # determine allowed chunk widths
        def feasible(w):
            for start in range(0, n, w):
                end = min(start + w, n)
                if prefix_sum[end] - prefix_sum[start] > self.max_count_per_prompt:
                    return False
            return True
        feasible_widths = []
        for w in range(1, n+1):
            if feasible(w):
                leftover = n % w
                feasible_widths.append((leftover, w))
        # Sort: smaller leftover is better; for ties, larger w is better
        feasible_widths.sort(key=lambda x: (x[0], -x[1]))   
        # Remove w=1 if there's another valid w
        if feasible_widths:
            # if 1 is the only feasible width, return it
            # else pick the best feasible width that isn't 1
            if len(feasible_widths) == 1 and feasible_widths[0][1] == 1:
                return 1
            for leftover, w in feasible_widths:
                if w != 1:
                    return w
            return 1  # if all feasible widths happen to be 1, return it
        else:
            # no feasible width found at all
            logger.warning("No feasible width found for chunking prompts")
            return 1
        
    def filter_detections(self, results, min_confidence: float, nms_iou: float | None, overdetect_factor: float | None, prompt_counts: dict[str, int]):
        """
        results: {'boxes_xyxy', 'conf', 'labels'}
        prompt_counts: mapping prompt→count (includes singletons)
        """

        # confidence filter
        boxes = results['boxes_xyxy']
        confs = results['conf']
        labels = results['labels']

        mask = confs > min_confidence
        boxes = boxes[mask]
        confs = confs[mask]
        labels = [lbl for i, lbl in enumerate(labels) if mask[i].item()]

        if not nms_iou is None:
            nms_idx = nms(boxes=boxes, scores=confs, iou_threshold=nms_iou)
            boxes = boxes[nms_idx]
            confs = confs[nms_idx]
            labels = [labels[idx] for idx in nms_idx]

        # return if no overdetect_factor is set
        if overdetect_factor is None:
            return {
                'boxes_xyxy': boxes,
                'conf':        confs,
                'labels':      labels,
            }

        # build flat targets from the prompt_counts
        targets = []
        for prompt, cnt in prompt_counts.items():
            targets.extend([prompt] * cnt)
        num_targets = len(targets)

        # sort detections remaining after confidence filter by descending confidence
        order = np.argsort(confs.cpu().numpy())[::-1]

        # first pass: pick exactly one detection per target prompt
        picked_idxs = []
        found = []
        for idx in order:
            lbl = labels[idx]
            if lbl in targets:
                targets.remove(lbl)
                picked_idxs.append(idx)
                found.append(lbl)
            if not targets:
                logger.info(f"Found all {num_targets} targets: {found}")
                break
        else:
            if picked_idxs:
                logger.info(f"Found {len(picked_idxs)}/{num_targets}: {found}")
            logger.warning(f"Missing {len(targets)}/{num_targets}: {set(targets)}")

        # pad out to budget given by overdetect_factor
        desired = ceil(num_targets * overdetect_factor)
        padded = []
        if overdetect_factor > 0.0 and desired > len(picked_idxs):
            for idx in order:
                if idx in picked_idxs:
                    continue
                picked_idxs.append(idx)
                padded.append(labels[idx])
                if len(picked_idxs) >= desired:
                    break

            if padded:
                if len(picked_idxs) < desired:
                    logger.warning(
                        f"Padded only {len(padded)} extras (total {len(picked_idxs)}) "
                        f"but wanted {desired}: {padded}"
                    )
                else:
                    logger.info(
                        f"Padded {len(padded)} extras to reach {desired}: {padded}"
                    )
        else:
            logger.debug(
                f"No padding needed ({len(picked_idxs)}/{desired} detections)"
            )

        # return the selected detections
        return {
            'boxes_xyxy': boxes[picked_idxs],
            'conf':       confs[picked_idxs],
            'labels':     [labels[i] for i in picked_idxs],
        }

    def infer(self, inference_inputs) -> Any:
        """
        Run the forward pass.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        results = []
        for cv2_img,  inference_parameters in zip(inference_inputs['cv2_images_bgr'], inference_inputs['inference_parameters']):
            prompts, min_confidence = inference_parameters.get("prompts"), inference_parameters.get("min_confidence")
            nms_iou, overdetect_factor, prompt_counts = inference_parameters.get("nms_iou"), inference_parameters.get("overdetect_factor"), inference_parameters.get("prompt_counts")
            chunk_size = self.determine_bin_width(prompts)
            labels, boxes, conf = [], [], []
            for start in range(0, len(prompts), chunk_size):
                end = min(start + chunk_size, len(prompts))
                local_idx = list(range(start, end))
                local_prompt = '. '.join(prompts[start:end]) + '.'
                local_result = self.model(cv2_img, texts=local_prompt, custom_entities=True, show=False, return_vis=False)['predictions'][0]
                labels.extend([local_idx[idx] for idx in local_result['labels']])
                boxes.append(torch.tensor(local_result['bboxes']).float().to(self.device))
                conf.append(torch.tensor(local_result['scores']).float().to(self.device))
            labels = [prompts[idx] for idx in labels]
            if len(labels) == 0:
                result = {'boxes_xyxy': torch.empty((0, 4), device=self.device),
                         'conf': torch.empty((0,), device=self.device),
                         'labels': []}
            else:
                boxes = torch.cat(boxes, dim=0)
                conf = torch.cat(conf, dim=0)
                result = {'boxes_xyxy': boxes,
                         'conf': conf,
                         'labels': labels}
            results.append(self.filter_detections(result, min_confidence, nms_iou, overdetect_factor, prompt_counts))
        return results

    def postprocess(self, outputs: Any) -> Dict[str, Any]:
        """
        Convert raw outputs into JSON-serializable results.
        Example: extract boxes, confs, and corresponding prompts.
        """
        results = []
        for output_for_img in outputs:
            result_for_img = []
            for box, conf, prompt in zip(output_for_img["boxes_xyxy"], output_for_img["conf"], output_for_img["labels"]):
                result_for_img.append({
                    "box_xyxy": box.cpu().int().tolist(),
                    "confidence": float(conf),
                    "prompt": prompt
                })
            results.append(result_for_img)
        return {"artifact": {"detections": results, "model": "mmgroundingdino"}}
