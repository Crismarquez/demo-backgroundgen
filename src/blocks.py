from typing import Optional, List
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import base64
import asyncio
import tempfile

import fal_client
from gradio_client import Client, handle_file

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.utils import url_to_base64, image_to_base64, base64_to_image, save_base64_to_file, plot_orientation_2d_datauri
from src.prompt import ProductAnalysis, product_analysis_prompt
from config.config import ENV_VARIABLES

def now_iso():
    return datetime.utcnow().isoformat()

LLM_OPENAI = ChatOpenAI(
    model="gpt-4o",
    api_key=ENV_VARIABLES["OPENAI_API_KEY"],
    temperature=0.3,
    seed=42,
)

class NormalizationBlock:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    @staticmethod
    def detect_white_bg(
        img: Image.Image,
        border_frac: float = 0.1,
        white_thresh: int = 250,
        min_white_ratio: float = 0.65
    ) -> bool:
        arr = np.array(img.convert("RGB"))
        h, w, _ = arr.shape
        b = max(1, int(min(h, w) * border_frac))
        top    = arr[:b,   :, :]
        bottom = arr[-b:,  :, :]
        left   = arr[b:-b, :b, :]
        right  = arr[b:-b, -b:, :]
        border_pixels = np.vstack([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ])
        white_mask = np.all(border_pixels > white_thresh, axis=1)
        return bool(white_mask.mean() >= min_white_ratio)

    def detect_size(self, img: Image.Image):
        return list(img.size)
    
    def aspect_ratio(self, img: Image.Image):
        return img.width / img.height

    async def run(self, input_image_data: str):
        img = base64_to_image(input_image_data)
        image_path = self.base_dir / "images" / f"normalized_{uuid4().hex}.jpg"
        img.save(image_path)
        white_bg = NormalizationBlock.detect_white_bg(img)
        size = self.detect_size(img)
        aspect_ratio = self.aspect_ratio(img)
        return {
            "white_bg": white_bg,
            "size": size,
            "aspect_ratio": aspect_ratio
        }

class FeatureExtractionBlock:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def detect_orientation(self, image: str):
        # Placeholder for computer vision logic
        #TODO: implement https://huggingface.co/papers/2412.18605
        # Decodificar y guardar la imagen como un archivo temporal
        header, encoded = image.split(",", 1)
        image_data = base64.b64decode(encoded)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        # Usar handle_file para cargar la imagen
        client = Client("Viglong/Orient-Anything", hf_token=ENV_VARIABLES["HF_TOKEN"])
        result = client.predict(
            img=handle_file(temp_file_path),
            do_rm_bkg=False,
            do_infer_aug=False,
            api_name="/predict"
        )

        # Convert the orientation values to float to ensure compatibility with np.deg2rad
        azimuth = float(result[1])
        polar = float(result[2])
        rotation = float(result[3])
        
        orientation_base64 = plot_orientation_2d_datauri(angles=[azimuth, polar, rotation, 1])

        return {"img_orientation": orientation_base64}

    async def analyze_product(self, image_url: str):
        message = HumanMessage(
            content=[
                {"type": "text", "text": product_analysis_prompt["system"]},
                {"type": "text", "text": product_analysis_prompt["user"]},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        )
        model_with_tools = (
            LLM_OPENAI
            .with_structured_output(schema=ProductAnalysis,
                                     method="function_calling",
                                     include_raw=False)
        )
        # Use the async invoke method (adjust to your LangChain version)
        response = await model_with_tools.ainvoke([message])
        return response.dict()

    async def run(self, normalization_result: dict, image_url: str):
        orientation = self.detect_orientation(image_url)
        analysis = await self.analyze_product(image_url)
        return {
            "orientation": orientation,
            "analysis": analysis
        }

class BackgroundGenerationBlock:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    async def _generate_mask_fal(
        self,
        image_url: str,
        prompt: Optional[str] = None
    ) -> str:
        if prompt is None:
            prompt = "Object in the middle of the image"
        handler = await fal_client.submit_async(
            "fal-ai/evf-sam",
            arguments={
                "prompt": prompt,
                "image_url": image_url,
                "revert_mask": True
            },
        )
        async for _ in handler.iter_events(with_logs=True):
            pass
        result = await handler.get()
        image_result = result["image"]
        image_b64 = url_to_base64(image_result["url"])
        save_base64_to_file(
            image_b64,
            self.base_dir / "images" / f"mask_fal_{uuid4().hex[:8]}.png"
        )
        return image_b64

    async def _generate_masks_fal_parallel(
        self,
        image_url: str,
        prompts: List[str]
    ) -> List[str]:
        tasks = [self._generate_mask_fal(image_url, p) for p in prompts]
        return await asyncio.gather(*tasks)

    def _generate_mask_grabcut(self, image_b64: str) -> (str, Path):
        img = base64_to_image(image_b64)
        arr = np.array(img.convert('RGB'))
        mask = np.zeros(arr.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        h, w = arr.shape[:2]
        rect = (10, 10, w-20, h-20)
        white_thr = 240
        white_px = np.all(arr > white_thr, axis=2)
        mask[white_px] = cv2.GC_BGD
        mask[~white_px] = cv2.GC_PR_FGD
        cv2.grabCut(arr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
        mask2 = 255 - mask2
        path = self.base_dir / "images" / f"mask_grabcut_{uuid4().hex[:8]}.png"
        cv2.imwrite(str(path), mask2)
        base64_mask = image_to_base64(path)
        return base64_mask, path

    def _combine_masks(self, masks: List[str]) -> Optional[str]:
        if not masks:
            return None
        base = np.array(base64_to_image(masks[0]).convert('L'))
        for m in masks[1:]:
            arr = np.array(base64_to_image(m).convert('L'))
            if arr.shape != base.shape:
                arr = cv2.resize(arr, (base.shape[1], base.shape[0]))
            base = cv2.bitwise_and(base, arr)
        img = Image.fromarray(base)
        out_path = self.base_dir / "images" / f"combined_mask_{uuid4().hex[:8]}.png"
        img.save(out_path)
        return image_to_base64(out_path)

    async def segmentation_masks(
        self,
        image_b64: str,
        prompts: Optional[List[str]] = None
    ) -> dict:
        mask_gc_b64, gc_path = self._generate_mask_grabcut(image_b64)
        masks_fal = None
        if prompts:
            masks_fal = await self._generate_masks_fal_parallel(image_b64, prompts[:5])
            combined_fal = self._combine_masks(masks_fal)
        else:
            single = "Object in the middle of the image" if isinstance(prompts, list) else None
            fal_b64 = await self._generate_mask_fal(image_b64, single)
            combined_fal = fal_b64
        final = None
        if combined_fal:
            final = self._combine_masks([mask_gc_b64, combined_fal])
        return {
            "mask_grabcut": mask_gc_b64,
            "mask_segmentation": combined_fal,
            "mask_combined": final
        }

    def improve_ratio(self, image_b64: str, mask_b64: Optional[str] = None) -> dict:
        """
        Adjust image to 1:1 aspect ratio by adding padding.
        If mask is provided, it will be padded in the same way.
        
        Returns a dict with the padded image and mask (if provided).
        """
        img = base64_to_image(image_b64)
        width, height = img.size
        
        # Calculate target size for 1:1 ratio
        target_size = max(width, height)
        
        # Create new image with white background
        new_img = Image.new('RGBA', (target_size, target_size), (255, 255, 255, 0))
        
        # Calculate position to paste original image (centered)
        left = (target_size - width) // 2
        top = (target_size - height) // 2
        
        # Paste original image onto new canvas
        new_img.paste(img, (left, top))
        
        # Save and convert to base64
        out_path = self.base_dir / "images" / f"padded_{uuid4().hex[:8]}.png"
        new_img.save(out_path)
        padded_img_b64 = image_to_base64(out_path)
        
        result = {"padded_image": padded_img_b64}
        
        # If mask is provided, pad it the same way
        if mask_b64:
            mask = base64_to_image(mask_b64)
            new_mask = Image.new('L', (target_size, target_size), 255)  # Black background for mask
            new_mask.paste(mask, (left, top))
            mask_path = self.base_dir / "images" / f"padded_mask_{uuid4().hex[:8]}.png"
            new_mask.save(mask_path)
            padded_mask_b64 = image_to_base64(mask_path)
            result["padded_mask"] = padded_mask_b64
            
        return result

    def create_background(
        self,
        image_url: str,
        mask_b64: str,
        prompt: str
    ) -> str:
        # Improve aspect ratio to 1:1 before inpainting
        padded_results = self.improve_ratio(image_url, mask_b64)
        
        result = fal_client.run(
            "fal-ai/flux-pro/v1/fill",
            arguments={
                "prompt": prompt,
                "image_url": padded_results["padded_image"],
                "mask_url": padded_results["padded_mask"],
                "n_images": 1
            }
        )
        img_info = result["images"][0]
        img_b64 = url_to_base64(img_info["url"])
        save_base64_to_file(
            img_b64,
            self.base_dir / "images" / f"inpainted_{uuid4().hex[:8]}.png"
        )
        return img_b64

    async def run(
        self,
        normalization_result: dict,
        features_result: dict,
        image_b64: str
    ) -> dict:
        seg = await self.segmentation_masks(
            image_b64,
            features_result["analysis"]["objects_white"]
        )
        bg = self.create_background(
            image_b64,
            seg["mask_combined"],
            features_result["analysis"]["background_suggestion"]
        )
        return {"mask": seg, "images_bg": bg}

class PostProductionBlock:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def refine_shadows(self, input_image_b64: str) -> str:
        return input_image_b64

    def scale_image(self, input_b64: str) -> str:
        temp = base64_to_image(input_b64)
        path = self.base_dir / "images" / f"upsc_tmp_{uuid4().hex[:8]}.png"
        temp.save(path)
        res = fal_client.run(
            "fal-ai/clarity-upscaler",
            arguments={"image_url": input_b64}
        )
        url = res.get("image", {}).get("url")
        if url:
            b = url_to_base64(url)
            save_base64_to_file(b, path)
            return b
        return input_b64

    async def run(self, input_image_b64: str) -> dict:
        shaded = self.refine_shadows(input_image_b64)
        scaled = self.scale_image(shaded)
        out_path = self.base_dir / "images" / f"final_{uuid4().hex[:8]}.png"
        save_base64_to_file(scaled, out_path)
        return {"output_image": scaled}
