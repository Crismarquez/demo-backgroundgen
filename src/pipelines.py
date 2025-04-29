from uuid import uuid4
from pathlib import Path
from typing import Optional

from src.blocks import (
    NormalizationBlock,
    FeatureExtractionBlock,
    BackgroundGenerationBlock,
    PostProductionBlock
)
from src.utils import load_image_as_base64, url_to_base64, save_base64_to_file
from config.config import DATA_DIR

class PipelineOrchestrator:
    def __init__(self, mode: str = "inference"):
        self.mode = mode
        self.run_id = uuid4().hex
        root = DATA_DIR / ("runs_batch" if mode == "batch" else "runs_inference")
        self.base_dir = root / self.run_id
        (self.base_dir / "images").mkdir(parents=True, exist_ok=True)
        self.blocks = {
            "normalization": NormalizationBlock(self.base_dir),
            "features": FeatureExtractionBlock(self.base_dir),
            "background": BackgroundGenerationBlock(self.base_dir),
            "post": PostProductionBlock(self.base_dir)
        }

    async def run(self, input_image_path: str, preferences: Optional[str] = None) -> str:
        # Obtener base64 de la imagen de entrada
        if input_image_path.startswith(('http://','https://')):
            b64 = url_to_base64(input_image_path)
        elif input_image_path.startswith('data:image'):
            b64 = input_image_path
        else:
            img_path = Path(input_image_path)
            b64 = load_image_as_base64(img_path)
        save_base64_to_file(
            b64,
            self.base_dir / "images" / f"input.{b64.split(';')[0].split('/')[1]}"
        )
        norm = await self.blocks["normalization"].run(b64)
        if norm["white_bg"]:
            feat = await self.blocks["features"].run(norm, b64)
            bg = await self.blocks["background"].run(norm, feat, b64)
            post = await self.blocks["post"].run(bg["images_bg"])
        else:
            post = await self.blocks["post"].run(b64)
        return self.run_id

    def run_step(self, block_name: str, method_name: str):
        # Implementación sin cambios
        raise NotImplementedError("run_step no soporta este bloque/método.")
