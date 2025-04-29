# test_pipeline.py

import sys
import os
import asyncio
from typing import Optional

# Para poder importar src/ y config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines import PipelineOrchestrator
from config.config import DATA_DIR, ENV_VARIABLES

MAX_RUNS = 3

async def run_pipeline_async(image_path: str, preferences: Optional[str] = None) -> str:
    """
    Versión async de tu pipeline runner.
    Devuelve el run_id generado por PipelineOrchestrator.
    """
    pipeline = PipelineOrchestrator(mode="inference") # "inference" o "batch"
    run_id = await pipeline.run(image_path, preferences)
    return run_id

async def main():
    sample_dir = DATA_DIR / "samples"
    images = [f for f in sample_dir.iterdir() if f.is_file()]

    total = min(MAX_RUNS, len(images))
    for i, image_path in enumerate(images[:MAX_RUNS]):
        print(f"Processing image {i+1}/{total}: {image_path}")
        try:
            run_id = await run_pipeline_async(str(image_path))
            print(f" → Run ID: {run_id}")
        except Exception as e:
            print(f" ✗ Error procesando {image_path}: {e}")
        print("-" * 50)

if __name__ == "__main__":
    # Ejecuta el main async
    asyncio.run(main())
