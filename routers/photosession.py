import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.background import (
    PhotographyBackgroundRequest, PhotographyBackgroundResponse
    )

from src.pipelines import PipelineOrchestrator
#from config.config import ENV_VARIABLES, N_IMAGES

router = APIRouter()

async def stream_pipeline(img_bytes: bytes):
    orch = PipelineOrchestrator(mode="inference")
    # Paso 0: normalización
    norm = await orch.blocks["normalization"].run(img_bytes)
    yield json.dumps({"step": 0, **norm}) + "\n"
    if norm["white_bg"]:
        feat = await orch.blocks["features"].run(norm, img_bytes)
        yield json.dumps({"step": 1, **feat}) + "\n"
        bg = await orch.blocks["background"].run(norm, feat, img_bytes)
        yield json.dumps({"step": 2, "mask": bg["mask"], "images_bg": bg["images_bg"]}) + "\n"
        post = await orch.blocks["post"].run(bg["images_bg"])
        yield json.dumps({"step": 3, "output": post["output_image"]}) + "\n"
    else:
        post = await orch.blocks["post"].run(img_bytes)
        yield json.dumps({"step": 3, "output": post["output_image"]}) + "\n"

@router.post("/photography/background", response_class=StreamingResponse)
async def process(request: PhotographyBackgroundRequest):
    img = request.image_b64
    return StreamingResponse(
        stream_pipeline(img),
        media_type="application/x-ndjson"
    )

