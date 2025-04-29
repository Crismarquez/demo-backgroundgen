from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class PhotographyBackgroundRequest(BaseModel):
    """Modelo de datos para la petición de generación de fondo fotográfico."""
    image_b64: str # Base64 encoded images
    preferences: Optional[str] = None

class PhotographyBackgroundResponse(BaseModel):
    """Modelo de datos para la respuesta de generación de fondo fotográfico."""
    run_id: str  # ID único de la ejecución
    output_path: str  # Ruta de la imagen de salida
    status: str = "success"  # Estado de la operación
    metadata: Optional[Dict[str, Any]] = None  # Metadatos adicionales del procesamiento