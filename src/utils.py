import base64
import requests
from urllib.parse import urlparse
from pathlib import Path
from io import BytesIO
from PIL import Image
import aiohttp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64

def rotation_axes_only(azimuth_deg: float, rotation_deg: float) -> np.ndarray:
    """
    Matriz de rotación que solo aplica:
      1) yaw (azimuth) sobre Z global
      2) roll (rotation) sobre Z (local)
    Ignora cualquier pitch/polar.
    """
    az = np.deg2rad(azimuth_deg)
    ro = np.deg2rad(rotation_deg)

    # yaw sobre Z
    Rz_az = np.array([
        [ np.cos(az), -np.sin(az), 0],
        [ np.sin(az),  np.cos(az), 0],
        [          0,           0, 1]
    ])
    # roll sobre Z
    Rz_ro = np.array([
        [ np.cos(ro), -np.sin(ro), 0],
        [ np.sin(ro),  np.cos(ro), 0],
        [          0,           0, 1]
    ])

    # primero yaw, luego roll
    return Rz_ro @ Rz_az

def plot_orientation_base64(
    azimuth_deg: float,
    polar_deg: float,       # Este parámetro queda sin usar
    rotation_deg: float,
    fmt: str = "png"
) -> str:
    """
    Genera un Data URI con ejes X,Y,Z:
      - X (rojo) apuntando al espectador,
      - Y (verde) horizontal,
      - Z (azul) vertical,
    sin que el pitch/polar los incline.
    """
    # Determina mime_type
    suffix = fmt.lower()
    if suffix in ("jpg","jpeg"):
        mime_type, fmt = "image/jpeg", "jpeg"
    elif suffix == "webp":
        mime_type, fmt = "image/webp", "webp"
    else:
        mime_type, fmt = "image/png",   "png"

    # Ejes unitarios en world
    eX = np.array([1,0,0])
    eY = np.array([0,1,0])
    eZ = np.array([0,0,1])

    # Matriz de rotación de ejes (ignora polar)
    R_axes = rotation_axes_only(azimuth_deg, rotation_deg)

    # world → camera = R_axes^T
    Xc = R_axes.T @ eX
    Yc = R_axes.T @ eY
    Zc = R_axes.T @ eZ

    # Plot 3D
    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.quiver(0,0,0, *Xc, color='red',   linewidth=8, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, *Yc, color='green', linewidth=8, arrow_length_ratio=0.1)
    ax.quiver(0,0,0, *Zc, color='blue',  linewidth=8, arrow_length_ratio=0.1)

    # Vista frontal estricta
    ax.view_init(elev=0, azim=0)
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_axis_off()

    # Encode a base64
    buf = BytesIO()
    plt.savefig(buf, format=fmt, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f"data:{mime_type};base64,{data}"

async def download_image(session, url):
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            encoded_str = base64.b64encode(content).decode("utf-8")
            prefix = "data:image/png;base64,"
            return prefix + encoded_str
    except aiohttp.ClientError as e:
        print(f"Error al descargar {url}: {e}")
        return None

def load_image_as_base64(image_path: Path) -> str:
    """
    Carga una imagen de los formatos .png, .jpg, .jpeg, .webp, .jfif.
    - Si es PNG, la convierte a JPG en memoria, eliminando transparencia.
    - Retorna la cadena Base64 con el prefijo `data:image/<mime>;base64,`.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".jfif"}
    suffix = image_path.suffix.lower()

    if suffix not in valid_extensions:
        raise ValueError(f"Formato de archivo no soportado: {suffix}")

    # Abre la imagen con Pillow
    with Image.open(image_path) as img:
        # Si es PNG, la convertimos a JPG internamente
        if suffix == ".png":
            # Convertir a RGB si viene con transparencia (RGBA) 
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            
            # Guardamos en memoria como JPEG
            mime_type = "image/jpeg"
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")

        else:
            # Para formatos distintos a PNG, simplemente volvemos a guardarlos en su formato original
            buffer = BytesIO()
            # Utilizamos el atributo 'format' de Pillow para que respete (JPEG/WEBP/JFIF, etc.)
            # Si no reconociera la extensión, podemos mapear manualmente.
            img_format = img.format if img.format else "JPEG"  # fallback
            img.save(buffer, format=img_format)

            encoded_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Determinamos el mime_type según la extensión
            if suffix in [".jpg", ".jpeg", ".jfif"]:
                mime_type = "image/jpeg"
            elif suffix == ".webp":
                mime_type = "image/webp"
            else:
                # fallback a image/png si fuera algún caso no contemplado
                mime_type = "image/png"

        return f"data:{mime_type};base64,{encoded_bytes}"

def url_to_base64(url: str) -> str:
    """
    Descarga una imagen desde una URL y la convierte a una cadena Base64 con el prefijo 'data:image/<mime>;base64,'.
    
    Soporta los formatos: .png, .jpg, .jpeg, .webp, .jfif.
    - Si la imagen es PNG, se convierte a JPG en memoria eliminando la transparencia.
    
    Args:
        url (str): URL de la imagen.
    
    Returns:
        str: Cadena de la imagen codificada en Base64.
    
    Raises:
        ValueError: Si no se puede descargar la imagen o el formato no es soportado.
    """
    # Descargar la imagen desde la URL
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Error al descargar la imagen. Código de estado: {response.status_code}")
    image_data = response.content

    # Intentar obtener la extensión a partir de la URL
    parsed_url = urlparse(url)
    suffix = Path(parsed_url.path).suffix.lower()
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".jfif"}
    
    # Si la URL contiene extensión y no es de las válidas, se lanza un error
    if suffix and suffix not in valid_extensions:
        raise ValueError(f"Formato de archivo no soportado: {suffix}")

    # Abrir la imagen usando Pillow desde los datos descargados
    with Image.open(BytesIO(image_data)) as img:
        # Determinar si la imagen es PNG, ya sea por la extensión o por el formato detectado
        es_png = (suffix == ".png") or (not suffix and img.format and img.format.upper() == "PNG")
        
        if es_png:
            # Convertir a JPEG para eliminar la transparencia si es necesario
            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")
            mime_type = "image/jpeg"
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            # Para otros formatos, guardamos la imagen en su formato original
            buffer = BytesIO()
            img_format = img.format if img.format else "JPEG"  # fallback
            img.save(buffer, format=img_format)
            encoded_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Determinar el mime type según la extensión o el formato detectado
            if suffix in [".jpg", ".jpeg", ".jfif"]:
                mime_type = "image/jpeg"
            elif suffix == ".webp":
                mime_type = "image/webp"
            else:
                # En caso de no tener extensión, utilizar el formato detectado por Pillow
                if img.format:
                    if img.format.upper() == "JPEG":
                        mime_type = "image/jpeg"
                    elif img.format.upper() == "WEBP":
                        mime_type = "image/webp"
                    elif img.format.upper() == "PNG":
                        mime_type = "image/png"
                    else:
                        mime_type = "image/jpeg"  # fallback por defecto
                else:
                    mime_type = "image/jpeg"
                    
        return f"data:{mime_type};base64,{encoded_bytes}"

def image_to_base64(image_path):
    """
    Convert an image file to a Base64-encoded string with MIME type prefix.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64-encoded string with MIME type prefix (data:image/<mime>;base64,...)
    """
    # Determine MIME type based on file extension
    suffix = Path(image_path).suffix.lower()
    if suffix in [".jpg", ".jpeg", ".jfif"]:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"
    elif suffix == ".gif":
        mime_type = "image/gif"
    else:
        # Default fallback
        mime_type = "image/jpeg"
    
    # Read and encode the file
    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read()).decode("utf-8")
        
    return f"data:{mime_type};base64,{encoded_bytes}"

def base64_to_image(base64_string):
    """Convert a base64 string to a PIL Image object"""
    if ',' in base64_string:
        # Handle data URLs (e.g., "data:image/jpeg;base64,...")
        base64_string = base64_string.split(',', 1)[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def save_base64_to_file(base64_string, output_path):
    """Save a base64 string to a file"""
    img = base64_to_image(base64_string)
    img.save(output_path)
    return output_path