#!/usr/bin/env python3
# download_dataset.py - Script para descargar imágenes desde URLs en un archivo Excel

import sys
import os
import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Para poder importar src/ y config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asegurarse de que el directorio de destino existe
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True, parents=True)

async def download_image(session, url, filename):
    """Descarga una imagen desde una URL y la guarda en el archivo especificado."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(filename, 'wb') as f:
                    f.write(content)
                return True
            else:
                logger.error(f"Error al descargar {url}: HTTP {response.status}")
                return False
    except Exception as e:
        logger.error(f"Error al descargar {url}: {e}")
        return False

async def download_all_images(excel_path, url_column="url"):
    """Lee un archivo Excel y descarga todas las imágenes de la columna url."""
    
    # Verificar que el archivo Excel existe
    if not os.path.exists(excel_path):
        logger.error(f"El archivo {excel_path} no existe.")
        return
    
    # Leer el archivo Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"Error al leer el archivo Excel: {e}")
        return
    
    # Verificar que la columna url existe
    if url_column not in df.columns:
        logger.error(f"La columna '{url_column}' no existe en el archivo Excel.")
        return
    
    # Filtrar URLs vacías
    urls = df[url_column].dropna().tolist()
    logger.info(f"Se encontraron {len(urls)} URLs para descargar.")
    
    # Crear sesión HTTP para reutilizar conexiones
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, url in enumerate(urls):
            # Crear nombre de archivo basado en el índice
            ext = os.path.splitext(url.split('?')[0])[-1] or '.jpg'  # Usar extensión de la URL o .jpg por defecto
            filename = SAMPLES_DIR / f"image_{i:04d}{ext}"
            
            # Crear tarea de descarga
            task = download_image(session, url, filename)
            tasks.append(task)
        
        # Ejecutar todas las descargas con una barra de progreso
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Descargando imágenes"):
            results.append(await f)
        
        # Mostrar resumen
        successful = sum(results)
        logger.info(f"Descarga completada: {successful} exitosas, {len(results) - successful} fallidas.")

def main():
    """Función principal que ejecuta la descarga de imágenes."""
    # Buscar archivos Excel en DATA_DIR
    excel_files = list(DATA_DIR.glob("*.xlsx"))
    
    if not excel_files:
        logger.error(f"No se encontraron archivos Excel en {DATA_DIR}")
        return
    
    # Si hay múltiples archivos, usar el primero
    excel_path = excel_files[0]
    if len(excel_files) > 1:
        logger.warning(f"Se encontraron múltiples archivos Excel. Usando {excel_path}")
    
    logger.info(f"Leyendo URLs desde {excel_path}")
    
    # Ejecutar la descarga asíncrona
    asyncio.run(download_all_images(str(excel_path)))

if __name__ == "__main__":
    main()
