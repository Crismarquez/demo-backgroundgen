#!/usr/bin/env python3
# join_predictions.py - Script to collect predictions from batch runs into a single directory

import sys
import os
import shutil
from pathlib import Path
import re
import logging

# Add parent directory to path to import src/ and config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def join_predictions(version_pipeline="v1"):
    """
    Collects input images and final predictions from all batch runs and 
    copies them to a single directory with sequential prefixes.
    
    Args:
        version_pipeline (str): Version identifier for the pipeline
    """
    # Create target directory
    target_dir = DATA_DIR / "experiments" / version_pipeline
    target_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created target directory: {target_dir}")
    
    # Get all run directories
    runs_batch_dir = DATA_DIR / "runs_batch"
    if not runs_batch_dir.exists():
        logger.error(f"Batch runs directory not found: {runs_batch_dir}")
        return
    
    run_dirs = [d for d in runs_batch_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(run_dirs)} batch run directories")
    
    # Process each run directory
    for i, run_dir in enumerate(sorted(run_dirs)):
        logger.info(f"Processing run {i+1}/{len(run_dirs)}: {run_dir.name}")
        
        # Find input image
        input_file = run_dir / "images" / "input.jpeg"
        if not input_file.exists():
            # Try alternative extensions
            for ext in [".jpg", ".png"]:
                alt_input = run_dir / f"input{ext}"
                if alt_input.exists():
                    input_file = alt_input
                    break
        
        if not input_file.exists():
            logger.warning(f"Input image not found in {run_dir}")
            continue
        
        # Find final prediction images - improved pattern matching
        final_images = []
        # Look for files starting with "final_" in the main directory
        final_images.extend(list(run_dir.glob("final_*")))
        # Also look in potential subdirectories
        for subdir in run_dir.iterdir():
            if subdir.is_dir():
                final_images.extend(list(subdir.glob("final_*")))
        
        if not final_images:
            logger.warning(f"No final prediction images found in {run_dir}")
            continue
        
        # Copy input image with prefix
        input_ext = input_file.suffix
        target_input = target_dir / f"{i}_input{input_ext}"
        shutil.copy2(input_file, target_input)
        
        # Copy each final prediction with prefix
        for final_img in final_images:
            # Extract the part after "final_" to preserve any meaningful suffixes
            final_name = final_img.name
            target_final = target_dir / f"{i}_{final_name}"
            shutil.copy2(final_img, target_final)
        
        logger.info(f"  → Copied {1 + len(final_images)} files from {run_dir.name}")
    
    logger.info(f"All predictions joined in {target_dir}")

def main():
    """Main function to execute the prediction joining process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Join predictions from batch runs into a single directory")
    parser.add_argument("--version", "-v", default="v4", help="Version identifier for the pipeline")
    
    args = parser.parse_args()
    join_predictions(args.version)

if __name__ == "__main__":
    main()
