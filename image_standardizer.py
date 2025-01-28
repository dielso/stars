    import os
from PIL import Image
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageStandardizer:
    def __init__(self, input_folder: str, output_folder: str):
        """
        Initialize the image standardizer
        
        Args:
            input_folder: Path to folder containing original images
            output_folder: Path where standardized images will be saved
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.supported_formats = {'.avif', '.jpeg', '.jpg', '.webp'}
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    def get_celebrity_name(self, filepath: str) -> str:
        """Extract celebrity name from filename"""
        return Path(filepath).stem

    def convert_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Convert a single image to JPG format
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Get celebrity name and create output path
            celebrity_name = self.get_celebrity_name(image_path)
            output_path = os.path.join(self.output_folder, f"{celebrity_name}.jpg")
            
            # Skip if output already exists
            if os.path.exists(output_path):
                logger.info(f"Skipping {celebrity_name} - already exists")
                return True, ""

            # Open and convert image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Save as JPG
                img.save(output_path, 'JPEG', quality=95)
                
            logger.info(f"Successfully converted {celebrity_name}")
            return True, ""

        except Exception as e:
            error_msg = f"Error converting {image_path}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def process_all_images(self) -> List[str]:
        """
        Process all images in the input folder
        
        Returns:
            List of error messages (empty if all successful)
        """
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(glob.glob(os.path.join(self.input_folder, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(self.input_folder, f"*{ext.upper()}")))

        if not image_files:
            logger.warning(f"No supported images found in {self.input_folder}")
            return ["No images found"]

        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images in parallel
        errors = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.convert_image, image_files)
            
            for success, error in results:
                if not success:
                    errors.append(error)

        return errors

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Standardize celebrity images to JPG format')
    parser.add_argument('input_folder', help='Folder containing original images')
    parser.add_argument('output_folder', help='Folder where standardized images will be saved')
    args = parser.parse_args()

    standardizer = ImageStandardizer(args.input_folder, args.output_folder)
    errors = standardizer.process_all_images()
    
    if errors:
        logger.error("The following errors occurred:")
        for error in errors:
            logger.error(error)
    else:
        logger.info("All images processed successfully")

if __name__ == "__main__":
    main()