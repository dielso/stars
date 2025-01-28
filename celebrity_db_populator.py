import os
import dlib
import glob
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import Dict, Optional, List
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CelebrityFaceDB:
    def __init__(self, predictor_path: str, face_rec_model_path: str, collection_name: str = "celebrity_faces"):
        """Initialize the celebrity face database"""
        logger.info("Loading models...")
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        
        logger.info("Initializing database...")
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path="celebrity_db")
        
        # Create collection for face embeddings
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Celebrity face embeddings", "embedding_size": 128}
        )

    def compute_face_descriptor(self, image_path: str) -> Optional[np.ndarray]:
        """Compute face descriptor for a given image"""
        try:
            img = dlib.load_rgb_image(image_path)
            dets = self.detector(img, 1)
            
            if len(dets) == 0:
                logger.warning(f"No face detected in {image_path}")
                return None
                
            # Get the first face detected
            d = dets[0]
            shape = self.sp(img, d)
            face_chip = dlib.get_face_chip(img, shape)
            face_descriptor = self.facerec.compute_face_descriptor(face_chip)
            
            return np.array(face_descriptor)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def add_celebrity(self, image_path: str) -> bool:
        try:
            celebrity_name = Path(image_path).stem
            
            descriptor = self.compute_face_descriptor(image_path)
            if descriptor is None:
                return False

            # Debug print
            logger.info(f"Descriptor shape: {descriptor.shape}")
            
            metadata = {
                "name": celebrity_name,
                "image_path": image_path,
                "added_date": datetime.now().isoformat()
            }
            
            self.collection.add(
                embeddings=[descriptor.tolist()],
                documents=[celebrity_name],
                metadatas=[metadata],
                ids=[f"celeb_{celebrity_name}"]
            )
            
            logger.info(f"Successfully added {celebrity_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding {image_path}: {str(e)}")
            return False

    # In CelebrityFaceDB class, modify populate_database method:
    def populate_database(self, images_folder: str) -> Dict[str, List[str]]:
        """Populate database with all celebrity images in the folder"""
        results = {
            "successful": [],
            "failed": []
        }

        image_files = glob.glob(os.path.join(images_folder, "*.jpg"))
        
        for image_path in image_files:
            celebrity_name = Path(image_path).stem
            if self.add_celebrity(image_path):
                results["successful"].append(celebrity_name)
            else:
                results["failed"].append(celebrity_name)
        
        return results
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        collection_data = self.collection.get()
        return {
            "total_celebrities": len(collection_data["ids"]),
            "celebrity_names": sorted([meta["name"] for meta in collection_data["metadatas"]])
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Populate celebrity face database')
    parser.add_argument('predictor_path', help='Path to shape predictor model')
    parser.add_argument('face_rec_model_path', help='Path to face recognition model')
    parser.add_argument('images_folder', help='Folder containing celebrity images')
    args = parser.parse_args()

    # Initialize database
    db = CelebrityFaceDB(args.predictor_path, args.face_rec_model_path)
    
    # Populate database
    logger.info("Starting database population...")
    results = db.populate_database(args.images_folder)
    
    # Print results
    logger.info("\nPopulation Results:")
    logger.info(f"Successfully added: {len(results['successful'])} celebrities")
    logger.info(f"Failed to add: {len(results['failed'])} celebrities")
    
    if results['failed']:
        logger.info("\nFailed celebrities:")
        for name in results['failed']:
            logger.info(f"- {name}")
    
    # Print database stats
    stats = db.get_database_stats()
    logger.info(f"\nTotal celebrities in database: {stats['total_celebrities']}")

if __name__ == "__main__":
    main()