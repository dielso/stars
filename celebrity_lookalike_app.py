import streamlit as st
import dlib
import chromadb
import numpy as np
import cv2
from PIL import Image
import os
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
import tempfile
from dotenv import load_dotenv


class FaceSwapSystem:
    def __init__(self):
        """Initialize the face swap system"""
        self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download = True, download_zip = True)
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def enhance_face_pr(self, img):
        load_dotenv()

    
        restorer = GFPGANer(
            model_path=os.getenv('GFPGAN_PATH'),
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None)
        
            # restore faces and background if necessary
        _, _, restored_image = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            weight=0.4)
        return restored_image


    def swap_face(self, source_img, target_img):
        """Swap face from source image to target image"""
        # Convert PIL images to cv2 format
        source_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        source_face = self.app.get(source_cv)[0]
        target_face = self.app.get(target_cv)[0]
        
        # Perform face swap
        result = self.swapper.get(target_cv, target_face, source_face, paste_back=True)
        
        # Convert back to PIL Image
        result_restored = self.enhance_face_pr(result)
        result_rgb = cv2.cvtColor(result_restored, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

class FaceLookalikeSystem:
    def __init__(self, predictor_path: str, face_rec_model_path: str, collection_name: str = "celebrity_faces"):
        """Initialize the face recognition system"""
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path="celebrity_db")
        self.collection = self.client.get_collection(name=collection_name)

    def compute_face_descriptor(self, image) -> np.ndarray:
        """Compute face descriptor for an image"""
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Detect faces
        dets = self.detector(image, 1)
        
        if len(dets) == 0:
            raise ValueError("No face detected in image")
            
        # Get the first face detected
        d = dets[0]
        shape = self.sp(image, d)
        face_chip = dlib.get_face_chip(image, shape)
        face_descriptor = self.facerec.compute_face_descriptor(face_chip)
        
        return np.array(face_descriptor)

    def find_lookalike(self, image, n_results: int = 1):
        """Find celebrity lookalike for the given image"""
        try:
            # Compute descriptor
            descriptor = self.compute_face_descriptor(image)
            
            # Query the database
            results = self.collection.query(
                query_embeddings=[descriptor.tolist()],
                n_results=n_results
            )
            
            if not results['ids'][0]:
                return None
            
            # Get the best match
            best_match = {
                'celebrity_name': results['metadatas'][0][0]['name'],
                'image_path': results['metadatas'][0][0]['image_path'],
                'similarity': self._compute_similarity(results['distances'][0][0])
            }
            
            return best_match
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

    def _compute_similarity(self, distance: float) -> float:
        """Convert distance to similarity percentage"""
        if distance < 0.6:
            return 100.0
        elif distance > 1.0:
            return 0.0
        else:
            return ((1 - distance) / 0.40) * 100

@st.cache_resource
def load_face_swap_system():
    return FaceSwapSystem()


def main():
    st.set_page_config(layout="wide")
    st.title("Celebrity Lookalike Finder")
    img_size_thumb = [600,600]
    
    # Initialize systems
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    
    try:
        recognition_system = FaceLookalikeSystem(predictor_path, face_rec_model_path)
        swap_system = load_face_swap_system()
        # interpolate_system = load_face_interpolation_system()
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()

    st.header("Upload Your Photo")
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

    # Create two columns
    col1, col2 = st.columns(2)

    result = None
    with col1:
        
        if uploaded_file is not None:
            # Display uploaded image
            st.header("You")
            with st.columns(3)[1]:
                image = Image.open(uploaded_file)
                image.thumbnail(img_size_thumb)
                st.image(image, caption="Your Photo")
            try:
                # Find lookalike
                result = recognition_system.find_lookalike(image)
                
                if result:
                    with col2:
                        st.header(result['celebrity_name'])
                        with st.columns(3)[1]:
                            # Load and display celebrity image
                            celebrity_image = Image.open(result['image_path'])
                            celebrity_image.thumbnail(img_size_thumb)
                            st.image(celebrity_image, caption=f"Celebrity: {result['celebrity_name']}")
                            # st.metric("Similarity Score", f"{result['similarity']:.1f}%")
                else:
                    with col2:
                        st.warning("No clear celebrity match found. Try another photo!")
                        
            except Exception as e:
                st.error(f"Error finding lookalike: {str(e)}")
        
        if result is not None:
            if st.button("Try Face Swap!"):
                try:
                    with st.spinner("Performing face swap..."):
                        # Swap faces
                        swapped_image = swap_system.swap_face(image, celebrity_image)
                        # swapped_image = interpolate_system.interpolate_faces(image, celebrity_image)
                        pres = swapped_image.copy()
                        pres.thumbnail([700,700])
                        with st.columns(3)[1]:
                            st.image(pres, caption="Face Swapped Result")
                        # Save to temporary file first
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            swapped_image.save(tmp_file.name)
                            with open(tmp_file.name, 'rb') as f:
                                st.download_button(
                                    "Download Face Swapped Image",
                                    f.read(),
                                    file_name=f"face_swap_with_{result['celebrity_name']}.jpg",
                                    mime="image/jpeg"
                                )
                except Exception as e:
                    st.error(f"Error in face swap: {str(e)}")

if __name__ == "__main__":
    main()