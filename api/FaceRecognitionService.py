import base64
import os
import tempfile
import uuid
import shutil
import numpy as np
import requests
import json
from deepface import DeepFace
from flask import current_app

class FaceRecognitionService:
    # --- Logic Functions (Atomic) ---

    @staticmethod
    def decode_base64(base64_string):
        """Decodes base64 string to binary."""
        try:
            return base64.b64decode(base64_string)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {str(e)}")

    @staticmethod
    def save_temp_image(image_data):
        """Saves binary data to a temporary file, returns path."""
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, 'wb') as f:
            f.write(image_data)
        return temp_path

    @staticmethod
    def get_embedding(img_path):
        """Generates embedding for an image path using VGG-Face."""
        try:
            results = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
            if results and len(results) > 0:
                return results[0]["embedding"]
            return None
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    @staticmethod
    def calculate_distance(embedding1, embedding2):
        """Calculates cosine distance (1 - cosine similarity)."""
        a = np.array(embedding1)
        b = np.array(embedding2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity

    @staticmethod
    def sanitize_label(label):
        """Sanitizes labels for file system safety."""
        return ''.join(c for c in label if c.isalnum() or c in ('-', '_')).strip()

    @staticmethod
    def cleanup_path(path):
        """Removes a file safely."""
        if path and os.path.exists(path):
            os.remove(path)

    # --- Orchestrator Functions (Workflows) ---

    @classmethod
    def identify_face_workflow(cls, base64_image, threshold=0.4):
        """Orchestrates decoding, embedding generation, and DB matching via Spring."""
        img_data = cls.decode_base64(base64_image)
        temp_path = cls.save_temp_image(img_data)
        
        try:
            current_embedding = cls.get_embedding(temp_path)
            if not current_embedding:
                return {'match': False, 'message': 'Could not process face'}

            # Fetch existing faces from Spring (Centralized Storage)
            spring_url = "http://localhost:8585/api/person/faces"
            try:
                resp = requests.get(spring_url, timeout=5)
                if resp.status_code != 200:
                    return {'match': False, 'message': f'Spring API error: {resp.status_code}'}
                faces_data = resp.json()
            except Exception as e:
                return {'match': False, 'message': f'Connection error to Spring: {str(e)}'}

            best_match = None
            min_dist = 1.0
            
            for face in faces_data:
                stored_face_data = face.get('faceData')
                if not stored_face_data: continue
                
                try:
                    # Stored faceData is a JSON-stringified embedding array
                    stored_embedding = json.loads(stored_face_data)
                    dist = cls.calculate_distance(current_embedding, stored_embedding)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = face.get('uid')
                except Exception:
                    continue
            
            # Lower distance means better match (cosine distance)
            if best_match and min_dist <= (threshold or 0.4):
                return {
                    'match': True,
                    'name': best_match,
                    'distance': float(min_dist)
                }
            
            return {'match': False, 'message': 'No match found'}
            
        finally:
            cls.cleanup_path(temp_path)

    @classmethod
    def register_face_workflow(cls, label, base64_image):
        """Generates and returns an embedding for registration (No disk storage)."""
        img_data = cls.decode_base64(base64_image)
        temp_path = cls.save_temp_image(img_data)
        try:
            embedding = cls.get_embedding(temp_path)
            if not embedding:
                raise ValueError("DeepFace failed to generate embedding")
            return embedding # Returns list of floats
        finally:
            cls.cleanup_path(temp_path)

    @staticmethod
    def clear_database():
        """Clears local labeled_faces folder for legacy cleanup."""
        uploads = current_app.config.get('UPLOAD_FOLDER')
        if not uploads:
            return False
        db_path = os.path.join(uploads, 'labeled_faces')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            os.makedirs(db_path, exist_ok=True)
        return True

