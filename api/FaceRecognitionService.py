import base64
import os
import tempfile
import uuid
import shutil
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
    def perform_search(img_path, db_path):
        """Runs DeepFace search on a path."""
        return DeepFace.find(img_path=img_path, 
                             db_path=db_path, 
                             model_name="VGG-Face", 
                             enforce_detection=False, 
                             silent=True)

    @staticmethod
    def sanitize_label(label):
        """Sanitizes labels for file system safety."""
        return ''.join(c for c in label if c.isalnum() or c in ('-', '_')).strip()

    @staticmethod
    def save_labeled_image(label, image_data, base_path):
        """Saves image to label directory, returns file path."""
        person_dir = os.path.join(base_path, label)
        os.makedirs(person_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(person_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        return file_path

    @staticmethod
    def clear_representations(db_path):
        """Deletes DeepFace cache file."""
        rep_path = os.path.join(db_path, "representations_vgg_face.pkl")
        if os.path.exists(rep_path):
            os.remove(rep_path)

    @staticmethod
    def cleanup_path(path):
        """Removes a file safely."""
        if path and os.path.exists(path):
            os.remove(path)

    # --- Orchestrator Functions (Workflows) ---

    @classmethod
    def identify_face_workflow(cls, base64_image):
        """Orchestrates decoding, scanning, and matching logic."""
        img_data = cls.decode_base64(base64_image)
        temp_path = cls.save_temp_image(img_data)
        
        try:
            uploads = current_app.config.get('UPLOAD_FOLDER')
            if not uploads:
                raise RuntimeError("UPLOAD_FOLDER not configured")
            db_path = os.path.join(uploads, 'labeled_faces')
            os.makedirs(db_path, exist_ok=True)

            dfs = cls.perform_search(temp_path, db_path)
            
            if len(dfs) > 0 and not dfs[0].empty:
                match = dfs[0].iloc[0]
                label = os.path.basename(os.path.dirname(match['identity']))
                return {
                    'match': True,
                    'name': label,
                    'distance': float(match['distance'])
                }
            return {'match': False, 'message': 'No match found'}
            
        finally:
            cls.cleanup_path(temp_path)

    @classmethod
    def register_face_workflow(cls, label, base64_image):
        """Orchestrates sanitization, decoding, saving, and cache clearing."""
        safe_label = cls.sanitize_label(label)
        if not safe_label:
            raise ValueError("Invalid label")

        uploads = current_app.config.get('UPLOAD_FOLDER')
        if not uploads:
            raise RuntimeError("UPLOAD_FOLDER not configured")
        
        db_path = os.path.join(uploads, 'labeled_faces')
        img_data = cls.decode_base64(base64_image)
        
        file_path = cls.save_labeled_image(safe_label, img_data, db_path)
        cls.clear_representations(db_path)
        
        return file_path

    @staticmethod
    def clear_database():
        """Logic for fully clearing the database directory."""
        uploads = current_app.config.get('UPLOAD_FOLDER')
        if not uploads:
            raise RuntimeError("UPLOAD_FOLDER not configured")
        db_path = os.path.join(uploads, 'labeled_faces')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            os.makedirs(db_path, exist_ok=True)
        return True
