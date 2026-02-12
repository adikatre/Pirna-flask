from flask import Blueprint, request, jsonify, current_app
from deepface import DeepFace
import base64
import os
import uuid
import tempfile
from model.user import User
import glob

identify_api_blueprint = Blueprint('identify_api', __name__, url_prefix='/api/identify')

@identify_api_blueprint.route('/', methods=['POST'])
def identify():
    temp_path = None
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'message': 'No image provided'}), 400

        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({'message': 'Invalid base64 image'}), 400

        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, 'wb') as f:
            f.write(image_data)

        # Use DeepFace to find the face in the database (UPLOAD_FOLDER)
        # We need to be careful about model loading time. 
        # DeepFace.find might re-load model on each call if not handled efficiently by library, 
        # but for this MVP scope it's acceptable.
        
        # db_path should be the root of uploads where user folders are located
        base_uploads = current_app.config['UPLOAD_FOLDER']
        if not base_uploads:
            return jsonify({'message': 'Server not configured for uploads'}), 500
            
        db_path = os.path.join(base_uploads, 'labeled_faces')
        
        # Verify db_path exists and has images
        if not os.path.exists(db_path):
             os.makedirs(db_path, exist_ok=True)
             # If empty, DeepFace might complain or just return empty
             # return jsonify({'message': 'Database path empty'}), 500

        try:
            # find returns a list of DataFrames
            dfs = DeepFace.find(img_path=temp_path, 
                                db_path=db_path, 
                                model_name="VGG-Face", 
                                enforce_detection=False, 
                                silent=True)
        except Exception as e:
            print(f"DeepFace find error: {e}")
            return jsonify({'message': f'Face recognition error: {str(e)}'}), 500
        
        if len(dfs) > 0 and not dfs[0].empty:
            # Get the first match (best match usually sorted by distance)
            match = dfs[0].iloc[0]
            identity_path = match['identity']
            
            # Identity path is absolute path to the matching image
            # Structure: .../uploads/labeled_faces/<label>/<filename>
            
            # Extract Label using os.path logic ensuring platform independence
            # We expect parent directory of the image to be the label
            label = os.path.basename(os.path.dirname(identity_path))
            
            return jsonify({
                'match': True,
                'name': label,
                'distance': float(match['distance'])
            }), 200
            
        return jsonify({'match': False, 'message': 'No user identified'}), 200

    except Exception as e:
        print(f"Server error in identify: {e}")
        return jsonify({'message': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@identify_api_blueprint.route('/add', methods=['POST'])
def add_face():
    """
    Add a labeled face image to the server-side database used by DeepFace.find.
    Expects JSON: {"image": "<base64 image>", "label": "person_name_or_uid"}
    Saves the decoded image to UPLOAD_FOLDER/labeled_faces/<label>/<uuid>.jpg and returns success.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'label' not in data:
            return jsonify({'message': 'Image and label required'}), 400

        label = data['label']
        # sanitize label to safe folder name
        safe_label = ''.join(c for c in label if c.isalnum() or c in ('-', '_')).strip()
        if not safe_label:
            return jsonify({'message': 'Invalid label'}), 400

        try:
            image_data = base64.b64decode(data['image'])
        except Exception as e:
            return jsonify({'message': 'Invalid base64 image'}), 400

        db_path = current_app.config.get('UPLOAD_FOLDER')
        if not db_path:
            return jsonify({'message': 'Server not configured for uploads'}), 500
        
        # Use a subdirectory for labeled faces
        labeled_faces_path = os.path.join(db_path, 'labeled_faces')
        person_dir = os.path.join(labeled_faces_path, safe_label)
        os.makedirs(person_dir, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(person_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)

        # Remove the pickle file to force DeepFace to re-index
        # DeepFace caches representations in .pkl files. 
        # If we add a new face, we must delete this cache or use enforce_detection=False finding logic which might be slow without cache.
        # But actually, DeepFace.find checks for changes if we are lucky, or we delete the representations pkl.
        representations_path = os.path.join(labeled_faces_path, "representations_vgg_face.pkl")
        if os.path.exists(representations_path):
            os.remove(representations_path)

        return jsonify({'message': 'Image saved', 'path': file_path}), 200
    except Exception as e:
        print(f"Error in add_face: {e}")
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@identify_api_blueprint.route('/delete_all', methods=['DELETE'])
def delete_all_faces():
    """
    Delete all labeled faces from the server-side database.
    """
    try:
        db_path = current_app.config.get('UPLOAD_FOLDER')
        if not db_path:
            return jsonify({'message': 'Server not configured for uploads'}), 500
            
        labeled_faces_path = os.path.join(db_path, 'labeled_faces')
        import shutil
        if os.path.exists(labeled_faces_path):
            shutil.rmtree(labeled_faces_path)
            os.makedirs(labeled_faces_path, exist_ok=True)
            
        return jsonify({'message': 'All labeled faces deleted'}), 200
    except Exception as e:
        print(f"Error in delete_all_faces: {e}")
        return jsonify({'message': f'Server error: {str(e)}'}), 500 
