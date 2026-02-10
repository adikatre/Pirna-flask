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
        db_path = current_app.config['UPLOAD_FOLDER']
        
        # Verify db_path exists and has images
        if not os.path.exists(db_path):
             return jsonify({'message': 'Database path not found'}), 500

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
            # Structure: .../uploads/<uid>/<filename>
            
            # Extract UID
            # We assume structure is standard: uploads/uid/filename based on pfp.py
            path_parts = identity_path.split(os.sep)
            # User ID should be the parent folder name
            if len(path_parts) >= 2:
                uid = path_parts[-2]
            else:
                uid = None
            
            if uid:
                # Find user in DB
                user = User.query.filter_by(_uid=uid).first()
                if user:
                    return jsonify({
                        'match': True,
                        'name': user.name,
                        'uid': user.uid,
                        'distance': float(match['distance'])
                    }), 200
                else:
                     return jsonify({'match': True, 'message': 'User not found in DB', 'uid': uid}), 200
            
        return jsonify({'match': False, 'message': 'No user identified'}), 200

    except Exception as e:
        print(f"Server error in identify: {e}")
        return jsonify({'message': f'Server error: {str(e)}'}), 500
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
