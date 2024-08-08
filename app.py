import os
import boto3
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import cv2
from utils.helper import Predictor

load_dotenv()

app = Flask(__name__)

# AWS S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET')
S3_REGION = os.getenv('S3_REGION')
S3_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
S3_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

predictor = Predictor()

def upload_to_s3(file_name, file_content, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    s3.upload_fileobj(file_content, bucket, object_name)

    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{object_name}"

@app.route('/')
def home():
    return 'This is the Flask app API'

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Process the image using your helper functions
    keypoint_frame, densepose_frame, keypoints, densepose = predictor.onImage(file.filename, 'output')

    # Convert annotated image to bytes
    keypoint_io = BytesIO()
    densepose_io = BytesIO()
    Image.fromarray(cv2.cvtColor(keypoint_frame, cv2.COLOR_BGR2RGB)).save(keypoint_io, 'PNG')
    Image.fromarray(cv2.cvtColor(densepose_frame, cv2.COLOR_BGR2RGB)).save(densepose_io, 'PNG')
    keypoint_io.seek(0)
    densepose_io.seek(0)

    # Upload original image and annotated image to S3
    original_img_url = upload_to_s3(file.filename, file.stream, S3_BUCKET)
    annotated_keypoint_img_url = upload_to_s3(f"annotated_keypoints_{file.filename}", keypoint_io, S3_BUCKET)
    annotated_densepose_img_url = upload_to_s3(f"annotated_densepose_{file.filename}", densepose_io, S3_BUCKET)

    response = {
        'keypoints': keypoints,
        'densepose': densepose,
        'original_image_url': original_img_url,
        'annotated_keypoints_image_url': annotated_keypoint_img_url,
        'annotated_densepose_image_url': annotated_densepose_img_url
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
