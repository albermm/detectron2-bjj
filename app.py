import boto3
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64
import main
import os
from dotenv import load_dotenv

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

def upload_to_s3(file_name, file_content, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    s3.upload_fileobj(file_content, bucket, object_name)

    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{object_name}"

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Process the image using your main.py and helper.py functions
    keypoints, annotated_image = main.process_image(img)

    # Convert annotated image to bytes
    img_io = BytesIO()
    annotated_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Upload original image and annotated image to S3
    original_img_url = upload_to_s3(file.filename, file.stream, S3_BUCKET)
    annotated_img_url = upload_to_s3(f"annotated_{file.filename}", img_io, S3_BUCKET)

    response = {
        'keypoints': keypoints,
        'original_image_url': original_img_url,
        'annotated_image_url': annotated_img_url
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
