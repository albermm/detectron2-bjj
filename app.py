# app.py on EC2
from flask import Flask, request, jsonify
import boto3
from botocore.config import Config
import uuid
from utils.helper import Predictor
from utils.find_position import find_position

app = Flask(__name__)

s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
BUCKET_NAME = 'bjj-pics'

@app.route('/get_upload_url', methods=['GET'])
def get_upload_url():
    file_name = f"inputs/{uuid.uuid4()}.jpg"
    presigned_url = s3_client.generate_presigned_url(
        'put_object',
        Params={'Bucket': BUCKET_NAME, 'Key': file_name},
        ExpiresIn=3600
    )
    return jsonify({'upload_url': presigned_url, 'file_name': file_name})

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    file_name = data['file_name']
    bucket = BUCKET_NAME

    # Download the image from S3
    local_file_name = '/tmp/image.jpg'
    s3_client.download_file(bucket, file_name, local_file_name)

    # Process the image
    predictor = Predictor()
    output_path = '/tmp/output'
    keypoint_frame, densepose_frame, keypoints, densepose = predictor.onImage(local_file_name, output_path)

    # Find position
    predicted_position = find_position(keypoints)

    # Upload results back to S3
    keypoints_key = f"outputs/keypoints_{file_name}"
    densepose_key = f"outputs/densepose_{file_name}"
    s3_client.upload_file(f'{output_path}_keypoints.jpg', bucket, keypoints_key)
    s3_client.upload_file(f'{output_path}_densepose.jpg', bucket, densepose_key)

    return jsonify({
        'status': 'success',
        'keypoints_file': keypoints_key,
        'densepose_file': densepose_key
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
