from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from botocore.config import Config
import uuid
from utils.helper import Predictor  
from utils.find_position import find_position

#Load environment variabbles from .env
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})





s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
BUCKET_NAME = 'bjj-pics'
dynamodb = boto3.resource('dynamodb')
dynamodb_table = dynamodb.Table('DYNAMODB_TABLE_NAME')

@app.route('/get_upload_url', methods=['GET'])
def get_upload_url():
    file_name = f"inputs/{uuid.uuid4()}.jpg"
    job_id = str(uuid.uuid4())  # Generate a unique job ID
    
    presigned_url = s3_client.generate_presigned_post(
        Bucket=BUCKET_NAME,
        Key=file_name,
        Fields={
            'x-amz-meta-job-id': job_id  # Set custom metadata
        },
        Conditions=[
            {'x-amz-meta-job-id': job_id}  # Ensure the metadata is set during upload
        ],
        ExpiresIn=3600
    )
    
    return jsonify({
        'presigned_post': presigned_url,
        'file_name': file_name,
        'job_id': job_id
    })



@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    file_name = data['file_name']
    job_id = data.get('job_id')
    bucket = BUCKET_NAME

    # Download the image from S3
    local_file_name = '/tmp/image.jpg'
    s3_client.download_file(bucket, file_name, local_file_name)

    # Process the image
    predictor = Predictor()
    output_path = '/tmp/output'
    keypoint_frame, keypoints, predicted_position = predictor.onImage(local_file_name, output_path)

    if keypoints is None:
        return jsonify({'status': 'error', 'message': 'Failed to process image'}), 500

    # Upload results back to S3
    keypoints_key = f"outputs/keypoints_{file_name}"
    keypoint_image_key = f"outputs/keypoint_frame_{file_name}"
    
    # Upload keypoint frame image
    s3_client.upload_file(f'{output_path}_keypoints.jpg', bucket, keypoint_image_key)

    # Upload keypoints JSON
    keypoints_json_key = f"outputs/keypoints_{file_name}.json"
    s3_client.upload_file(f'{output_path}_keypoints.json', bucket, keypoints_json_key)

    # Store metadata JSON
    metadata = {
        'status': 'success',
        'predicted_position': predicted_position,
        'message': 'Processing completed successfully',
        'job_id': job_id
    }
    metadata_key = f"outputs/metadata_{file_name}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata),
        ContentType='application/json'
    )
    
    # Construct URLs for S3 objects
    s3_base_url = f"https://{bucket}.s3.amazonaws.com/"
    keypoint_image_url = s3_base_url + keypoint_image_key
    keypoints_json_url = s3_base_url + keypoints_json_key
    metadata_url = s3_base_url + metadata_key
    
    return jsonify({
        'status': 'success',
        'keypoint_image_url': keypoint_image_url,
        'keypoints_json_url': keypoints_json_url,
        'predicted_position': predicted_position,
        'job_id': job_id
    })


@app.route('/get_job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    try:
        response = dynamodb_table.get_item(Key={'job_id': job_id})
        item = response.get('Item')
        
        if not item:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify({
            'status': item['status'],
            'image_url': item.get('image_url'),
            'keypoints_url': item.get('keypoints_url'),
            'position': item.get('position')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
   

