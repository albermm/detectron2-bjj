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
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime
import logging

#Load environment variabbles from .env
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
BUCKET_NAME = 'bjj-pics'
dynamodb = boto3.resource('dynamodb')
dynamodb_table = dynamodb.Table('BJJ_App_Table')

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

    # Construct URLs for S3 objects
    s3_base_url = f"https://{bucket}.s3.amazonaws.com/"
    keypoint_image_url = s3_base_url + keypoint_image_key
    keypoints_json_url = s3_base_url + keypoints_json_key

    # Prepare data for DynamoDB
    current_time = datetime.utcnow().isoformat()
    item = {
        'PK': f"JOB#{job_id}",  # Assuming job_id is unique
        'SK': f"JOB#{job_id}",
        'status': 'success',
        'position': predicted_position,
        'job_id': job_id,
        'image_url': keypoint_image_url,
        'keypoints_url': keypoints_json_url,
        'updatedAt': current_time
    }

    # Update DynamoDB
    try:
        response = dynamodb_table.update_item(
            Key={'PK': item['PK'], 'SK': item['SK']},
            UpdateExpression="SET #status = :status, #position = :position, image_url = :image_url, keypoints_url = :keypoints_url, updatedAt = :updatedAt",
            ExpressionAttributeNames={
                '#status': 'status',
                '#position': 'position'
            },
            ExpressionAttributeValues={
                ':status': item['status'],
                ':position': item['position'],
                ':image_url': item['image_url'],
                ':keypoints_url': item['keypoints_url'],
                ':updatedAt': item['updatedAt']
            },
            ReturnValues="UPDATED_NEW"
        )
        print(f"DynamoDB update successful: {response}")
    except Exception as e:
        print(f"Error updating DynamoDB: {str(e)}")
        # Consider how you want to handle this error. You might want to return an error response or continue with the process.

    return jsonify({
        'status': 'success',
        'keypoint_image_url': keypoint_image_url,
        'keypoints_json_url': keypoints_json_url,
        'predicted_position': predicted_position,
        'job_id': job_id
    })
@app.route('/get_job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    print(f"Retrieving job status for job_id: {job_id}")  # Use print instead of logger
    try:
        # Construct the key for querying DynamoDB
        key = {
            'PK': job_id,
            'SK': job_id
        }

        # Query DynamoDB
        response = dynamodb_table.get_item(Key=key)
        print(f"DynamoDB response: {response}")  # Use print instead of logger
        
        item = response.get('Item')
        
        if not item:
            print(f"Job not found for job_id: {job_id}")  # Use print instead of logger
            return jsonify({'error': 'Job not found'}), 404
        
        # Construct the response
        result = {
            'status': item.get('status'),
            'image_url': item.get('image_url'),
            'keypoints_url': item.get('keypoints_url'),
            'position': item.get('position'),
            'job_id': item.get('job_id'),
            'timestamp': item.get('timestamp')  # Changed from 'updatedAt' to 'timestamp'
        }

        print(f"Returning result for job_id {job_id}: {result}")  # Use print instead of logger
        return jsonify(result)

    except Exception as e:
        print(f"Error retrieving job status for job_id {job_id}: {str(e)}")  # Use print instead of logger
        return jsonify({'error': 'An error occurred while retrieving the job status'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
   

