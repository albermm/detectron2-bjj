from flask import Flask, request, jsonify
import json
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
    #keypoint_frame, densepose_frame, keypoints, densepose = predictor.onImage(local_file_name, output_path)
    keypoint_frame, keypoints = predictor.onImage(local_file_name, output_path)


    # Find position from keypoints
    predicted_position = find_position(keypoints)

    # Upload results back to S3
    keypoints_key = f"outputs/keypoints_{file_name}"
    #densepose_key = f"outputs/densepose_{file_name}"
    s3_client.upload_file(f'{output_path}_keypoints.jpg', bucket, keypoints_key)
    #s3_client.upload_file(f'{output_path}_densepose.jpg', bucket, densepose_key)

    # Store metadata
    metadata = {
        'status': 'success',
        'predicted_position': predicted_position,
        'message': 'Processing completed successfully'
    }
    metadata_key = f"outputs/metadata_{file_name}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata),
        ContentType='application/json'
    )
    return jsonify({
        'status': 'success',
        'keypoints_file': keypoints_key,
        'position': predicted_position
    })

@app.route('/get_result/<file_name>', methods=['GET'])
def get_result(file_name):
    bucket = BUCKET_NAME
    keypoints_key = f"outputs/keypoints_{file_name}"
    metadata_key = f"outputs/metadata_{file_name}.json"
    #densepose_key = f"outputs/densepose_{file_name}"
    try:
        # Check if the keypoints file exists in S3
        s3_client.head_object(Bucket=bucket, Key=keypoints_key)
        
        # Retrieve metadata (including predicted_position and status)
        metadata_object = s3_client.get_object(Bucket=bucket, Key=metadata_key)
        metadata = json.loads(metadata_object['Body'].read().decode('utf-8'))
        #s3_client.head_object(Bucket=bucket, Key=densepose_key)
        return jsonify({
            'keypoints_file': keypoints_key,
            'status': metadata.get('status', 'success'),
            'position': metadata.get('predicted_position'),
            'message': metadata.get('message', '')
        })
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return jsonify({'status': 'processing'})
        else:
            return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
   

