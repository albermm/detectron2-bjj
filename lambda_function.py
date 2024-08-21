import boto3
import cv2
import numpy as np
from utils.helper import Predictor
from utils.find_position import find_position

s3_client = boto3.client('s3')
BUCKET_NAME = 'bjj-pics'

def lambda_handler(event, context):
    # Get the uploaded file information
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Exit early if the object is in the outputs/ prefix
    if key.startswith('outputs/'):
        print(f"Skipping processing for output file: {key}")
        return {
            'statusCode': 200,
            'body': 'Skipped processing output file'
        }

    # Download the image from S3
    local_file_name = '/tmp/image.jpg'
    s3_client.download_file(bucket, key, local_file_name)

    # Process the image
    predictor = Predictor()
    output_path = '/tmp/output'
    keypoint_frame, densepose_frame, keypoints, densepose = predictor.onImage(local_file_name, output_path)

    # Find position
    predicted_position = find_position(keypoints)

    # Upload results back to S3
    original_filename = key.split('/')[-1]
    keypoints_key = f"outputs/keypoints_{original_filename}"
    densepose_key = f"outputs/densepose_{original_filename}"
    s3_client.upload_file(f'{output_path}_keypoints.jpg', BUCKET_NAME, keypoints_key)
    s3_client.upload_file(f'{output_path}_densepose.jpg', BUCKET_NAME, densepose_key)

    # Here you would typically store results in DynamoDB or another database

    return {
        'statusCode': 200,
        'body': 'Image processed successfully'
    }