import boto3
import json
import os
import time
from botocore.exceptions import ClientError
from shared_utils import logger, Config, update_job_status

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['BJJ_App_Table'])

def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        logger.info(f"Processing file: {key} from bucket: {bucket}")

        # Skip processing for already processed images
        if key.startswith('outputs/'):
            logger.info("Skipping processed image")
            return {
                'statusCode': 200,
                'body': json.dumps('Skipping processed image')
            }

        # Retrieve the job ID and user ID from S3 object metadata
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
            job_id = response['Metadata'].get('job-id')
            user_id = response['Metadata'].get('user-id')
            logger.info(f"Retrieved job_id: {job_id}, user_id: {user_id}")
        except ClientError as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            job_id = f"auto-{int(time.time())}"
            user_id = "unknown"
            logger.warning(f"Generated auto job_id: {job_id}, user_id: {user_id}")

        # Determine file type
        file_type = 'video' if key.lower().endswith(tuple(Config.VIDEO_EXTENSIONS)) else 'image'

        # Update DynamoDB with initial status
        update_job_status(job_id, user_id, 'PROCESSING', file_type, key)

        # Call the EC2 API to process the file
        ec2_url = f"{Config.EC2_BASE_URL}/process_image" if file_type == 'image' else f"{Config.EC2_BASE_URL}/process_video"
        try:
            response = requests.post(ec2_url, json={'file_name': key, 'job_id': job_id, 'user_id': user_id}, timeout=Config.API_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()
            
            logger.info(f"EC2 response: {response_data}")
            if response_data.get('status') == 'success' or response_data.get('status') == 'processing':
                message = f"{file_type.capitalize()} processing initiated successfully!"
                # For images, we update with the results. For videos, we just update the status.
                if file_type == 'image':
                    update_job_status(
                        job_id,
                        user_id,
                        'COMPLETED',
                        file_type,
                        key,
                        response_data.get('predicted_position'),
                        response_data.get('keypoint_image_url')
                    )
                else:
                    update_job_status(job_id, user_id, 'PROCESSING', file_type, key)
            else:
                message = f"{file_type.capitalize()} processing failed!"
                update_job_status(job_id, user_id, 'FAILED', file_type, key)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            message = f"{file_type.capitalize()} processing failed due to an error: {str(e)}"
            update_job_status(job_id, user_id, 'FAILED', file_type, key)
        
        return {
            'statusCode': 200,
            'body': json.dumps(message)
        }
    except Exception as e:
        logger.error(f"Unhandled exception in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {str(e)}")
        }