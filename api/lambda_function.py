import json
import boto3
import os
import logging
import urllib3
from botocore.exceptions import ClientError
import time
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['BJJ_App_Table'])
http = urllib3.PoolManager()

def lambda_handler(event, context):
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Processing file: {key} from bucket: {bucket}")

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

        file_type = 'video' if key.lower().endswith(('.mp4', '.mov', '.avi')) else 'image'

        update_job_status(job_id, user_id, 'PROCESSING', file_type, key)

        ec2_url = f"{os.environ['EC2_BASE_URL']}/process_image" if file_type == 'image' else f"{os.environ['EC2_BASE_URL']}/process_video"
        try:
            response = http.request('POST', 
                                    ec2_url, 
                                    body=json.dumps({'file_name': key, 'job_id': job_id, 'user_id': user_id}),
                                    headers={'Content-Type': 'application/json'},
                                    timeout=30.0)
            
            if response.status == 200:
                response_data = json.loads(response.data.decode('utf-8'))
                logger.info(f"EC2 response: {response_data}")
                if response_data.get('status') == 'success' or response_data.get('status') == 'processing':
                    message = f"{file_type.capitalize()} processing initiated successfully!"
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
            else:
                raise Exception(f"Request failed with status {response.status}")
            
        except Exception as e:
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

def update_job_status(job_id, user_id, status, file_type, file_name, position=None, s3_path=None):
    try:
        item = {
            'PK': f"USER#{user_id}",
            'SK': f"JOB#{job_id}",
            'status': status,
            'file_type': file_type,
            'file_name': file_name,
            'updatedAt': Decimal(str(time.time()))  # Convert to Decimal
        }
        if position:
            item['position'] = position
        if s3_path:
            item['s3_path'] = s3_path

        # Convert any potential float values to Decimal
        for key, value in item.items():
            if isinstance(value, float):
                item[key] = Decimal(str(value))
            logger.info(f"Item key: {key}, value: {value}, type: {type(value)}")

        logger.info(f"Attempting to put item in DynamoDB: {json.dumps(item, default=str)}")
        table.put_item(Item=item)
        logger.info(f"DynamoDB update successful for job {job_id}")
    except Exception as e:
        logger.error(f"Error updating DynamoDB: {str(e)}")
        raise

# Add this helper function to handle Decimal serialization
def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError