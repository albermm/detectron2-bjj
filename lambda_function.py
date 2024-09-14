import boto3
import requests
import json
import os

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['BJJ_App_Table'])

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Skip processing for already processed images
    if key.startswith('outputs/'):
        return {
            'statusCode': 200,
            'body': json.dumps('Skipping processed image')
        }

    # Retrieve the job ID from S3 object metadata
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        job_id = response['Metadata'].get('x-amz-meta-job-id')
        print(f"job_id: {job_id}")
    except Exception as e:
        print(f"Error retrieving job ID: {str(e)}")
        job_id = None

    # Update DynamoDB with initial status
    if job_id:
        update_dynamodb(job_id, 'PROCESSING', None, None, None)

    # Call the EC2 API to process the image
    ec2_url = "http://52.72.247.7:5000/process_image"
    try:
        response = requests.post(ec2_url, json={'file_name': key, 'job_id': job_id})
        response.raise_for_status()
        response_data = response.json()
        
        print(f"EC2 response: {response_data}")

        if response_data.get('status') == 'success':
            message = 'Image processed successfully!'
            # Update DynamoDB with success status and results
            update_dynamodb(
                job_id,
                'COMPLETED',
                response_data.get('keypoint_image_url'),
                response_data.get('keypoints_json_url'),
                response_data.get('predicted_position')
            )
        else:
            message = 'Image processing failed!'
            update_dynamodb(job_id, 'FAILED', None, None, None)
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        message = f"Image processing failed due to an error: {str(e)}"
        update_dynamodb(job_id, 'FAILED', None, None, None)
    
    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }

def update_dynamodb(job_id, status, image_url, keypoints_url, position):
    if not job_id:
        print("No job ID provided, skipping DynamoDB update")
        return

    try:
        item = {
            'job_id': job_id,
            'status': status,
            'timestamp': int(time.time())
        }
        if image_url:
            item['image_url'] = image_url
        if keypoints_url:
            item['keypoints_url'] = keypoints_url
        if position:
            item['position'] = position

        table.put_item(Item=item)
        print(f"DynamoDB updated for job {job_id}")
    except Exception as e:
        print(f"Error updating DynamoDB: {str(e)}")