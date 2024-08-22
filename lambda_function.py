import boto3
import requests
import json

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Skip processing for already processed images
    if key.startswith('outputs/'):
        return {
            'statusCode': 200,
            'body': json.dumps('Skipping processed image')
        }

    # Call the EC2 API to process the image
    ec2_url = "http://54.165.110.106:5000/process_image"
    try:
        response = requests.post(ec2_url, json={'file_name': key})
        response.raise_for_status()  # Check if the request was successful
        response_data = response.json()
        
        # Handle the response as needed
        print(f"EC2 response: {response_data}")

        # Check for success status
        if response_data.get('status') == 'success':
            message = 'Image processed successfully!'
        else:
            message = 'Image processing failed!'
        
    except requests.exceptions.RequestException as e:
        # Handle any errors with the HTTP request
        print(f"Request failed: {e}")
        message = f"Image processing failed due to an error: {str(e)}"
    
    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }

