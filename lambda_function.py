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
    ec2_url = "http://http://3.90.218.53:5000/process_image"
    response = requests.post(ec2_url, json={'file_name': key})
    response_data = response.json()

    # Handle the response as needed
    print(f"EC2 response: {response_data}")

    return {
        'statusCode': 200,
        'body': json.dumps('Processing complete!')
    }
