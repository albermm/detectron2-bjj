import boto3
from botocore.exceptions import ClientError

# Initialize the DynamoDB resource
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('BJJ_App_Table')

# Function to add a User
def add_user(user_id, username, email, created_at):
    try:
        response = table.put_item(
            Item={
                'PK': f'USER#{user_id}',
                'SK': f'METADATA#{user_id}',
                'username': username,
                'email': email,
                'createdAt': created_at
            }
        )
        print(f"User {user_id} added successfully")
    except ClientError as e:
        print(e.response['Error']['Message'])

# Function to add a Job
def add_job(job_id, status, created_at, updated_at):
    try:
        response = table.put_item(
            Item={
                'PK': f'JOB#{job_id}',
                'SK': f'METADATA#{job_id}',
                'status': status,
                'createdAt': created_at,
                'updatedAt': updated_at
            }
        )
        print(f"Job {job_id} added successfully")
    except ClientError as e:
        print(e.response['Error']['Message'])

# Function to add a User's Job
def add_user_job(user_id, job_id, job_name, created_at):
    try:
        response = table.put_item(
            Item={
                'PK': f'USER#{user_id}',
                'SK': f'JOB#{job_id}',
                'jobName': job_name,
                'createdAt': created_at
            }
        )
        print(f"User {user_id}'s job {job_id} added successfully")
    except ClientError as e:
        print(e.response['Error']['Message'])

# Function to add a Job Result
def add_job_result(job_id, result_id, image_url, keypoints_url, position, created_at):
    try:
        response = table.put_item(
            Item={
                'PK': f'JOB#{job_id}',
                'SK': f'RESULT#{result_id}',
                'image_url': image_url,
                'keypoints_url': keypoints_url,
                'position': position,
                'createdAt': created_at
            }
        )
        print(f"Job {job_id} result {result_id} added successfully")
    except ClientError as e:
        print(e.response['Error']['Message'])

if __name__ == "__main__":
    # Sample data for testing
    add_user('123', 'testuser', 'test@example.com', '2024-09-15')
    add_job('456', 'in_progress', '2024-09-15', '2024-09-16')
    add_user_job('123', '456', 'Sample Job', '2024-09-15')
    add_job_result('456', '789', 'http://example.com/image.jpg', 'http://example.com/keypoints.json', 'top', 
'2024-09-15')

