import pytest
from unittest.mock import patch
from unittest.mock import MagicMock
import logging
from tests.integration.mock_AWS_setup import aws_credentials, s3_client, dynamodb_resource, app, client

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.mark.e2e
def test_image_processing_workflow(client, s3_client, dynamodb_resource):
    # Configure s3_client mock
    s3_client.generate_presigned_post.return_value = {
        'url': 'https://test-bucket.s3.amazonaws.com',
        'fields': {'key': 'test_file.jpg'}
    }

    # Configure dynamodb_resource mock
    mock_table = MagicMock()
    dynamodb_resource.Table.return_value = mock_table
    mock_table.get_item.return_value = {
        'Item': {
            'status': 'COMPLETED',
            'file_type': 'image',
            'position': 'test_position',
            's3_path': 'test_s3_path'
        }
    }
    
    # Step 1: Get upload URL
    response = client.get('/get_upload_url?file_type=image&user_id=test_user')
    if response.status_code != 200:
        logger.error(f"Get upload URL failed. Status code: {response.status_code}")
        logger.error(f"Response content: {response.get_data(as_text=True)}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    upload_data = response.get_json()
    logger.debug(f"Upload data: {upload_data}")
    
    # Step 2: Simulate image upload
    s3_client.put_object(Bucket='test-bucket', Key=upload_data['file_name'], Body=b'mock image data')
    
    # Step 3: Process image
    with patch('utils.helper.Predictor.onImage') as mock_onImage:
        mock_onImage.return_value = ('mock_frame', [{'keypoints': [1, 2, 3]}], 'MockPosition')
        response = client.post('/process_image', json={
            'file_name': upload_data['file_name'],
            'job_id': upload_data['job_id'],
            'user_id': 'test_user'
        })
    assert response.status_code == 200
    process_data = response.get_json()
    
    # Step 4: Get job status
    response = client.get(f"/get_job_status/{upload_data['job_id']}?user_id=test_user")
    assert response.status_code == 200
    status_data = response.get_json()
    assert status_data['status'] == 'COMPLETED'
    assert status_data['file_type'] == 'image'
    assert 'position' in status_data
    assert 's3_path' in status_data

    # Verify S3 and DynamoDB
    assert s3_client.get_object(Bucket='test-bucket', Key=f"outputs/keypoint_frame_{upload_data['file_name']}")
    table = dynamodb_resource.Table('test-table')
    job_item = table.get_item(Key={'PK': f"USER#test_user", 'SK': f"JOB#{upload_data['job_id']}"})['Item']
    assert job_item['status'] == 'COMPLETED'