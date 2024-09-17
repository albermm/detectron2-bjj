import pytest
from unittest.mock import patch
from .mock_AWS_setup import s3_client, dynamodb_resource, app, client

@pytest.mark.e2e
def test_image_processing_workflow(client, s3_client, dynamodb_resource):
    # Step 1: Get upload URL
    response = client.get('/get_upload_url?file_type=image&user_id=test_user')
    assert response.status_code == 200
    upload_data = response.get_json()
    
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
    
    # Verify S3 and DynamoDB
    assert s3_client.get_object(Bucket='test-bucket', Key=f"outputs/keypoint_frame_{upload_data['file_name']}")
    table = dynamodb_resource.Table('test-table')
    job_item = table.get_item(Key={'PK': f"USER#test_user", 'SK': f"JOB#{upload_data['job_id']}"})['Item']
    assert job_item['status'] == 'COMPLETED'