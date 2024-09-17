import pytest
from unittest.mock import patch
from tests.integration.mock_AWS_setup import aws_credentials, s3_client, app, client


def test_process_image_api_integration(client, s3_client):
    with patch('utils.helper.Predictor.onImage') as mock_onImage:
        mock_onImage.return_value = ('mock_frame', [{'keypoints': [1, 2, 3]}], 'MockPosition')
        
        # Upload a mock image to S3
        s3_client.put_object(Bucket='test-bucket', Key='inputs/test_user/test.jpg', Body=b'mock image data')
        
        response = client.post('/process_image', json={
            'file_name': 'inputs/test_user/test.jpg',
            'job_id': 'test_job',
            'user_id': 'test_user'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert data['predicted_position'] == 'MockPosition'
        
        # Verify that the processed image was uploaded to S3
        assert s3_client.get_object(Bucket='test-bucket', Key='outputs/keypoint_frame_inputs/test_user/test.jpg')
    pass