import pytest
from app import app
from unittest.mock import patch, MagicMock
from utils.shared_utils import Config

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_upload_url_success(client):
    with patch('app.s3_client.generate_presigned_post') as mock_presigned_post:
        mock_presigned_post.return_value = {'url': 'https://test-bucket.s3.amazonaws.com', 'fields': {}}
        response = client.get('/get_upload_url?file_type=image&user_id=test_user')
        assert response.status_code == 200
        assert 'presigned_post' in response.json
        assert 'file_name' in response.json
        assert 'job_id' in response.json
        assert 'user_id' in response.json

def test_get_upload_url_invalid_file_type(client):
    response = client.get('/get_upload_url?file_type=invalid&user_id=test_user')
    assert response.status_code == 400
    assert 'error' in response.json

def test_get_upload_url_missing_user_id(client):
    response = client.get('/get_upload_url?file_type=image')
    assert response.status_code == 400
    assert 'error' in response.json

def test_process_image_success(client):
    with patch('app.s3_client.download_file'), \
         patch('app.Predictor.onImage') as mock_on_image, \
         patch('app.s3_client.upload_file'), \
         patch('app.update_job_status'):
        mock_on_image.return_value = (None, [[[0, 0, 0]]], 'test_position')
        data = {'file_name': 'test.jpg', 'job_id': 'test_job', 'user_id': 'test_user'}
        response = client.post('/process_image', json=data)
        assert response.status_code == 200
        assert response.json['status'] == 'success'
        assert 'keypoint_image_url' in response.json
        assert 'keypoints_json_url' in response.json
        assert response.json['predicted_position'] == 'test_position'

def test_process_image_failure(client):
    with patch('app.s3_client.download_file'), \
         patch('app.Predictor.onImage') as mock_on_image:
        mock_on_image.return_value = (None, None, None)
        data = {'file_name': 'test.jpg', 'job_id': 'test_job', 'user_id': 'test_user'}
        response = client.post('/process_image', json=data)
        assert response.status_code == 400
        assert 'error' in response.json

def test_process_video_success(client):
    with patch('app.s3_client.download_file'), \
         patch('app.process_video_async'), \
         patch('app.update_job_status'):
        data = {'video_file_name': 'test.mp4', 'job_id': 'test_job', 'user_id': 'test_user'}
        response = client.post('/process_video', json=data)
        assert response.status_code == 200
        assert response.json['status'] == 'processing'
        assert response.json['job_id'] == 'test_job'
        assert response.json['user_id'] == 'test_user'

def test_process_video_failure(client):
    with patch('app.s3_client.download_file') as mock_download:
        mock_download.side_effect = Exception('Download failed')
        data = {'video_file_name': 'test.mp4', 'job_id': 'test_job', 'user_id': 'test_user'}
        response = client.post('/process_video', json=data)
        assert response.status_code == 500
        assert 'error' in response.json

def test_get_job_status_success(client):
    mock_item = {
        'PK': 'USER#test_user',
        'SK': 'JOB#test_job',
        'status': 'COMPLETED',
        'file_type': 'image',
        'file_name': 'test.jpg',
        'updatedAt': '2023-09-16T12:00:00',
        'position': 'test_position',
        's3_path': 'test_path'
    }
    with patch('app.dynamodb_table.get_item') as mock_get_item:
        mock_get_item.return_value = {'Item': mock_item}
        response = client.get('/get_job_status/test_job?user_id=test_user')
        assert response.status_code == 200
        assert response.json == mock_item

def test_get_job_status_not_found(client):
    with patch('app.dynamodb_table.get_item') as mock_get_item:
        mock_get_item.return_value = {}
        response = client.get('/get_job_status/test_job?user_id=test_user')
        assert response.status_code == 404
        assert 'error' in response.json

def test_get_job_status_missing_user_id(client):
    response = client.get('/get_job_status/test_job')
    assert response.status_code == 400
    assert 'error' in response.json