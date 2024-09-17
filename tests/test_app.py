import pytest
from flask import json
from unittest.mock import patch, MagicMock
import warnings
from werkzeug.utils import import_string

warnings.filterwarnings("ignore", category=DeprecationWarning, module="werkzeug")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flask")

# Mock shared_utils before importing app
with patch('utils.shared_utils') as mock_shared_utils:
    mock_shared_utils.BUCKET_NAME = 'test-bucket'
    from api.app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_s3_client():
    with patch('api.app.s3_client') as mock:
        yield mock

@pytest.fixture
def mock_update_job_status():
    with patch('api.app.update_job_status') as mock:
        yield mock

@pytest.fixture
def mock_process_image():
    with patch('api.app.Predictor.onImage') as mock:
        yield mock

@pytest.fixture
def mock_process_video_async():
    with patch('api.app.process_video_async') as mock:
        yield mock
@pytest.fixture
def mock_dynamodb_table():
    with patch('api.app.dynamodb_table') as mock:
        yield mock

def test_get_upload_url(client, mock_s3_client, mock_update_job_status):
    mock_s3_client.generate_presigned_post.return_value = {'url': 'test_url', 'fields': {}}
    response = client.get('/get_upload_url?file_type=image&user_id=test_user')
   
def test_process_image(client, mock_process_image):
    mock_process_image.return_value = (
        'mock_keypoint_frame',
        'mock_keypoints',
        'TestPosition'
    )
    response = client.post('/process_image', json={
        'file_name': 'test.jpg',
        'job_id': 'test_job',
        'user_id': 'test_user'
    })
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'keypoint_image_url' in data
    assert data['predicted_position'] == 'TestPosition'

def test_process_video(client, mock_process_video_async):
    response = client.post('/process_video', json={
        'video_file_name': 'test.mp4',
        'job_id': 'test_job',
        'user_id': 'test_user'
    })

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'processing'
    assert data['job_id'] == 'test_job'
    assert mock_process_video_async.called

def test_get_job_status(client, mock_dynamodb_table):
    mock_dynamodb_table.get_item.return_value = {
        'Item': {
            'status': 'COMPLETED',
            'file_type': 'image'
        }
    }
    
    response = client.get('/get_job_status/test_job?user_id=test_user')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'COMPLETED'
    assert data['file_type'] == 'image'

def test_get_job_status(client, mock_dynamodb_table):
    mock_dynamodb_table.get_item.return_value = {
        'Item': {
            'status': 'COMPLETED',
            'file_type': 'image'
        }
    }
    response = client.get('/get_job_status/test_job?user_id=test_user')