import pytest
from utils.shared_utils import update_job_status
from .mock_AWS_setup import dynamodb_resource

def test_update_job_status_integration(dynamodb_resource):
    table = dynamodb_resource.Table('test-table')
    
    job_id = 'test_job'
    user_id = 'test_user'
    status = 'COMPLETED'
    file_type = 'image'
    file_name = 'test.jpg'
    
    update_job_status(job_id, user_id, status, file_type, file_name)
    
    response = table.get_item(Key={'PK': f"USER#{user_id}", 'SK': f"JOB#{job_id}"})
    item = response.get('Item')
    
    assert item is not None
    assert item['status'] == status
    assert item['file_type'] == file_type
    assert item['file_name'] == file_name