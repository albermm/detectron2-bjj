import pytest
from unittest.mock import patch
from utils.shared_utils import update_job_status

@patch('utils.shared_utils.dynamodb_table')
def test_update_job_status_integration(mock_dynamodb_table):
    job_id = 'test_job'
    user_id = 'test_user'
    status = 'COMPLETED'
    file_type = 'image'
    file_name = 'test.jpg'

    update_job_status(job_id, user_id, status, file_type, file_name)

    mock_dynamodb_table.put_item.assert_called_once()

    # Check the arguments of the put_item call
    call_args = mock_dynamodb_table.put_item.call_args
    assert call_args is not None, "put_item was not called"
    
    item = call_args[1]['Item']
    assert item['PK'] == f"USER#{user_id}"
    assert item['SK'] == f"JOB#{job_id}"
    assert item['status'] == status
    assert item['file_type'] == file_type
    assert item['file_name'] == file_name