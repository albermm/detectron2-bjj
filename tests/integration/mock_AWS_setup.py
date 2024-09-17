import pytest
import os
from unittest.mock import MagicMock

# Mock boto3
boto3 = MagicMock()

@pytest.fixture(scope='function')
def aws_credentials():
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'

@pytest.fixture(scope='function')
def s3_client(aws_credentials):
    return boto3.client('s3', region_name='us-east-1')

@pytest.fixture(scope='function')
def dynamodb_resource(aws_credentials):
    mock_resource = MagicMock()
    mock_table = MagicMock()
    mock_resource.Table.return_value = mock_table
    return mock_resource

# Mock app and client fixtures
@pytest.fixture
def app():
    from api.app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app

@pytest.fixture
def client(app):
    return app.test_client()