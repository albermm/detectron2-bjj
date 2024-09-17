import pytest
import boto3
from moto import mock_s3, mock_dynamodb

@pytest.fixture(scope='function')
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    import os
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'

@pytest.fixture(scope='function')
def s3_client(aws_credentials):
    with mock_s3():
        conn = boto3.client('s3', region_name='us-east-1')
        # Create a mock bucket
        conn.create_bucket(Bucket='test-bucket')
        yield conn

@pytest.fixture(scope='function')
def dynamodb_resource(aws_credentials):
    with mock_dynamodb():
        conn = boto3.resource('dynamodb', region_name='us-east-1')
        # Create a mock table
        conn.create_table(
            TableName='test-table',
            KeySchema=[
                {'AttributeName': 'PK', 'KeyType': 'HASH'},
                {'AttributeName': 'SK', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'PK', 'AttributeType': 'S'},
                {'AttributeName': 'SK', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        )
        yield conn

@pytest.fixture
def app(s3_client, dynamodb_resource):
    from api.app import app
    app.config['TESTING'] = True
    app.config['S3_CLIENT'] = s3_client
    app.config['DYNAMODB_RESOURCE'] = dynamodb_resource
    return app

@pytest.fixture
def client(app):
    return app.test_client()