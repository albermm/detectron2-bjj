import unittest
from unittest.mock import patch, MagicMock
import json
import lambda_function  # Replace with your Lambda function file name

class TestLambdaFunction(unittest.TestCase):

    @patch('lambda_function.boto3.client')
    @patch('lambda_function.boto3.resource')
    @patch('lambda_function.requests.post')
    def test_lambda_handler(self, mock_post, mock_dynamodb_resource, mock_s3_client):
        # Mock DynamoDB resource
        mock_dynamodb_table = MagicMock()
        mock_dynamodb_resource.return_value.Table.return_value = mock_dynamodb_table

        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_s3_client.return_value.head_object.return_value = {
            'Metadata': {'job-id': '12345'}
        }
        mock_s3_client = mock_s3_client

        # Mock EC2 processing response
        mock_post.return_value = MagicMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'status': 'success',
            'keypoint_image_url': 'http://example.com/keypoint_image',
            'keypoints_json_url': 'http://example.com/keypoints_json',
            'predicted_position': 'top-left'
        }

        # Event and context
        event = {
            'Records': [{
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'image.jpg'}
                }
            }]
        }
        context = {}

        # Call the Lambda function
        response = lambda_function.lambda_handler(event, context)

        # Check DynamoDB update
        mock_dynamodb_table.put_item.assert_called_once_with(
            Item={
                'job_id': '12345',
                'status': 'COMPLETED',
                'timestamp': unittest.mock.ANY,  # time.time() returns float
                'image_url': 'http://example.com/keypoint_image',
                'keypoints_url': 'http://example.com/keypoints_json',
                'position': 'top-left'
            }
        )

        # Check the Lambda function response
        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(json.loads(response['body']), 'Image processed successfully!')

if __name__ == '__main__':
    unittest.main()
