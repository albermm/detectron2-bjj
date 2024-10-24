import os
import logging
from dotenv import load_dotenv
import boto3
from botocore.config import Config as BotoConfig
import uuid
from datetime import datetime
from decimal import Decimal

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'bjj-pics')
    DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME', 'BJJ_App_Table')
    EC2_BASE_URL = os.getenv('EC2_BASE_URL', 'http://52.72.247.7:5000')
    APP_PORT = int(os.getenv('APP_PORT', 5000))
    S3_REGION = os.getenv('S3_REGION', 'us-east-1')

    # S3 configuration
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    PRESIGNED_URL_EXPIRATION = 3600  # 1 hour

    # EC2 configuration
    API_TIMEOUT = 30  # seconds

    # Get the directory of the current script
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    


    # Model configuration
    # Set the MODEL_PATH relative to the project root
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.joblib')
    MAX_KEYPOINTS = 18
    KEYPOINT_MEAN = 0.0  # Value from model trainer
    KEYPOINT_STD = 1.0   # Value from model trainer

    # Keypoint configuration
    KEYPOINT_CONFIG = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    KEYPOINT_THRESHOLD = 0.1
    KEYPOINT_VIZ_THRESHOLD = 0.1  # For visualization
    BOUNDING_BOX_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    BOX_THRESHOLD = 0.3  # For bounding boxes

    # Quality thresholds
    MIN_KEYPOINT_QUALITY = 0.1  # Minimum quality to consider keypoints valid
    MIN_TRACKING_QUALITY = 0.2  # Minimum quality for tracking

    # File type configuration
    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')

# Use Config class attributes directly
BUCKET_NAME = Config.BUCKET_NAME
DYNAMODB_TABLE_NAME = Config.DYNAMODB_TABLE_NAME
EC2_BASE_URL = Config.EC2_BASE_URL
APP_PORT = Config.APP_PORT
S3_REGION = Config.S3_REGION

# AWS clients
s3_client = boto3.client('s3', region_name=S3_REGION, config=BotoConfig(signature_version='s3v4'))
dynamodb = boto3.resource('dynamodb')
dynamodb_table = dynamodb.Table(DYNAMODB_TABLE_NAME)

def generate_job_id():
    return str(uuid.uuid4())

def update_job_status(job_id, user_id, status, file_type, file_name, position=None, s3_path=None, **kwargs):
    try:
        now = datetime.utcnow()
        item = {
            'PK': f"USER#{user_id}",
            'SK': f"JOB#{job_id}",
            'status': status,
            'file_type': file_type,
            'file_name': file_name,
            'updatedAt': now.isoformat(),
            'updatedAtTimestamp': int(now.timestamp())
        }

        if position:
            item['position'] = position
        if s3_path:
            item['s3_path'] = s3_path

        # Add any additional kwargs to the item
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                item[key] = Decimal(str(value))
            elif isinstance(value, bool):
                item[key] = value
            else:
                item[key] = str(value)

        dynamodb_table.put_item(Item=item)
        logger.info(f"DynamoDB update successful for job {job_id}")
    except Exception as e:
        logger.error(f"Error updating DynamoDB: {str(e)}")
        raise

def get_s3_url(key):
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"

def validate_file_type(file_type):
    if file_type not in ['image', 'video']:
        raise ValueError('Invalid file type. Must be either "image" or "video"')

def validate_user_id(user_id):
    if not user_id:
        raise ValueError('User ID is required')
def get_job_details(job_id, user_id):
    try:
        response = dynamodb_table.get_item(
            Key={
                'PK': f"USER#{user_id}",
                'SK': f"JOB#{job_id}"
            }
        )
        item = response.get('Item')
        if not item:
            logger.warning(f"Job details not found for job_id: {job_id}, user_id: {user_id}")
            return None
        return item
    except Exception as e:
        logger.error(f"Error retrieving job details for job_id: {job_id}, user_id: {user_id}: {str(e)}")
        return None
#Make sure to export all necessary items
__all__ = [
    'logger',
    'Config',
    'BUCKET_NAME',
    'DYNAMODB_TABLE_NAME',
    'MODEL_PATH',
    'EC2_BASE_URL',
    'APP_PORT',
    's3_client',
    'dynamodb',
    'dynamodb_table',
    'generate_job_id',
    'update_job_status',
    'get_s3_url',
    'validate_file_type',
    'validate_user_id',
    'get_job_details'
]
'''
class Config:
    @classmethod
    def get_bucket_name(cls):
        return os.getenv('S3_BUCKET_NAME', 'bjj-pics')

    @classmethod
    def get_dynamodb_table_name(cls):
        return os.getenv('DYNAMODB_TABLE_NAME', 'BJJ_App_Table')

    @classmethod
    def get_ec2_base_url(cls):
        return os.getenv('EC2_BASE_URL', 'http://52.72.247.7:5000')

    @classmethod
    def get_app_port(cls):
    return int(os.getenv('APP_PORT', 5000))
    '''

    