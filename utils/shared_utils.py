import os
import logging
from dotenv import load_dotenv
import boto3
from botocore.config import Config as BotoConfig
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    # S3 configuration
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'bjj-pics')
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    PRESIGNED_URL_EXPIRATION = 3600  # 1 hour

    # DynamoDB configuration
    DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME', 'BJJ_App_Table')

    # EC2 configuration
    EC2_BASE_URL = os.getenv('EC2_BASE_URL', 'http://52.72.247.7:5000')
    API_TIMEOUT = 30  # seconds

    # Application configuration
    APP_PORT = int(os.getenv('APP_PORT', 5000))

    # Model configuration
    MODEL_PATH = '../trained_model.joblib'
    MAX_KEYPOINTS = 18
    KEYPOINT_MEAN = 0.5  # Replace with actual value from model training
    KEYPOINT_STD = 0.2   # Replace with actual value from model training

    # Keypoint configuration
    KEYPOINT_CONFIG = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    KEYPOINT_THRESHOLD = 0.7

    # File type configuration
    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')

# AWS clients
s3_client = boto3.client('s3', config=BotoConfig(signature_version='s3v4'))
dynamodb = boto3.resource('dynamodb')
dynamodb_table = dynamodb.Table(Config.DYNAMODB_TABLE_NAME)

def generate_job_id():
    return str(uuid.uuid4())

def update_job_status(job_id, user_id, status, file_type, file_name, position=None, s3_path=None):
    try:
        item = {
            'PK': f"USER#{user_id}",
            'SK': f"JOB#{job_id}",
            'status': status,
            'file_type': file_type,
            'file_name': file_name,
            'updatedAt': datetime.utcnow().isoformat()
        }
        if position:
            item['position'] = position
        if s3_path:
            item['s3_path'] = s3_path

        dynamodb_table.put_item(Item=item)
        logger.info(f"DynamoDB update successful for job {job_id}")
    except Exception as e:
        logger.error(f"Error updating DynamoDB: {str(e)}")
        raise

def get_s3_url(key):
    return f"https://{Config.BUCKET_NAME}.s3.amazonaws.com/{key}"

def validate_file_type(file_type):
    if file_type not in ['image', 'video']:
        raise ValueError('Invalid file type. Must be either "image" or "video"')

def validate_user_id(user_id):
    if not user_id:
        raise ValueError('User ID is required')