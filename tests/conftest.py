import sys
from unittest.mock import MagicMock
import os

# Mock detectron2 and its submodules
sys.modules['detectron2'] = MagicMock()
sys.modules['detectron2.utils'] = MagicMock()
sys.modules['detectron2.utils.visualizer'] = MagicMock()
sys.modules['detectron2.data'] = MagicMock()
sys.modules['detectron2.engine'] = MagicMock()
sys.modules['detectron2.config'] = MagicMock()

# Mock boto3 and related modules
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.config'] = MagicMock()

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock environment variables
os.environ['S3_BUCKET_NAME'] = 'test-bucket'