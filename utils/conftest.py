import sys
from unittest.mock import MagicMock

# Mock detectron2
sys.modules['detectron2'] = MagicMock()
sys.modules['detectron2.utils'] = MagicMock()
sys.modules['detectron2.utils.visualizer'] = MagicMock()
sys.modules['detectron2.data'] = MagicMock()
sys.modules['detectron2.engine'] = MagicMock()
sys.modules['detectron2.config'] = MagicMock()

# Mock joblib
sys.modules['joblib'] = MagicMock()

# Add any other modules that need to be mocked here