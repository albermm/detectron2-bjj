from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.shared_utils import Config
from utils.shared_utils import (
    s3_client, dynamodb_table, generate_job_id,
    update_job_status, get_s3_url, validate_file_type, validate_user_id, logger, Config
)
from utils.helper import Predictor, process_video_async
from threading import Thread
from flasgger import Swagger

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
Swagger(app)

@app.route('/get_upload_url', methods=['GET'])
def get_upload_url():
    """
    Get a pre-signed URL for file upload
    ---
    parameters:
      - name: file_type
        in: query
        type: string
        enum: [image, video]
        required: true
        description: Type of file to upload
      - name: user_id
        in: query
        type: string
        required: true
        description: ID of the user
    responses:
      200:
        description: Successful response
        schema:
          properties:
            presigned_post:
              type: object
            file_name:
              type: string
            job_id:
              type: string
            user_id:
              type: string
      400:
        description: Bad request
      500:
        description: Server error
    """
    try:
        file_type = request.args.get('file_type', 'image')
        user_id = request.args.get('user_id')

        validate_file_type(file_type)
        validate_user_id(user_id)

        file_extension = '.jpg' if file_type == 'image' else '.mp4'
        file_name = f"inputs/{user_id}/{generate_job_id()}{file_extension}"
        job_id = generate_job_id()
        content_type = 'image/jpeg' if file_type == 'image' else 'video/mp4'

        presigned_url = s3_client.generate_presigned_post(
            Bucket=Config.BUCKET_NAME,
            Key=file_name,
            Fields={
                'x-amz-meta-job-id': job_id,
                'x-amz-meta-user-id': user_id,
                'Content-Type': content_type,
            },
            Conditions=[
                {'x-amz-meta-job-id': job_id},
                {'x-amz-meta-user-id': user_id},
                ['content-length-range', 1, Config.MAX_FILE_SIZE],
                {'Content-Type': content_type},
            ],
            ExpiresIn=Config.PRESIGNED_URL_EXPIRATION
        )

        update_job_status(job_id, user_id, 'PENDING', file_type, file_name)

        return jsonify({
            'presigned_post': presigned_url,
            'file_name': file_name,
            'job_id': job_id,
            'user_id': user_id
        })

    except ValueError as ve:
        logger.warning(f"Validation error in get_upload_url: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_upload_url: {str(e)}")
        return jsonify({'error': 'An error occurred while generating upload URL'}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Process an uploaded image
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            file_name:
              type: string
            job_id:
              type: string
            user_id:
              type: string
    responses:
      200:
        description: Successful response
        schema:
          properties:
            status:
              type: string
            keypoint_image_url:
              type: string
            keypoints_json_url:
              type: string
            predicted_position:
              type: string
            job_id:
              type: string
      400:
        description: Bad request
      500:
        description: Server error
    """
    try:
        data = request.json
        file_name = data['file_name']
        job_id = data['job_id']
        user_id = data['user_id']

        local_file_name = '/tmp/image.jpg'
        s3_client.download_file(Config.BUCKET_NAME, file_name, local_file_name)

        predictor = Predictor()
        output_path = '/tmp/output'
        keypoint_frame, keypoints, predicted_position = predictor.onImage(local_file_name, output_path)

        if keypoints is None:
            raise ValueError('Failed to process image')

        keypoint_image_key = f"outputs/keypoint_frame_{file_name}"
        keypoints_json_key = f"outputs/keypoints_{file_name}.json"

        s3_client.upload_file(f'{output_path}_keypoints.jpg', Config.BUCKET_NAME, keypoint_image_key)
        s3_client.upload_file(f'{output_path}_keypoints.json', Config.BUCKET_NAME, keypoints_json_key)

        keypoint_image_url = get_s3_url(keypoint_image_key)
        keypoints_json_url = get_s3_url(keypoints_json_key)

        update_job_status(
            job_id,
            user_id,
            'COMPLETED',
            'image',
            file_name,
            predicted_position,
            keypoint_image_url
        )

        return jsonify({
            'status': 'success',
            'keypoint_image_url': keypoint_image_url,
            'keypoints_json_url': keypoints_json_url,
            'predicted_position': predicted_position,
            'job_id': job_id
        })

    except ValueError as ve:
        logger.warning(f"Validation error in process_image: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Process an uploaded video
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            video_file_name:
              type: string
            job_id:
              type: string
            user_id:
              type: string
    responses:
      200:
        description: Successful response
        schema:
          properties:
            status:
              type: string
            job_id:
              type: string
            user_id:
              type: string
            message:
              type: string
      500:
        description: Server error
    """
    try:
        data = request.json
        video_file_name = data['video_file_name']
        job_id = data['job_id']
        user_id = data['user_id']

        update_job_status(job_id, user_id, 'PROCESSING', 'video', video_file_name)

        local_video_path = '/tmp/video.mp4'
        s3_client.download_file(Config.BUCKET_NAME, video_file_name, local_video_path)

        output_path = '/tmp/output'
        
        # Start video processing in a background thread
        thread = Thread(target=process_video_async, args=(local_video_path, output_path, job_id, user_id))
        thread.start()

        return jsonify({
            'status': 'processing',
            'job_id': job_id,
            'user_id': user_id,
            'message': 'Video processing started'
        })

    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        update_job_status(job_id, user_id, 'FAILED', 'video', video_file_name)
        return jsonify({'error': 'An error occurred while processing the video'}), 500

@app.route('/get_job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get the status of a job
    ---
    parameters:
      - name: job_id
        in: path
        type: string
        required: true
        description: ID of the job
      - name: user_id
        in: query
        type: string
        required: true
        description: ID of the user
    responses:
      200:
        description: Successful response
        schema:
          properties:
            PK:
              type: string
            SK:
              type: string
            status:
              type: string
            file_type:
              type: string
            file_name:
              type: string
            updatedAt:
              type: string
            position:
              type: string
            s3_path:
              type: string
      404:
        description: Job not found
      400:
        description: Bad request
      500:
        description: Server error
    """
    try:
        user_id = request.args.get('user_id')
        validate_user_id(user_id)

        response = dynamodb_table.get_item(Key={'PK': f"USER#{user_id}", 'SK': f"JOB#{job_id}"})
        item = response.get('Item')
        
        if not item:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(item)

    except ValueError as ve:
        logger.warning(f"Validation error in get_job_status: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_job_status: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving the job status'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.APP_PORT)