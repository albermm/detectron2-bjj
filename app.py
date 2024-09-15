from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from shared_utils import (
    s3_client, dynamodb_table, BUCKET_NAME, generate_job_id,
    update_job_status, get_s3_url, validate_file_type, validate_user_id, logger
)
from helper import Predictor, process_video_async
from threading import Thread

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/get_upload_url', methods=['GET'])
def get_upload_url():
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
            Bucket=BUCKET_NAME,
            Key=file_name,
            Fields={
                'x-amz-meta-job-id': job_id,
                'x-amz-meta-user-id': user_id,
                'Content-Type': content_type,
            },
            Conditions=[
                {'x-amz-meta-job-id': job_id},
                {'x-amz-meta-user-id': user_id},
                ['content-length-range', 1, 100 * 1024 * 1024],
                {'Content-Type': content_type},
            ],
            ExpiresIn=3600
        )

        update_job_status(job_id, user_id, 'PENDING', file_type, file_name)

        return jsonify({
            'presigned_post': presigned_url,
            'file_name': file_name,
            'job_id': job_id,
            'user_id': user_id
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_upload_url: {str(e)}")
        return jsonify({'error': 'An error occurred while generating upload URL'}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        file_name = data['file_name']
        job_id = data['job_id']
        user_id = data['user_id']

        local_file_name = '/tmp/image.jpg'
        s3_client.download_file(BUCKET_NAME, file_name, local_file_name)

        predictor = Predictor()
        output_path = '/tmp/output'
        keypoint_frame, keypoints, predicted_position = predictor.onImage(local_file_name, output_path)

        if keypoints is None:
            raise ValueError('Failed to process image')

        keypoint_image_key = f"outputs/keypoint_frame_{file_name}"
        keypoints_json_key = f"outputs/keypoints_{file_name}.json"

        s3_client.upload_file(f'{output_path}_keypoints.jpg', BUCKET_NAME, keypoint_image_key)
        s3_client.upload_file(f'{output_path}_keypoints.json', BUCKET_NAME, keypoints_json_key)

        keypoint_image_url = get_s3_url(keypoint_image_key)
        keypoints_json_url = get_s3_url(keypoints_json_key)

        update_job_status(job_id, user_id, 'SUCCESS', 'image', file_name, predicted_position, keypoint_image_url)

        return jsonify({
            'status': 'success',
            'keypoint_image_url': keypoint_image_url,
            'keypoints_json_url': keypoints_json_url,
            'predicted_position': predicted_position,
            'job_id': job_id
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        data = request.json
        video_file_name = data['video_file_name']
        job_id = data['job_id']
        user_id = data['user_id']

        update_job_status(job_id, user_id, 'PROCESSING', 'video', video_file_name)

        local_video_path = '/tmp/video.mp4'
        s3_client.download_file(BUCKET_NAME, video_file_name, local_video_path)

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
    try:
        user_id = request.args.get('user_id')
        validate_user_id(user_id)

        response = dynamodb_table.get_item(Key={'PK': f"USER#{user_id}", 'SK': f"JOB#{job_id}"})
        item = response.get('Item')
        
        if not item:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(item)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_job_status: {str(e)}")
        return jsonify({'error': 'An error occurred while retrieving the job status'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)