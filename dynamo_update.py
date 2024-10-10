import boto3
import time
from datetime import datetime, timedelta

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('BJJ_App_Table')  # Replace with your actual table name

def update_item(pk, sk):
    current_time = int(time.time())
    one_year_from_now = current_time + 31536000  # 365 days in seconds
    submission_date = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d')

    try:
        response = table.update_item(
            Key={
                'PK': pk,
                'SK': sk
            },
            UpdateExpression="SET createdAt = if_not_exists(createdAt, :current_time), "
                             "updatedAt = :current_time, "
                             "submission_date = if_not_exists(submission_date, :submission_date), "
                             "processing_start_time = if_not_exists(processing_start_time, :current_time), "
                             "processing_end_time = if_not_exists(processing_end_time, :current_time), "
                             "expiry_date = :expiry_date, "
                             "#status = if_not_exists(#status, :pending_status), "
                             "file_type = if_not_exists(file_type, :unknown_type)",
            ExpressionAttributeValues={
                ':current_time': current_time,
                ':submission_date': submission_date,
                ':expiry_date': one_year_from_now,
                ':pending_status': 'PENDING',
                ':unknown_type': 'unknown'
            },
            ExpressionAttributeNames={
                '#status': 'status'  # 'status' is a reserved word in DynamoDB
            },
            ReturnValues="UPDATED_NEW"
        )
        print(f"Successfully updated item: {pk}, {sk}")
        return response
    except Exception as e:
        print(f"Error updating item {pk}, {sk}: {str(e)}")
        return None

def convert_timestamp(timestamp_str):
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except ValueError:
        return int(timestamp_str)

# Scan the table to get all items
scan_response = table.scan()
items = scan_response['Items']

# Update each item
for item in items:
    pk = item['PK']
    sk = item['SK']
    update_item(pk, sk)

    # Convert existing timestamp fields
    for field in ['createdAt', 'updatedAt', 'timestamp']:
        if field in item and isinstance(item[field], str):
            unix_timestamp = convert_timestamp(item[field])
            table.update_item(
                Key={'PK': pk, 'SK': sk},
                UpdateExpression=f"SET {field} = :val",
                ExpressionAttributeValues={':val': unix_timestamp}
            )

    # Ensure consistent naming
    if 'keypoint_image_url' in item:
        table.update_item(
            Key={'PK': pk, 'SK': sk},
            UpdateExpression="SET image_url = :val REMOVE keypoint_image_url",
            ExpressionAttributeValues={':val': item['keypoint_image_url']}
        )

# Handle pagination if there are more items
while 'LastEvaluatedKey' in scan_response:
    scan_response = table.scan(ExclusiveStartKey=scan_response['LastEvaluatedKey'])
    items = scan_response['Items']
    for item in items:
        update_item(item['PK'], item['SK'])

print("Table update complete.")