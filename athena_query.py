import boto3
import time
import json
from botocore.exceptions import ClientError

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('BJJ_App_Table')  

def start_athena_query(query, database, s3_output, parameters=None):
    client = boto3.client('athena')
    
    try:
        execution_params = {
            'QueryString': query,
            'QueryExecutionContext': {
                'Database': database
            },
            'ResultConfiguration': {
                'OutputLocation': s3_output,
            }
        }
        if parameters:
            execution_params['QueryExecutionContext']['ParameterValues'] = [{'Name': str(i), 'Value': str(val)} for i, val in enumerate(parameters)]
        
        response = client.start_query_execution(**execution_params)
        return response['QueryExecutionId']
    except ClientError as e:
        print(f"Error starting Athena query: {e}")
        return None

def get_query_results(query_execution_id):
    client = boto3.client('athena')
    
    while True:
        try:
            response = client.get_query_execution(QueryExecutionId=query_execution_id)
            state = response['QueryExecution']['Status']['State']
            
            if state == 'SUCCEEDED':
                result = client.get_query_results(QueryExecutionId=query_execution_id)
                return result['ResultSet']['Rows']
            elif state in ['FAILED', 'CANCELLED']:
                return None
            
            time.sleep(5)  # Wait for 5 seconds before checking again
        except ClientError as e:
            print(f"Error getting query results: {e}")
            return None

def get_latest_job_id(user_id):
    try:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('PK').eq(f"USER#{user_id}"),
            ScanIndexForward=False,
            Limit=1
        )
        if response['Items']:
            return response['Items'][0]['SK'].split('#')[1]  # Assuming SK is in format "JOB#jobId"
        else:
            return None
    except ClientError as e:
        print(f"Error querying DynamoDB: {e}")
        return None

def query_position_data(user_id, job_id):
    # Query for the recently processed video
    recent_query = """
    SELECT 
        position,
        COUNT(*) as position_count,
        CAST(SUM(duration) AS DOUBLE) as total_duration
    FROM 
        bjj_positions
    WHERE 
        user_id = ?
        AND job_id = ?
    GROUP BY 
        position
    ORDER BY 
        position_count DESC
    """
    
    # Query for all user's videos
    total_query = """
    SELECT 
        position,
        COUNT(*) as position_count,
        CAST(SUM(duration) AS DOUBLE) as total_duration
    FROM 
        bjj_positions
    WHERE 
        user_id = ?
    GROUP BY 
        position
    ORDER BY 
        position_count DESC
    """
    
    results = {}
    
    for query_type, query in [("recent", recent_query), ("total", total_query)]:
        query_execution_id = start_athena_query(
            query,
            'bjj_analytics',  # Athena database name
            's3://bjj-athena-results/',
            parameters=[user_id, job_id] if query_type == "recent" else [user_id]
        )
        
        if query_execution_id:
            query_results = get_query_results(query_execution_id)
            if query_results:
                # Process the results
                processed_results = []
                for row in query_results[1:]:  # Skip the header row
                    processed_results.append({
                        'position': row['Data'][0]['VarCharValue'],
                        'count': int(row['Data'][1]['VarCharValue']),
                        'total_duration': float(row['Data'][2]['VarCharValue'])
                    })
                results[query_type] = processed_results
    
    return results

# Lambda handler
def lambda_handler(event, context):
    try:
        # Parse the user_id from the event (assuming it's passed in the path parameters)
        user_id = event['pathParameters']['userId']
        
        # Get the latest job_id for this user
        job_id = get_latest_job_id(user_id)
        
        if not job_id:
            return {
                'statusCode': 404,
                'body': json.dumps('No jobs found for this user')
            }
        
        results = query_position_data(user_id, job_id)
        
        if results:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'position_data': results
                })
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps('Failed to retrieve position data')
            }
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {str(e)}")
        }