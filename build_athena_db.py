
import boto3
from botocore.exceptions import ClientError

def create_athena_database(database_name, s3_output_location):
    client = boto3.client('athena')
    
    create_database_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
    
    try:
        response = client.start_query_execution(
            QueryString=create_database_query,
            ResultConfiguration={
                'OutputLocation': s3_output_location,
            }
        )
        print(f"Database creation initiated. QueryExecutionId: {response['QueryExecutionId']}")
        return response['QueryExecutionId']
    except ClientError as e:
        print(f"Error creating Athena database: {e}")
        return None

def create_athena_table(database_name, table_name, s3_data_location, s3_output_location):
    client = boto3.client('athena')
    
    create_table_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {database_name}.{table_name} (
        job_id STRING,
        user_id STRING,
        player_id STRING,
        position STRING,
        start_time DOUBLE,
        end_time DOUBLE,
        duration DOUBLE,
        video_timestamp DOUBLE
    )
    STORED AS PARQUET
    LOCATION '{s3_data_location}'
    """
    
    try:
        response = client.start_query_execution(
            QueryString=create_table_query,
            QueryExecutionContext={
                'Database': database_name
            },
            ResultConfiguration={
                'OutputLocation': s3_output_location,
            }
        )
        print(f"Table creation initiated. QueryExecutionId: {response['QueryExecutionId']}")
        return response['QueryExecutionId']
    except ClientError as e:
        print(f"Error creating Athena table: {e}")
        return None

# Usage
database_name = 'bjj_analytics'
table_name = 'bjj_positions'
s3_data_location = 's3://bjj-pics/processed_data/'
s3_output_location = 's3://bjj-athena-results'

# Create database
db_query_execution_id = create_athena_database(database_name, s3_output_location)
if db_query_execution_id:
    print(f"Database '{database_name}' creation initiated.")
else:
    print("Failed to initiate database creation.")

# Create table
table_query_execution_id = create_athena_table(database_name, table_name, s3_data_location, s3_output_location)
if table_query_execution_id:
    print(f"Table '{table_name}' creation initiated in database '{database_name}'.")
else:
    print("Failed to initiate table creation.")

print("Script execution completed. Check AWS Athena console for the status of database and table creation.")