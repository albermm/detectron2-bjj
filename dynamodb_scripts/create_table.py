import boto3
from botocore.exceptions import ClientError

# Initialize the DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  # Adjust the region if needed

def create_table():
    try:
        table = dynamodb.create_table(
            TableName='BJJ_App_Table',
            KeySchema=[
                {
                    'AttributeName': 'PK',
                    'KeyType': 'HASH'  # Partition key
                },
                {
                    'AttributeName': 'SK',
                    'KeyType': 'RANGE'  # Sort key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'PK',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'SK',
                    'AttributeType': 'S'
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        
        print("Creating table...")
        table.meta.client.get_waiter('table_exists').wait(TableName='BJJ_App_Table')
        print("Table created successfully")

    except ClientError as e:
        print(f"Error creating table: {e.response['Error']['Message']}")

if __name__ == "__main__":
    create_table()

