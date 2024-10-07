import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from .shared_utils import logger, Config, s3_client

def update_position_in_parquet(user_id, job_id, position_id, new_name):
    try:
        # Construct the S3 path
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        s3_path = f'processed_data/user_id={user_id}/date={current_date}/{job_id}.parquet'

        # Download the parquet file from S3
        local_path = f'/tmp/{job_id}.parquet'
        s3_client.download_file(Config.BUCKET_NAME, s3_path, local_path)

        # Read the parquet file
        table = pq.read_table(local_path)
        df = table.to_pandas()

        # Update the position
        df.loc[df['id'] == position_id, 'position'] = new_name

        # Write the updated dataframe back to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, local_path)

        # Upload the updated file back to S3
        s3_client.upload_file(local_path, Config.BUCKET_NAME, s3_path)

        logger.info(f"Successfully updated position for job {job_id}")
        return True

    except Exception as e:
        logger.error(f"Error updating position in parquet: {str(e)}")
        return False