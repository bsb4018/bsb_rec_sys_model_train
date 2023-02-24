import os
from from_root import from_root
from src.constants.cloud_constants import AWS_ACCESS_KEY_ID,AWS_REGION,AWS_SECRET_ACCESS_KEY,S3_FEATURE_REGISTRY_BUCKET_NAME
from src.constants.file_path_constants import FEAST_FEATURE_STORE_REPO_NAME,FEAST_FEATURE_STORE_REPO_PATH
from boto3 import Session
import os

class AwsStorage:
    """
    Get AWS credentials
    """
    def __init__(self):
        self.ACCESS_KEY_ID = os.getenv(AWS_ACCESS_KEY_ID)
        self.SECRET_KEY = os.getenv(AWS_SECRET_ACCESS_KEY)
        self.REGION_NAME = os.getenv(AWS_REGION)
        self.BUCKET_NAME = S3_FEATURE_REGISTRY_BUCKET_NAME

    def get_aws_storage_config(self):
        return self.__dict__

class StorageConnection:
    """
    Created connection with S3 bucket using boto3 api to fetch the model from Repository.
    """
    def __init__(self):
        self.config = AwsStorage()
        self.session = Session(aws_access_key_id=self.config.ACCESS_KEY_ID,
                               aws_secret_access_key=self.config.SECRET_KEY,
                               region_name=self.config.REGION_NAME)
        self.s3 = self.session.resource("s3")
        self.bucket = self.s3.Bucket(self.config.BUCKET_NAME)

    def download_feature_store_registries_s3(self):
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        
        s3_folder = FEAST_FEATURE_STORE_REPO_NAME
        local_dir = FEAST_FEATURE_STORE_REPO_PATH
        bucket = self.bucket
        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)


if __name__ == "__main__":
    connection = StorageConnection()
    connection.download_feature_store_registries_s3()

