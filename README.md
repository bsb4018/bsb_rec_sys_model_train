# Course Recommender System - Model Training and Evaluation

### Problem Statement
Course Recommender System for recommending courses to users in a education-website platform
Users Get Recommended in three ways
1. By Interest Tag -> Users enters interest topic tags and get recommendations based on that
2. By Similar Users -> Existing Users enter user id and get recommendations based on similar interactions of other users with the platform 
3. By User's Previous Experiences -> New Users who have not interacted with the platform get recommendations based on past experiences during onboarding

### Solution Proposed 
For 1. We use MongoDB to store our courses topic tagwise and recommend random courses based on the interests
For 2 and 3. We train a hybrid recommender System on user-course interaction features and user features using LightFM framework

## Tech Stack Used
1. Python 
2. Feast Feature Store
3. AWS
4. Docker
5. MongoDB
6. Apache Airflow
7. Graphana and Prometheus


## Infrastructure Required.
1. AWS S3
2. AWS Redshift
3. AWS Glue
4. AWS Dynamo DB
5. Git Actions


## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for some data storage. You also need AWS account to access S3, Redshift, Glue, DynamoDB Services. You also need to have terraform installed and configured. Also need to have installed Apache Airflow.


## Project Architecture
![image]()


### Step 1: Clone the repository
```bash
git clone https://github.com/bsb4018/bsb_rec_sys_mti.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p venv python=3.8 -y
```

```bash
conda activate venv/
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Get AWS credentials
```bash
Get the Feature Repo Bucket Name from Data Store https://github.com/bsb4018/bsb_rec_sys_data_store.git 
Goto src/constants/cloud_constants.py and replace the name S3_FEATURE_REGISTRY_BUCKET_NAME accordingly

Create a S3 bucket to export the model and artifacts
Goto src/constants/cloud_constants.py and replace the name S3_TRAINING_BUCKET_NAME accordingly

Get a note of the following
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION_NAME
```


### Step 5 - Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_REGION_NAME=<AWS_REGION_NAME>

```


### Step 5 - Run locally
```bash
python main.py
```


## Runing Through Docker

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_REGION_NAME=<AWS_REGION_NAME> --build-arg MONGODB_URL=<MONGODB_URL> . 

```

3. Run the Docker image
```
docker run -d -p 8070:8070 <IMAGE_NAME>
```
