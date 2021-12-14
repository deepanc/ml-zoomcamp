## Capstone Project: Multi-class Image Classification
The dataset chosen for the Project is an apparel dataset from Kaggle (https://www.kaggle.com/kaiska/apparel-dataset).
The dataset contains 8 different clothing categories in 9 different colors. For example, dataset includes categories like 
Black Dress, Black Pants, Blue Dress, Blue Pants etc.

The aim of the project is to identify the class which the item belongs to based on the image input.
This can be extended to correctly identify an item from a Product Catalogue and further come up with showing similar items
to the user in a web application.

### Dataset
Train and Validation datasets were not readily available here from kaggle. Hence I have downloaded the entire dataset and 
split them to train and validation using the split function of ImageDataGenerator.

### Project File Details

* notebook.ipynb - Jupyter notebook created for experimention
* test_and_convert.ipynb - Extracted the final model and converted to TFLite
* lambda_function.py - Lambda function file created for deployment to AWS
* test.py - Test file
* xception-clothing-model.tflite - Final model converted to TFLite
* dataset/ - This folder contains sample files for testing the model
* cloud_deployment/ - This folder contains screenshots of cloud deployment.


### Steps to run the application locally

1. Download the code from github repo.

2. Dataset can be downloaded from Kaggle (https://www.kaggle.com/kaiska/apparel-dataset) and 
needs to be extracted to the dataset folder for the notebook to work.

3. From the root folder build the docker image using the following command:
````
docker build -t capstone-project-v1 .
````
4. Run the docker image using the below command:
````
docker run -it -p 8080:8080 capstone-project-v1:latest
````
5. For testing the application locally, uncomment the url line in test.py and execute the test file.

URL for local testing in test.py - http://localhost:8080/2015-03-31/functions/function/invocations
````
python3 test.py
````
Currently 3 sample images are provided in the image for local testing (test_black.jpg, test_red.jpg and test_yellow.jpg)
These images can be used for testing in test.py

### Cloud Deployment

Docker image created was pushed to the ECR repo and a lambda function was created to test the same. 
Later an API was exposed for the lambda function using API Gateway.
Please refer to the screenshots in the repo for the same:

* cloud_deployment/lambda*.png - screenshots related to lambda function and testing
* cloud_deployment/API_Gateway*.png - screenshots related to API gateway deployments
* cloud_deployment/test*.png - screenshots related to executing test.py using the deployed API Gateway URL
