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
* click_data.csv - Data set
* train.py - Training the final model and saving to pickle file.
* predict.py - Loading the model and serving it via a Flask application
* predict_test.py - Testing the model

### Steps to run the application

1. Build a docker image using the below command
````
docker build -t midterm-project .
````
2. Run the docker image

````
docker run -it -p 9696:9696 midterm-project
````

3. Test the application

````
python predict_test.py
````

### Cloud Deployment

Application was deployed to AWS ElasticBeanStalk. Please refer to the below screenshots in the repo:

* EBSDeployment_Aws_Screenshot.png
* ElasticBeanStalkDeployment_Screenshot.png
