## MidTerm Project: Click Prediction
The data is about advertisements shown alongside search results in a search engine and whether or not people clicked on these ads.
The task is to build the best possible model to predict whether a user will click on a given ad.

A search session contains information on user id, the query issued by the user, ads displayed to the user and the 
target feature indicating whether a user clicked at least one of the ads in this session. 
The number of ads displayed to a user in a session is called ‘depth’. 
The order of an ad in the displayed list is called ‘position’. 
An ad is displayed as a short text called ‘title’, followed by a slightly longer text called ’description’ 
and a URL called ‘display URL’.

### Dataset Features
The dataset has the following features:
* Click – binary variable indicating whether a user clicked on at least one ad.
* Impression - the number of search sessions in which AdID was impressed by UserID who issued Query.
* Url_hash - URL is hashed for anonymity
* AdID - ID for the advertisement
* AdvertiserID - ID of the advertiser
* Depth - number of ads displayed to a user in a session
* Position - order of an ad in the displayed list
* QueryID - This is an ID field referred from another data file.
* KeywordID - This is an ID field referring to a keyword ID referring to another data file.
* TitleID - ID for the title. The titles are again described in another data file.
* DescriptionID - ID for the description. Description is described in another data file.
* UserID – ID for the user. Other details of the User is defined in another data file.

Note: As we can see from the features list all the columns are IDs.

### Project File Details

* click-prediction.ipynb - Jupyter notebook created for experimenting with various models
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

Application was deployed to AWS ElasticBeanStalk. Screenshot for the same has been attached.