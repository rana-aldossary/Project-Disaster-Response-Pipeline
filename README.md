# Project-Disaster-Response-Pipeline

## Project Summary
The Disaster Response Pipeline project is the second project in the Data Scientist Nanodegree Udacity course. This project focus on analyzing and processing messages has been sent during the disaster and build a machine learning pipeline to classify the disaster messages. Also, it  includes a visualization of the data using a web app. Moreover, the user can enter the messages to get the classified messages.





## How to Run The Project
* Instructions:
  1. Run the following commands in the project's root directory to set up your database and model.
     * To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
      * To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  2. Go to `app` directory: `cd app`

  3. Run your web app: `python run.py`

  4. Click the `PREVIEW` button to open the homepage



## Files In The Repository
* app
  * This folder contain the HTML files and 'run.py' file 
* data
  *  This folder contain the data and the database
* models
  * This folder contain machine learning pipeline and the model exported as pickle file
  


## Acknowledgement
The codes in the template folder are blong to the Udacity.
