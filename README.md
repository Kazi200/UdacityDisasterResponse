# UdacityDisasterResponse
Udacity - Data Scientist Nanodegree - Project 2

The purpose of the project is to be able to categroise messages received in a disaster scenario so that disaster relief agencies are able to respond appropriately.  

The project itself has 3 components:
1. The ETL Pipeline component, which serves to load the data, clean it and store it.
2. The ML Pipeline component, which serves to classify the messages by using machine learning techniques.
3. The Web App component, which surfaces the process on the web to be visualised and used to classify individual messages.  

The files contained in this repo are:
1. process_data.py (ETL process)
2. train_classifier.py (ML process)
3. run.py (Web app)

Instructions to run the web app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
