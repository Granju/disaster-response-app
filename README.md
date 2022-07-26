# Disaster Response Pipeline Project

### About this project:

This project consists in training an algorithm on some labelled messages (gathered from various sources) all relating to some disaster situations (ex: earthquakes, fires, floods, power outages etc) and build a web app where a user can enter a message to be labelled by the algorithm. There are 36 categories and each message can have multiple labels so this is a multioutput classifier model.

The messages used to trained the algorithm have been provided by Figure Eight (now Appen). First the data is cleaned and saved in a SQLite database to be used for the model training. A NLP pipeline is then used to train a multi output classifier (based on an Adaboost classifier). The model is then optimised using CVgridsearch and saved in a pickle file. The model is then loaded in a Flask app to allow for classification of user inputed messages.

### Installing the project:

1. Clone the GitHub repository: https://github.com/Granju/disaster-response-app.git

2. Install all the requirements listed in requirements.txt.

3. Install the following Pypi package: https://pypi.org/project/jf-tokenize-package/ which is used to load a tokenizer function required to run train_classifier.py and app.py.

### Running the app locally:

1. Run the ETL pipeline and save the database by running:
  python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

2. Run the ML pipeline that trains the classifier and creates the pickled model by running:
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Run app.py to launch the app.

The app is hosted on Heroku and can be accessed [here] (https://jf-disaster-response-app.herokuapp.com/).

### Project structure:

  .
  ├── requirements.txt
  ├── nltk.txt
  ├── Procfile
  ├── README.md
  ├── app.py
  ├── data
  │   ├── DisasterResponse.db
  │   ├── categories.csv
  │   ├── messages.csv
  │   └── process_data.py
  ├── model
  │   ├── classifier.pkl
  │   └── train_classifier.py
  └── templates
      ├── go.html
      └── master.html

The files categories.csv and messages.csv contain the data provided by Figure Eight with the messages and their labels.
