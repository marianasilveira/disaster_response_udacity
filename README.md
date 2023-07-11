##
### Disaster Response Pipeline Project
###

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation

The scikit-learn and nltk libraries are required. The code should run with no issues using Python versions 3.

### Instructions:
Run the following commands in the project's root directory to set up your database and model. The training data 'disaster_messages.csv' and 'disaster_categories.csv' can be found on Udacity's website.

1. To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. To run the application: 
        `python app/run.py`


## File Descriptions
An application available here. It classifies a message in accordance to the proposed model, which was trained using Tweets about disaster responses.

## Licensing, Authors, Acknowledgements

Must give credit to Figure Eight for the data. Feel free to use the code here as you would like!


