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

1. To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. To run the application: 
        `python app/run.py`


## File Descriptions
The application available here classifies a message in accordance to the proposed model, which was trained using Tweets related to different disasters. 

	- ./data contains the etl pipeline code, and the CSV and DB files containing the data. Due to their sizes, the CSV files are not provided in this repository, but they can be downloaded [here](https://learn.udacity.com/nanodegrees/nd025/parts/cd0018/lessons/c5de7207-8fdb-4cd1-b700-b2d7ce292c26/concepts/c6d64c4f-5877-4eab-815b-e1c6495b0201).

	- ./models contains the ml pipeline and the proposed model's pickle file.

	- ./app contains the backend and frontend codes for the web application.

 
.## Licensing, Authors, Acknowledgements

Must give credit to Figure Eight for the data. Feel free to use the code here as you would like!


