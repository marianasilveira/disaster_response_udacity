##
### Disaster Response Pipeline Project
###

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation

The scikit-learn and nltk libraries are required. The code should run with no issues using Python versions 3.

## Instructions

Run the following commands in the project's root directory to set up your database and model. The training data 'disaster_messages.csv' and 'disaster_categories.csv' can be found on Udacity's website.

1. To run ETL pipeline that cleans data and stores in database:

	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. To run ML pipeline that trains classifier and saves:

	`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. To run the application: 
	
	`python app/run.py`

## File Descriptions

This application classifies a text message using Random Forests, whose traning data were Tweets related to different natural and non-natural disasters. The practical impacts and benefits of this application consist on helping people and/or organizations to plan a course of action in an event of a disaster, e.g. managing traffic, medical assistence, food distribution and shelter occupation.

This repository is structured as described below. The CSV files are not provided, but they can be downloaded [here](https://learn.udacity.com/nanodegrees/nd025/parts/cd0018/lessons/c5de7207-8fdb-4cd1-b700-b2d7ce292c26/concepts/c6d64c4f-5877-4eab-815b-e1c6495b0201).

	- Root Directory
		-README.md
		- data
			- process_data.py
            		- disaster_categories.csv
			- disaster_messages.csv
			- DisasterResponse.db
		- models
            		- train_classifier.py
            		- classifier.pkl
    		- app
			-templates
				-go.html
				-master.html
			- run.py
 
## Licensing, Authors, Acknowledgements

Must give credit to Figure Eight for the data. Feel free to use the code here as you would like!


