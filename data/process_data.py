# python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine    

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv files and merge to a single dataframe
    Input:
        - messages_filepath - string - path to messages csv file
        - categories_filepath - string - path to categories csv file
    Returns:
        - df - dataframe - merged dataframe

    '''
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    # Merge datasets
    df = messages.merge(categories, how='outer', on='id') 
    return df

def clean_data(df):
    '''
    Clean dataframe, reshaping categories into columns
    Input:
        - df - original dataframe
    Returns:
        - df - dataframe - cleaned dataframe

    '''
    # Create a dataframe of the 36 individual category columns
    categories = df.iloc[:,-1].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[1]
    
    # Extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # Rename the columns of 'categories'
    categories.columns = category_colnames
    
    # Convert category from string to numeric
    for column in categories.columns:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column and
    # concatenate the original dataframe with the new dataframe
    df = pd.concat([df.drop(labels=['categories'],axis=1),categories],axis=1)
    
    # Drop lines whose values are different from 0 or 1
    # 0.78 % of category 'related' is 2
    for column in categories.columns:
        drop_index = (df[column]!=0) & (df[column]!=1)
        df.drop(df.index[drop_index], inplace = True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Reset index
    df.reset_index(drop=True)
    
    return df

def save_data(df, database_filepath):
    '''
    Save data in SQLite database
    Input:
        - df - dataframe to be saved
        - database_filepath - string - path to database
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql(database_filepath, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()