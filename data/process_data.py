import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Create one dataframe from the two csv files at the paths specified.

    Input:
    Two filepaths for the messages and categories files.

    Output:
    A dataframe with both files merged.
    """

    #opening the messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merging into one dataframe
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Cleans the dataframe so that categories are separated and binary classes.

    Input:
    The dataframe with raw data.

    Output:
    A cleaned dataframe with binary classes for the categories.
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)

    # extracting category names from the first row
    row = categories.iloc[:1]
    category_colnames = [x[:-2] for x in row.values[0]]

    # renaming the categories columns
    categories.columns = category_colnames

    # cleaning the values in the categories columns
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)

    # replacing the categories column in df
    df.drop('categories',axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # dropping columns useless for the ML model
    df.drop(['original', 'child_alone'], axis=1, inplace=True)

    # fixing erronous values in the related category
    df['related'].replace(2, 1, inplace=True)

    # dropping duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filepath):
    """
    Saves a dataframe into a SQL database.

    Input:
    The dataframe and the database filepath.

    Output:
    None
    """

    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('messageslabeled', engine, if_exists='replace', index=False)


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
