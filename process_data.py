import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # loads the data from a local csv files and merges them into a single dataframe
    # returns a DataFrame

    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = pd.merge(messages_df, categories_df, how='inner', on='id')
    return df


def clean_data(df):
    # cleans the loaded data by splitting compound columns into single columns
    # drops duplicates from the datasets
    # accepts DataFrame as input
    # returns cleaned DataFrame

    categories = df.categories.str.split(';',expand=True)

    row = categories.head(1)

    category_colnames = row.apply(lambda x: x.str[:-2],axis=1).values
    category_colnames = category_colnames[0]
    categories.columns = category_colnames

    df[category_colnames] = df.categories.apply(lambda x: pd.Series(str(x).split(";")))

    for column in category_colnames:
        # set each value to be the last character of the string
        df[column] = df[column].str[-1:]

        # convert column from string to numeric
        df[column] = df[column].astype(int)

    df.drop(['categories'],axis=1,inplace=True)

    df.related.replace(2,1,inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    # saves the DataFrame into a SQLlite table
    # accepts DataFrame and filename as inputs
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')
    pass


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
