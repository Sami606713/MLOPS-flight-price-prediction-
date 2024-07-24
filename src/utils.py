import os
import pandas as pd
import numpy as np
import logging 

logging.basicConfig(level=logging.INFO)

# Make a fun for reading data
def read_data():
    """
    This fun is responsible for reading the data form any source.
    
    It can read the data and return the data as a dataframe.
    """

    # read the data
    logging.info("Reading the data from source")
    df=pd.read_csv("Given_data/flight_price.csv")

    logging.info("Successfully read the data")

    return df


def clean_data(df):
    """
    This function is responsible for cleaning the data.
    Cleaning data like correct data types handling missing values drop duplicates etc.
    After Analysis we can see that there is very less nbr of missing values so we can drop missing values.
    We can also see that data contain duplicates so we can drop the duplicates.
    """

    # Remove missing values
    logging.info("Removing Null Values")
    df.dropna(inplace=True)

    # Remove Duplicates
    logging.info("Removing Duplicates Values")
    df.drop_duplicates(inplace=True)

    # Change the data type of data of journey b/c it is in object form
    df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y')

    logging.info("Data Cleaning Successfully")

    return df
