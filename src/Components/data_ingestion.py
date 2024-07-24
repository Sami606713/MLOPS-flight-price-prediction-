import pandas as pd
from src.utils import read_data,clean_data
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)


def inisiate_data_ingestion():
    """
    In this fun first we will read the data.
    Then we will apply basic preprocessing like handling missing valeues removing duplicates and handling datatypes
    """

    logging.info("Getting the data")
    df=read_data()

    logging.info("Apply data cleaning")

    df=clean_data(df=df)

    return df

# if __name__=="__main__":
#     inisiate_data_ingestion()