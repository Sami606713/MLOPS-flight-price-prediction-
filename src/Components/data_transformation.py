from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_file
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

"""
- In this file i can do all the feature transformation
- Feature transformation like impute values, encoding scaling etc.
"""

class data_transformation_config:
    def __init__(self,train_data_path,test_data_path):
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
        self.processor_path=os.path.join("Models","processor.pkl")
    
    def inisiate_data_transformation(self):
        logging.info("Reading the train and test data")
        train_data=pd.read_csv(self.train_data_path)

        test_data=pd.read_csv(self.test_data_path)
        logging.info("Reading train and test data successfully")
        
        # saperate feature and labels
        logging.info("saperate the feature and labels")
        x_train=train_data.drop(columns=['Price'])
        y_train=train_data['Price'].values.reshape(-1,1)

        x_test=test_data.drop(columns=['Price'])
        y_test=test_data['Price'].values.reshape(-1,1)

        # saperate num and categorical columns
        num_col=x_train.select_dtypes("number").columns
        cat_col=x_train.select_dtypes("object").columns

        # call the build pipeline fun
        processor=self.build_pipeline(cat_col=cat_col,num_col=num_col)
        
        # apply the processor on train and test data
        logging.info("applying the transformation on train and test data")
        x_train_transform=processor.fit_transform(x_train)
        x_test_transform=processor.transform(x_test)
        logging.info("Transformation done")

        # concate the transfrom data with output columns
        logging.info("Concate the train and test array")
        train_array=np.c_[x_train_transform,y_train]
        test_array=np.c_[x_test_transform,y_test]
        logging.info("concatination done")

        logging.info("save the processor")
        save_file(obj=processor,file_path=self.processor_path)

        return [
            train_array,
            test_array
        ]


    def build_pipeline(self,num_col,cat_col):
        # Build numerical pipeline
        logging.info("Build numerical pipeline")
        num_pipe=Pipeline(steps=[
            ("impute",SimpleImputer(strategy='median')),
            ("scale",StandardScaler())
        ])

        # build a categorical pipeline
        logging.info("Build a categorical pipeline")
        cat_pipe=Pipeline(steps=[
            ("impute",SimpleImputer(strategy="most_frequent")),
            ("encode",OneHotEncoder(drop="first",sparse_output=False,
                                    handle_unknown='ignore'))
        ])

        # Build a columns transformer
        processor=ColumnTransformer(transformers=[
            ("Num_transform",num_pipe,num_col),
            ("Cat_transform",cat_pipe,cat_col)
        ])


        return processor

