import pandas as pd
import numpy as np
import streamlit as st
from src.Pipelines.Prediction_pipeline import predict

# Page config setting
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("# 💰 Flight Price Prediction ✈️")

# get the user input
df=pd.read_csv('Data/Process/train.csv')

col1,col2,col3=st.columns(3)
with col1:
    Airline=st.selectbox("Airline",df['Airline'].value_counts().index,key="airline")
with col2:
    Source=st.selectbox("Source",df['Source'].value_counts().index,key='source')
with col3:
    Destination=st.selectbox("Destination",df['Destination'].value_counts().index,key='destination')


col1,col2=st.columns(2)
with col1:
    Total_Stops=st.selectbox("Total_Stops",df['Total_Stops'].value_counts().index,key="stops")
with col2:
    Duration_In_Min=st.number_input("Total duration_time in min",min_value=20,key='dur_min')

col1,col2=st.columns(2)
with col1:
    Dep_hour=st.number_input("Dep_hour",min_value=1,max_value=12,key="Dep_hour")
with col2:
    Dep_min=st.number_input("Dep_min",min_value=1,max_value=60,key='dep_min')

col1,col2=st.columns(2)
with col1:
    Arrival_hour=st.number_input("Arrival_hour",min_value=1,max_value=12,key="Arrival_hour")
with col2:
    Arrival_min=st.number_input("Arrival_min",min_value=1,max_value=60,key='Arrival_min')

col1,col2=st.columns(2)
with col1:
    Journey_day=st.selectbox("Journey_day",df['Journey_day'].value_counts().index,key='j_d')
with col2:
     Journey_month=st.selectbox("Journey_month",df['Journey_month'].value_counts().index,key="j_m")


if st.button('Predict Price'):
    # Convert the data into dict
    dic={
        "Airline":Airline,"Source":Source,"Destination":Destination,
        "Total_Stops":Total_Stops,"Duration_In_Min":Duration_In_Min,
        "Dep_hour":Dep_hour,"Dep_min":Dep_min,
        "Arrival_hour":Arrival_hour,"Arrival_min":Arrival_min,
        "Journey_month":Journey_month,"Journey_day":Journey_day
    }
    df=pd.DataFrame(dic,index=[0])
    st.dataframe(df)

    # Pass the data in predict fun
    price=np.round(predict(data=df),3)

    st.success(f"Price of flight is {price}")