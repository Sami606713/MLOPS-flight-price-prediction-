# Flight Price Prediction
- In this project i will build a flight price prediction model that can predict the price of flight based on its factor.
- I can approcah this project using `MLOPs` principal.
- We can also use all the `open source` as i needed in this project.

# Project Folder Structure
This repository includes a script, `template.py`, which automates the creation of the project structure. Running this script will automatically create the necessary folders and files for your project.


| Folder/File                           | Description                        |
|---------------------------------------|------------------------------------|
| `Data/Raw/`                           | Folder for raw data                 |
| `Data/Process/`                       | Folder for processed data           |
| `src/Components/`                    | Source code for components          |
| `src/Components/__init__.py`          | Initialization for components       |
| `src/Components/data_ingestion.py`    | Data ingestion script               |
| `src/Components/data_transformation.py` | Data transformation script         |
| `src/Components/model_training.py`    | Model training script               |
| `src/Pipelines/`                     | Source code for pipelines           |
| `src/Pipelines/__init__.py`           | Initialization for pipelines        |
| `src/Pipelines/Prediction_pipeline.py` | Prediction pipeline script         |
| `Models/`                            | Folder for models                   |
| `Notebooks/`                         | Folder for Jupyter notebooks        |
| `app.py`                             | Main application script             |
| `test_environment.py`                | Script to test the environment      |
| `Dockerfile`                         | Dockerfile for containerization     |
| `.dockerignore`                      | Docker ignore file                  |
| `setup.py`                           | Setup script for packaging          |
| `requirements.txt`                   | Python package requirements         |
| `src/__init__.py`                    | Initialization for the src package  |
| `src/utils.py`                       | Utility functions                   |

# Data Ingestion
- In this step i can read the data form any source so in this case i have store the data in my local folder
- But generally in this step we can get all the data form all the sources and combine them into single data.
- In this step we can also do basic data cleaning like handling missing values, removing duplicates and handling data types
![Data Ingestion](Images/data-ingestion-diagram.webp)

