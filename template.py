"""
- In template.py we can automate the project structure.
- If user can run this file all the project structure will be automatically created.
"""

import os
import logging
logging.basicConfig(level=logging.INFO)

# Make the directories of needed
folders=[
    os.path.join("Data","Raw"),
    os.path.join("Data","Process"),

    os.path.join("src","Components"),
    os.path.join("src","Pipelines"),

    "Models",
    "Notebooks"
]

for fol in folders:
    if not os.path.exists(fol):
        os.makedirs(fol)

        # make a git keep file
        git_keep=os.path.join(fol,".gitkeep")
        with open(git_keep,"w") as f:
            pass 

    else:
        print(f"{fol} already present")

# Now next step is to place the file inside the folder

files=[
    os.path.join("src","__init__.py"),
    os.path.join("src","utils.py"),

    os.path.join("src/Components","__init__.py"),
    os.path.join("src/Components","data_ingestion.py"),
    os.path.join("src/Components","data_transformation.py"),
    os.path.join("src/Components","model_training.py"),
    
    os.path.join("src/Pipelines","__init__.py"),
    os.path.join("src/Pipelines","Prediction_pipeline.py"),
    
    "app.py",
    "test_environment.py",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "requirements.txt"  
]

for file in files:
    if not os.path.exists(file) or (os.path.getsize(file)==0):
        with open(file,"w"):
            pass
    else:
        print(f"File {file} already exists")