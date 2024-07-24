from setuptools import find_packages,setup

def get_packages(path):
    """
        - This fun is responsible for reading the requirement.txt file and get all the packages.
        - After reading it will return all the packages name as a list
    """
    with open(path,"r") as f:
        requirement= f.readlines()
    
    requirement=[req.strip() for req in requirement]
    return requirement

setup(
    author="Samiullah",
    author_email="sami606713@gmail.com",
    name="Flight Pricre Prediciton",
    description="In this project our goal is to predict the price of flight",
    packages=find_packages(),
    install_requires=get_packages('requirements.txt')
)
