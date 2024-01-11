Executing "Analyzing DDoS Attacks using Machine Learning Techniques"


Step 1 - Download the dataset from "https://www.unb.ca/cic/datasets/ddos-2019.html". The dataset contains a list of csv files.

Step 2 - Open python notebook or any python IDE. 

Step 3 - Install all the necessary libraries and packages that are required to execute our project. They are 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

Step 3 - Using pandas library, load the various datasets using filepath. Note- The filepath can be different in your computer, you can just use the filename with ".csv" extension.

Step 4 - Concatenate all the loaded datasets into a signle dataset.

Step 5 - After loading the dataset, it needs to be cleaned so perform some pre process data cleaning techniques to remove unwanted data. It includes removing "nan" values, removing non-integer values and any other values that are inconsistent with our  required data.

Step 6 - Split the data into training and testing data. The trainin g data is used to train the model and testing data is used to test the trained model.

Step 7 - We need to select best features needed to be trained so that the model can have the best accuracy while predicting in the fututre. We take 5 best features to train on those features.

Step 8 - Find the correlation between the best selected features.

Step 9 - Using Random Forest Classifier, train the datasets and test it. After testing it we calculate the accuracy of the prediction. 

Step 10 - We follow the same method using Decision Tree Classifier and measure the accuracy of that classifier as well. 

Step 11 - The highest accuracy will determine which classifier is best for detecting DDoS attacks. In our case it was Random Forest Classifier.
