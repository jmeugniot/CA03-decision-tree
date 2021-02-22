# CA03-decision-tree
Build a decision tree classifier models, optimize model, and measure model performance

# Description
The dataset is obtained from the Census Bureau and represents salaries of people along with seven demographic variables. The following is a description of our dataset:
•	Number of target classes: 2 ('>50K' and '<=50K') [ Labels: 1, 0 ]
•	Number of attributes (Columns): 7
•	Number of instances (Rows): 48,842

## Instructions:
1. Download Tuning_test2.csv, census_data.csv files to local computer
2. Open ipynb file
3. Click "Open in Colab" button or click the Colab link
4. Upload these two files to goolge drive
5. Run code by lines or select runtime in the menu bar - click run all in the dropdown list

## Usage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

## Questions answered
Q.1.1 Why does it makes sense to discretize columns for this problem?
Q.1.2	What might be the issues (if any) if we DID NOT discretize the columns.
Q.7.1 Decision Tree Hyper-parameter variation vs. performance
Q.8.1	How long was your total run time to train the model? 
Q.8.2	Did you find the BEST TREE? 
Q.8.3	Draw the Graph of the BEST TREE Using GraphViz
Q.8.4	What makes it the best tree?
Q.10.1	What is the probability that your prediction for this person is accurate?

