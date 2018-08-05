from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#Function to handle the non-numeric data.
def handle_non_numeric_data(df):
  columns=df.columns.values
 
  for column in columns:
    text_digit_vals={}
    def convet_to_int(val):
      return text_digit_vals[val]
   
    if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
      column_content=df[column].values.tolist()
      unique_element=set(column_content)
      x=0
      for unique in unique_element:
       if unique not in text_digit_vals:
        text_digit_vals[unique]=x
        x+=1
    
      df[column]=list(map(text_digit_vals.get, df[column]))
  return df 


#Filepath of the training and testing dataset.
train_filepath='/home/shashank/NRP/Linear Regression/train.csv'
test_filepath='/home/shashank/NRP/Linear Regression/test.csv'



#Read the csv file
train_data=pd.read_csv(train_filepath)
test_data=pd.read_csv(test_filepath)

y=train_data.SalePrice
X=train_data.drop(['Id','SalePrice'],axis=1)

X=handle_non_numeric_data(X)

#Split tha data for testing and training.
train_X,test_X,train_y,test_y=train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

#Imputer for adding value where data is blank or NaN.
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
X_scaled = preprocessing.scale(train_X)
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_X,train_y)

# Make predictions using the testing set
pred_y = regr.predict(test_X)



print(mean_squared_error(test_y,pred_y))
print(mean_absolute_error(test_y,pred_y))




