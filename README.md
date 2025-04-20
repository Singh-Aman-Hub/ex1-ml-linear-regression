# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AMAN SINGH
RegisterNumber:  212224040020

```
```
#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.linear_model import LinearRegression
```
```
#displaying few records of the datasets
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
```
```
#separating the independent(X) and dependent(Y) variable
x=df.iloc[:,:-1].values  #study hours
y=df.iloc[:,1].values   #scores

# spliting the dataset into training and testing set(2/3 for training purpose)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=1/3,random_state=0)


#Training the model
reg= LinearRegression()
reg.fit(x_train,y_train)


y_pred= reg.predict(x_test)

print("Predicted values: ",y_pred.round(2))
print("Actual values: ",y_test)

```
```
#Plotting the Graphs
plt.scatter(x_train,y_train,color="red",label="Actual Scores",marker="*")
plt.plot(x_train,reg.predict(x_train),color="black",label="Best Fitted Line")
plt.title("Hourses vs Scores - Training data")
plt.xlabel("Hours Studied per day")
plt.ylabel("Marks Scored")
plt.legend()
plt.show()

plt.scatter(x_test,y_test,marker="o",label="Actual Scores",color="purple")
plt.plot(x_test,reg.predict(x_test),label="Best Fitter Line",color="red")
plt.title("Hours vs Scores - testing data set")
plt.xlabel("Hour studied per day")
plt.ylabel("Marks scored")
plt.legend()
plt.show()


```

```
#calculating the error metrics

mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)

print("Mean Sqaure error (mse) = ",mse)
print("Mean Absolute error (mae) = ",mae)
print("Root Mean Sqaure error (mse) = ",rmse)
```

## Output:



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
