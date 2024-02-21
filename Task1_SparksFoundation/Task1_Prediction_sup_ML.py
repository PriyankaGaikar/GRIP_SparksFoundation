#!/usr/bin/env python
# coding: utf-8

# <h1>Author :- Priyanka Madhavrao Gaikar

# <h1>TASK 1 - Prediction using Supervised ML

# -  To Predict the percentage of marks of the students based on the number of hours they studied

# In[1]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[2]:


# Reading the Data 
data = pd.read_csv ('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data.head(10)


# In[3]:


# Check if there any null value in the Dataset
data.isnull == True


# <h4>There is no null value in the Dataset so, we can now visualize our Data.

# In[4]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# <h4>From the above scatter plot there looks to be correlation between the 'Marks Percentage' and 'Hours Studied', Lets plot a regression line to confirm the correlation.

# In[5]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# <h4>It is confirmed that the variables are positively correlated.

# <h3>Training the Model

# <h3>1) Splitting the Data

# In[6]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# <h3>2) Fitting the Data into the model

# In[7]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# <h3>Predicting the Percentage of Marks

# In[8]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# <h3>Comparing the Predicted Marks with the Actual Marks

# In[9]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# <h3>Visually Comparing the Predicted Marks with the Actual Marks

# In[10]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# <h3>Evaluating the Model

# In[11]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# -  Small value of Mean absolute error states that the chances of error or wrong forecasting through the model are very less.

# <h3>What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[12]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# <h4> According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.

# In[ ]:




