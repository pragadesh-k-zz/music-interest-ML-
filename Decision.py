import pandas as pd # importing the packages
from sklearn.tree import DecisionTreeClassifier # it is the ML algorithm

file = pd.read_csv('Bookone.csv') # open file.csv
input_data = file.drop(columns = ['like']) # Splitied input data 
y = file['like'] # Splitied output data

model = DecisionTreeClassifier() # algorithm
model.fit(input_data,y) # training the algorithm by given the i/o and o/p
prediction = model.predict([[3,1],[9,0]]) # asking machine to give the answer for our ques
print(prediction) # answer