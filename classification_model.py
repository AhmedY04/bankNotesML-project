import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv('data.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_val,Y_train,Y_val = train_test_split(x,y, test_size=0.2)

model = LogisticRegression().fit(X_train, Y_train)
prediction = model.predict(X_val)
accuracy = accuracy_score(prediction, Y_val)
precision = precision_score(prediction, Y_val)
recall = recall_score(prediction, Y_val)

print(accuracy)
print(precision)
print(recall)

