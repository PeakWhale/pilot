import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv("student-mat.csv",sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]
predict = "G3"
x = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])
print(data.head())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42)
linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
acc=linear.score(x_test,y_test)
print('R-squared on test set:', acc)

print('Coefficient: \n',linear.coef_)
print ('Intercept: \n',linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

