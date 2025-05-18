# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VARSHA SARATHY
RegisterNumber:  212223040233
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
![443931901-6f9b3d22-21e4-4ea5-afe2-494538d0cd80](https://github.com/user-attachments/assets/391d20dd-35eb-496f-ae9f-955fb06a4a2c)

![443931936-0889158a-6a7c-42c2-922b-6db0b6738678](https://github.com/user-attachments/assets/f9986560-6b08-4dd1-9f8a-afb9bdea3acc)

![443931969-8a3cc5e7-e9b7-4f2f-a3f2-fa341493b3e9](https://github.com/user-attachments/assets/c867d3ae-602b-4ed6-9611-6af804adbf48)

![443932002-dfeeeee0-b0c6-4f5a-ae01-dcbdd366f833](https://github.com/user-attachments/assets/1faf129a-a34e-4972-9063-53805dbb1ff6)

![443932021-05883084-2bee-45e0-8f53-489116c54499](https://github.com/user-attachments/assets/ffaa37bf-f63d-44db-8771-6f8d4e6b4827)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
