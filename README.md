# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the required library and read the dataframe.
3. Write a function computeCost to generate the cost function.
4. Perform iterations og gradient steps with learning rate.
5. Plot the Cost function using Gradient Descent and generate the required graph. 
6. Stop the program 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRAJAN P
RegisterNumber: 212223240121 
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]

    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_scaled=scaler.fit_transform(X1)
y1_scaled=scaler.fit_transform(y)
print(X)
print(X1_scaled)

theta=linear_regression(X1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction .reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![309775345-8dafef2a-bfee-4144-aa0a-1a553f93ab5e](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/81a3aaa9-d3d3-4042-bb98-6398ea8e2469)

![309775391-edf19e3b-d4d0-42d5-be58-c2d578a73afd](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/869337a6-98e0-43ad-ad75-3e2309d2b346)

![309775584-9f6be3b9-3b62-4ad9-8d99-231a5270f28b](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/e735f525-e27a-4f6d-96fb-a5e3c3f784b6)

![309775615-91066ad0-2735-480f-9b53-9e4a347aba79](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/7d44484b-3216-4e96-9eb3-47b1ff716ce9)

![309775642-d693da73-c874-412d-960f-bba1cf76f30e](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/97cf8e74-e0ca-45ea-8eaa-2d95d4cd5672)

![309775671-90c2429b-add8-445b-94ce-f577e04ca9ef](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/4f327fd3-76c5-48ff-9a93-1879cfb23b96)

![309775701-63e11248-f153-40c6-b1fa-79a0ad4646e7](https://github.com/PRAJAN-23013995/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150313345/74088cd3-3107-4f2f-bf24-614908d06bea)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
