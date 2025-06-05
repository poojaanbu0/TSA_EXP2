# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
#### Date:12.03.25
#### Register Number: 212222240072

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1.Import necessary libraries (NumPy, Matplotlib)

2.Load the dataset

3.Calculate the linear trend values using least square method

4.Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('/content/seattle_weather_1948-2017.csv')
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df = df.dropna(subset=['PRCP'])
df['Date_ordinal'] = df['DATE'].apply(lambda x: x.toordinal())
X = df['Date_ordinal'].values.reshape(-1, 1)
y = df['PRCP'].values
print(df.columns)
```
A - LINEAR TREND ESTIMATION
```
#Trend equation using Linear Equation
linear_model = LinearRegression()
linear_model.fit(X, y)
df['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10, 6))
plt.plot(df['DATE'], df['PRCP'], label='Original Data', color='blue')  
plt.plot(df['DATE'], df['Linear_Trend'], color='yellow', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```
## Polynomial TRend EStimation 4th degree
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
df['Polynomial_Trend'] = poly_model.predict(X_poly)
plt.figure(figsize=(10, 6))
plt.plot(df['DATE'], df['PRCP'], label='Original Data', color='blue')  # Changed 'PRCP' to the target variable
plt.plot(df['DATE'], df['Polynomial_Trend'], color='green', label='Polynomial Trend (Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
A - LINEAR TREND ESTIMATION
![365763543-0db73474-7a0d-468a-97dc-f024919214b9](https://github.com/user-attachments/assets/67a1d430-196b-4f91-baf9-948b3ff7e8b8)

B- POLYNOMIAL TREND ESTIMATION
![365763635-8fd6bfd6-1e84-4983-b1aa-8e2380499122](https://github.com/user-attachments/assets/71dd46b9-59ef-4239-8a42-51b2c31b2f1b)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
