from sklearn.linear_model import LinearRegression as LR
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_fwf("challenge_siraj_raval_1_dataset.txt",skipinitail=True)
x_data=data[['Brain']]
y_data=data[['Body']]

#print(data[0][0])

"""x_data=[]
y_data=[]
j=0
for i in data:
	x_data.append(i[0])
	y_data.append(i[1])

print(y_data)"""

model=LR()
model=model.fit(x_data,y_data)
pred=model.predict(x_data)
#visual
plt.scatter(x_data,y_data)
plt.plot(x_data,pred)
plt.show()

#printing error for each predicition
error=y_data-pred
print("ERROR IN PREDICTION IS :")
print(error)



