from sklearn.linear_model import LinearRegression as LR
from matplotlib import pyplot as plt

import numpy as np

body_data=np.array([[20],[40],[115],[15],[44],[250],[5]]) #THIS ARRAY HOLDS ["BODY WEIGHT"] FOR DIFFERENT ANIMALS
brain_data=np.array([[4],[12],[16],[2],[12],[34],[1]])  #THIS ARRAY HOLDS ["BRAIN WEIGHT"] FOR DIFFERENT ANIMALS
model=LR()
model=model.fit(body_data,brain_data) #training of data

body_w=int(input("ENTER THE BODY WEIGHT:"))

#visualization

plt.scatter(body_data,brain_data,c='r')
plt.plot(body_data,model.predict(body_data))
plt.xlabel("BODY WEIGHT")
plt.ylabel("BRAIN WEIGHT")
plt.title("RELATION BETWEEN BRAIN AND BODY WEIGHT")
plt.show()

pred=str(model.predict(body_w));  #save the prediction


print("BRAIN WEIGHT FOR GIVEN BODY WEIGHT IS: "+str(pred))

