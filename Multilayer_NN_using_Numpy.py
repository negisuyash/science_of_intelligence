import numpy as np

x=np.array([[0,1,1],[2,5,6],[3,4,1],[3,3,1]])
y=np.array([[4,3,2,1]]).T  #4x1

def sigmoid(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

w0=np.random.randn(3,3)
w1=np.random.rand(3,2)
w2=np.random.randn(2,1)

for i in range(15):
	l0=x
	l1=sigmoid(l0.dot(w0))
	l2=sigmoid(l1.dot(w1))
	l3=sigmoid(l2.dot(w2))

	l3_error=y-l3
	print(l3_error[0][0]+l3_error[1][0]+l3_error[2][0]+l3_error[3][0])
	l3_delta=l3_error*sigmoid(l3,deriv=True)
	l2_error=l3_delta.dot(w2.T)
	l2_delta=l2_error*sigmoid(l2,deriv=True)
	l1_error=l2_delta.dot(w1.T)
	l1_delta=l1_error*sigmoid(l1,deriv=True)

	w2+=(l2.T.dot(l3_delta))
	w1+=(l1.T.dot(l2_delta))
	w0+=(l0.T.dot(l1_delta))

#4x3 3x3=4x3 3x2=4x2 2x1=4x1