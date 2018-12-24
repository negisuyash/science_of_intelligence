import numpy

def sigmoid(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+numpy.exp(-x))
numpy.random.seed(1)
x=numpy.array([[0,1,1],[2,5,6],[3,4,1],[3,3,1]])
#y=numpy.array([[4],[3],[2],[1]])
y=numpy.array([[4,3,2,1]]).T

w0=numpy.random.randn(3,4)
w1=numpy.random.randn(4,1)

for j in range(1000000):
	l0=x
	l1=sigmoid(x.dot(w0))
	l2=sigmoid(l1.dot(w1))

	

	l2_error=y-l2
	print(l2_error[0][0]+l2_error[1][0]+l2_error[2][0]+l2_error[3][0])
	l2_delta=l2_error*sigmoid(l2,deriv=True)
	l1_error=l2_delta.dot(w1.T)
	l1_delta=l1_error*sigmoid(l1,deriv=True)

	w1+=(l1.T.dot(l2_delta))
	w0+=(l0.T.dot(l1_delta))

print(l2)
	


