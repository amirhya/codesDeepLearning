import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris =load_iris()
X=iris.data[:,(2,3)]
y=(iris.target==0).astype(int)
clf=Perceptron()
clf.fit(X,y)

yPred=clf.predict([[2,0.5]])
print(yPred)