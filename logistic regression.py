from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
iris=datasets.load_iris()
#print(list(iris.keys()))
#print(iris['data'])
#print("check",iris.data)
#print(iris['target'])
#print(iris['data'].shape)
x=iris["data"][:,3:]
#try using this y=(iris["target"]==2).astype(np.int) it says np.int is deprecated search for numpy deprecated
y=(iris["target"]==2).astype(int)#to make it classifier not regression
clf= LogisticRegression()
clf.fit(x,y)
example=clf.predict([[1.6]])
print(example)
x_new=np.linspace(0,3,1000).reshape(-1,1)
#print(x_new)
x_prob=clf.predict_proba(x_new)
#print("see",x)
#print(x_prob)
plt.plot(x_new,x_prob[:,1],"g-","label=virginica")
plt.show()
