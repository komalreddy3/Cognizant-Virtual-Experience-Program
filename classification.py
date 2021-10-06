from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris= datasets.load_iris()
#printing features and labels with description
#print(iris.DESCR)
features=iris.data
labels=iris.target
print(features[0],labels[0])

#training the classifier
clf= KNeighborsClassifier()
clf.fit(features,labels)
preds=clf.predict([[1,1,1,1]])
print(preds)
