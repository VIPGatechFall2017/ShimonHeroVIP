import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' 
# Specifying column names.
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)

iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['species_num'] = [iris_class[i] for i in iris.species]

x = iris.drop(['species', 'species_num'], axis = 1)
y = iris.species_num

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train,y_train)

print (knn.score(x_test,y_test))