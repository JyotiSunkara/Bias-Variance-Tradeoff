import pickle
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy

file = open('Assignment/Q1_data/data.pkl', 'rb')
data =  pickle.load(file)
file.close()

count = 0 
for item in data:
    print(item)
    count = count + 1

print("There are ", count, "entries!")

print(data.shape)


X = data[:, 0]
Y = data[:, 1]


# for item in X:
#     print(item)

# for item in Y:
#     print(item)

# print(X.shape)
# print(Y.shape)

# Partitioning into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1) 

# Partition into 10 training sets to obtain 10 models 

# Training y = mx
linearRegressor = LinearRegression()  
linearRegressor.fit(X_train[:,numpy.newaxis],Y_train)

plot.scatter(X_train, Y_train, color = 'red')
plot.plot(X_train, linearRegressor.predict(X_train[:,numpy.newaxis]), color = 'blue')
plot.title('X vs Y (Training set)')
plot.xlabel('X')
plot.ylabel('Y')
plot.show()






