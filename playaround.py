import pickle
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

file = open('Assignment/Q1_data/data.pkl', 'rb')
data =  pickle.load(file)
file.close()

X = data[:, 0]
Y = data[:, 1]

# Partitioning into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1) 

# Reshuffling data
training_data = np.column_stack((X_train, Y_train))
np.random.shuffle(training_data)
X_train = training_data[:, 0]
X_train = X_train[:, np.newaxis]
Y_train = training_data[:, 1]

# Splitting training data into 10 parts
X_train_split = np.array_split(X_train, 10)
Y_train_split = np.array_split(Y_train, 10)

print(len(X_train_split), len(X_train_split[0]))

# Training y = mx
linearRegressor = LinearRegression()
i = 0

while i < 10:  
    
    linearRegressor.fit(X_train_split[i], Y_train_split[i])
    plot.scatter(X_train_split[i], Y_train_split[i], color = 'red')
    plot.plot(X_train_split[i], linearRegressor.predict(X_train_split[i]), color = 'blue')
    plot.title('X vs Y (Training set)')
    plot.xlabel('X')
    plot.ylabel('Y')
    plot.show()
    i += 1





