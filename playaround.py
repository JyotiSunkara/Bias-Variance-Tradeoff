import pickle
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

file = open('Assignment/Q1_data/data.pkl', 'rb')
data =  pickle.load(file)
file.close()

X = data[:, 0]
X = X[:,np.newaxis]
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

# print(len(X_train_split), len(X_train_split[0]))

# Training polynomials
linearRegressor = LinearRegression()



j = 0  
while j < 10: # Choosing training subset 

    bias_array = []
    variance_array = []
    i = 1
    while i < 10: # Choosing polynomial power

        poly = PolynomialFeatures(i)
        X_poly = poly.fit_transform(X_train_split[j])
        # print(X_poly)
        linearRegressor.fit(X_poly, Y_train_split[j])        
        # plot.scatter(X_train_split[j], Y_train_split[j], color = 'red')
        # plot.scatter(X_train_split[j], linearRegressor.predict(X_poly), color = 'blue')
        # plot.title('X vs Y (Training set)')
        # plot.xlabel('X')
        # plot.ylabel('Y')
        # plot.show()
        i = i + 1

        # print(X_test.shape, X_train_split[j].shape)
        X_test_poly = poly.fit_transform(X_test)
        Outputs = linearRegressor.predict(X_test_poly)
        # print(Outputs)
        # print(Outputs.shape)
        bias = np.mean((np.mean(Outputs) - Y_test) ** 2)
        bias = np.sqrt(bias)
        variance = np.var(Outputs)
        bias_array.append(bias)
        variance_array.append(variance)
        print(variance)
        print(bias)

    plot.plot(range(1,10), bias_array, color = 'red')
    plot.plot(range(1,10), variance_array, color = 'blue')
    plot.title('Bias^2 & Variance')
    plot.xlabel(j)
    plot.ylabel('Y')
    plot.show()

    # fig, ax = plot.subplots()
    # collabel = "Bias^2", "Variance"
    # celltext = np.append(bias_array, variance_array)
    # ax.xaxis.set_visible(False) 
    # ax.yaxis.set_visible(False)
    # ax.table(cellText = celltext, colLabels = collabel, loc='center')

    # ax.plot(celltext[:,0],celltext[:,1])

    # fig.tight_layout()

    # plot.show()
    
    j = j + 1
    # plot.show()




    






