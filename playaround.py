import pickle
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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

# j = 0
i = 1 
degree_array = [] 
meanVariance = []
meanBias = []

# print(Outputs.shape)

# print(len(Y_test))
while i < 10: # Choosing polynomial power 
    
    # bias_array = []
    # variance_array = []
    Outputs = np.empty([len(Y_test)])
    poly = PolynomialFeatures(i)
    X_test_poly = poly.fit_transform(X_test)
    
    # i = 1
    j = 0
    while j < 10: # Choosing training set

       
        X_poly = poly.fit_transform(X_train_split[j])
        # print(X_poly)
        linearRegressor.fit(X_poly, Y_train_split[j])        
        # plot.scatter(X_train_split[j], Y_train_split[j], color = 'red')
        # plot.scatter(X_train_split[j], linearRegressor.predict(X_poly), color = 'blue')
        # plot.title('X vs Y (Training set)')
        # plot.xlabel('X')
        # plot.ylabel('Y')
        # plot.show()
        


        # print(X_test.shape, X_train_split[j].shape)

        # X_test_poly = poly.fit_transform(X_test)
        # print(linearRegressor.predict(X_test_poly).shape)
        Outputs = np.column_stack((Outputs, linearRegressor.predict(X_test_poly)))
        j = j + 1

        

        # print(Outputs)
        print(Outputs.shape)
        # bias = np.mean((np.mean(Outputs) - Y_test) ** 2)
        # variance = np.var(Outputs)
        # print(variance)
        # print(bias)
        # bias = np.sqrt(bias)

        # bias_array.append(bias)
        # variance_array.append(variance)
        # degree_array.append(int(i))    

    degree_array.append(i)
    variance = np.var(Outputs, axis = 0)
    meanVariance.append(np.mean(variance))
    # bias = (Outputs - Y_test) ** 2
    # meanBias.append(np.mean(bias))

    i += 1
    # plot.show()
     

# plot.plot(range(1,10), meanBias, color = 'red')
plot.plot(range(1,10), meanVariance, color = 'blue')
plot.title('Bias^2 & Variance')
plot.xlabel("Complexity")
plot.ylabel('Y')
plot.show()

# fig, ax = plot.subplots()
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')

# df = pd.DataFrame(np.column_stack((range(1,10) ,meanBias, meanVariance)) , columns=list("DBV"))
# df.D = df.D.astype(int)
# ax.table(cellText=df.values, colLabels=df.columns, loc='center')
# fig.tight_layout()
# plot.title('Bias^2 & Variance' + str(j), loc = 'center')
# plot.show()
    





    






