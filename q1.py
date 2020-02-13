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

linearRegressor = LinearRegression()

meanVariance = []
meanBias = []

for i in range(1, 10): # Choosing polynomial power 
    
    poly_prediction = []
    poly = PolynomialFeatures(i)
    X_test_poly = poly.fit_transform(X_test)

    for j in range(0, 10): # Choosing training set
        X_poly = poly.fit_transform(X_train_split[j])
        linearRegressor.fit(X_poly, Y_train_split[j])  # Training model on subset

        X_test_poly = poly.fit_transform(X_test)            
        poly_prediction.append(linearRegressor.predict(X_test_poly)) # Predicting test set output on model
        j += 1
        
    # poly_prediction is a 10*500 matrix

    meanVariance.append(np.mean(np.var(poly_prediction, axis = 0)))
    bias = abs(np.mean(poly_prediction, axis = 0) - Y_test)# 500 bias values - takes mean of all same polynomial models at a test point
    meanBias.append(np.mean(bias)) # Averages over all test points

    i += 1
     

plot.plot(range(1,10), meanBias, color = 'red')
plot.plot(range(1,10), meanVariance, color = 'blue')
plot.title('Bias & Variance')
plot.xlabel("Complexity")
plot.ylabel('Y')
plot.show()

fig, ax = plot.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(np.column_stack((range(1,10) ,meanBias, meanVariance)) , columns=['Degree', 'Bias', 'Variance'])
df.D = df.D.astype(int)
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
fig.tight_layout()
plot.title('Bias & Variance' + str(j), loc = 'center')
plot.show()
    





    






