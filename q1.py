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
meanBiasSquare = []


for i in range(1, 10): # Choosing polynomial power 
    
    poly_prediction = []
    poly = PolynomialFeatures(i, include_bias=False)
    X_test_poly = poly.fit_transform(X_test)

    for j in range(0, 10): # Choosing training set
        X_poly = poly.fit_transform(X_train_split[j])
        linearRegressor.fit(X_poly, Y_train_split[j])  # Training model on subset      
        poly_prediction.append(linearRegressor.predict(X_test_poly)) # Predicting test set output on model
        j += 1

    # poly_prediction is a 10*500 matrix

    meanVariance.append(np.mean(np.var(poly_prediction, axis = 0)))
    # bias = abs(np.mean(poly_prediction, axis = 0) - Y_test)# 500 bias values - takes mean of all same polynomial models at a test point
    bias_square = (np.mean(poly_prediction, axis = 0) - Y_test)**2 # 500 bias^2 values - takes mean of all same polynomial models at a test point
    meanBias.append(np.sqrt(np.mean(bias_square))) # Averages over all test points
    meanBiasSquare.append(np.mean(bias_square)) # Averages over all test points

    i += 1
     
meanVariance = pd.Series(meanVariance)
plot.plot(range(1,10), meanBiasSquare, color = 'red', label = "Bias^2")
plot.plot(range(1,10), meanVariance, color = 'blue', label = "Variance")
plot.title('Bias^2 & Variance')
plot.xlabel("Complexity")
plot.ylabel('Error')
plot.legend()
plot.savefig('Images/graph1.png') 
plot.savefig('graph1.png') 
plot.close() 

fig, axOne = plot.subplots()

color = 'tab:red'
axOne.set_xlabel("Complexity")
axOne.set_ylabel("Error", color=color)
axOne.plot(range(1,10), meanBiasSquare, color=color, label="Bias^2")
axOne.tick_params(axis='y', labelcolor=color)
plot.legend(loc = 'upper left')
axTwo = axOne.twinx()
color = 'tab:blue'
axTwo.set_ylabel("Error", color=color)
axTwo.plot(range(1,10), meanVariance, color=color, label="Variance")
axTwo.tick_params(axis='y', labelcolor=color)
plot.legend(loc = 'upper right')
plot.savefig("Images/DoubleScale1.png")
plot.close()

plot.plot(range(1,10), meanBiasSquare, color = 'red', label = "Bias^2")
plot.plot(range(1,10), meanVariance*100, color = 'blue', label = "Variance*100")
plot.title('Bias^2 & Variance*100')
plot.xlabel("Complexity")
plot.ylabel('Error')
plot.legend()
plot.savefig("Images/ScaleVariance1.png")
plot.show()
plot.close() 

plot.plot(range(1,10), meanBiasSquare, color = 'red', label = "Bias^2")
plot.title('Bias^2')
plot.xlabel("Complexity")
plot.ylabel('Error')
plot.legend()
plot.savefig('Images/bias1.png') 
plot.close()   

# plot.plot(range(1,10), meanVariance, color = 'blue', label = "Variance")
# plot.title('Variance')
# plot.xlabel("Complexity")
# plot.ylabel('Error')
# plot.legend()
# plot.savefig('Images/variance1.png') 
# plot.close()    

fig, ax = plot.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
df = pd.DataFrame(np.column_stack((range(1,10) ,meanBias, meanBiasSquare, meanVariance)) , columns=['Degree', 'Bias', 'Bias^2', 'Variance'])
df.Degree = df.Degree.astype(int)
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
fig.tight_layout()
plot.subplots_adjust(top=0.9)
plot.title('Bias & Variance', loc = 'center')
plot.savefig('Images/table1.png') 
plot.savefig('table1.png')
plot.close()           
    





    






