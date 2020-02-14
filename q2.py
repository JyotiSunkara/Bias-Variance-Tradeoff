import pickle
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

file = open('Assignment/Q2_data/X_train.pkl', 'rb')
X_train =  pickle.load(file)
file.close()
file = open('Assignment/Q2_data/Y_train.pkl', 'rb')
Y_train =  pickle.load(file)
file.close()
file = open('Assignment/Q2_data/X_test.pkl', 'rb')
X_test =  pickle.load(file)
file.close()
file = open('Assignment/Q2_data/Fx_test.pkl', 'rb')
Y_test =  pickle.load(file)
file.close()

# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X_test = X_test[:,np.newaxis]
linearRegressor = LinearRegression()

meanVariance = []
meanBiasSquare = []

for i in range(1, 10): # Choosing polynomial power 
    
    poly_prediction = []
    poly = PolynomialFeatures(i)
    X_test_poly = poly.fit_transform(X_test)

    for j in range(0, 20): # Choosing training set
        x_train = X_train[j]
        x_train = x_train[:,np.newaxis]

        X_poly = poly.fit_transform(x_train)
        linearRegressor.fit(X_poly, Y_train[j])  # Training model on subset

        X_test_poly = poly.fit_transform(X_test)            
        poly_prediction.append(linearRegressor.predict(X_test_poly)) # Predicting test set output on model
        
        # plot.scatter(X_train[j], Y_train[j], color = 'red')
        # plot.scatter(X_train[j], linearRegressor.predict(X_poly), color = 'blue')
        # plot.title('Polynomial Degree '+ str(i) + ' X vs Y on Training set ' + str(j))
        # plot.xlabel('X')
        # plot.ylabel('Y')
        # plot.savefig('Bodels/Poly'+str(i)+'Model'+str(j)+'.png')   
        # plot.close()   

        j += 1
        
    # poly_prediction is a 20*80 matrix

    meanVariance.append(np.mean(np.var(poly_prediction, axis = 0)))
    bias_square = (np.mean(poly_prediction, axis = 0) - Y_test)**2 # 80 bias values - takes mean of all same polynomial models at a test point
    meanBiasSquare.append(np.mean(bias_square)) # Averages over all test points

    i += 1
     

plot.plot(range(1,10), meanBiasSquare, color = 'red', label = "Bias^2")
plot.plot(range(1,10), meanVariance, color = 'blue', label = "Variance")
plot.title('Bias^2 & Variance')
plot.xlabel("Complexity")
plot.ylabel('Error')
plot.legend()
plot.savefig('graph2.png')           


fig, ax = plot.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(np.column_stack((range(1,10) ,meanBiasSquare, meanVariance)) , columns=['Degree', 'Bias^2', 'Variance'])
df.Degree = df.Degree.astype(int)
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
fig.tight_layout()
plot.subplots_adjust(top=0.9)
plot.title('Bias^2 & Variance', loc = 'center')
plot.savefig('table2.png')           



