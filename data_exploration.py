"""A classification problem "classifies" data into classes(types).
    That is predicting what type of thing something is.
    Binary classification classifies data as one thing or the other(1 of 2)
    Multi_class classification has multiple classes but assigns one label to each
    Multi_label classification could have multiple different classes but each target sample has more than 1 label"""

import sklearn.datasets
from sklearn.datasets import make_circles
import pandas as pd

#Making Custom Dataset
#Make 1000 samples
n_samples = 1000

#Create circles

""""For below code, we have 1000 data points arranged in two concentric circles. Each point has 
two features (its x(X[:, 0]) and y(X[:, 1]) coordinates on a 2D plot) and a label of either 0 
(inner circle) or 1 (outer circle). The model's job is to take a point's 
coordinates as input and predict which circle it belongs to. This is binary 
classification — two possible outputs. Unlike linear regression where the 
answer was a continuous number, here the answer is always one of two classes."""

X, y = make_circles(n_samples,
                    noise = 0.03,
                    random_state = 42) #Same as a random seed

print(f"First 5 samples of X: {X[:5]}")
print(f"\nFirst 5 samples of y: {y[:5]}")

#Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "Label": y})
print(circles.head(10)) #Prints all pandas table as string
#print(circles.head(10)) #Prints table with first 10 rows of data

#Visualize circle
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], #First ones are x axis
            y=X[:, 1], #Second ones are y axis
            c=y,
            cmap=plt.cm.RdYlBu) #Just a color map
plt.show()

#check input and output shapes
print(X.shape, y.shape)

#Turn Data to tensors
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X[:5], "\n", y[:5])

#Split into training and test set randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2, #What size goes to the test to 20% here
                                                    random_state = 42)
print(len(X_train))