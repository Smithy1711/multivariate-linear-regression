import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('home.txt', names=["size", "bedroom", "price"]) # reads data and gives header names from home.txt

# normalise data to ensure size does not dominate number of bedroom

my_data = (my_data - my_data.mean())/my_data.std() # mean normalisation

# set matrixes
X = my_data.iloc[:, 0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones, X), axis=1)

y = my_data.iloc[:,2:3].values
theta = np.zeros([1,3])

# hyper params
alpha = 0.005
iters = 500


# cost function
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2) # squared differences between predicted and actual values
    return np.sum(tobesummed)/(2*len(X))

# gradient descent function

def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X))* np.sum(X*(X @ theta.T - y))
        cost[i] = computeCost(X,y,theta)

    return theta,cost

# run the gradient descent and cost func
gradient, cost = gradientDescent(X,y,theta,iters,alpha)
print(gradient)
finalCost = computeCost(X,y,gradient)
print(finalCost)


#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 
plt.grid(True)
plt.show()