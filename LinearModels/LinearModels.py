import numpy as np
import pandas as pd
import random


# Using pseudo inverse, directly yields the weights and intercept
class linear_regression_svd:
    
    def __init__(self):
        return 
    
    def initialize_x0(self, data):
        data["b_x0"] = [1] * len(data)
        return data
                    
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.X = self.initialize_x0(self.X)
        self.X = np.asarray(self.X)
        
        self.theta = np.linalg.pinv(self.X).dot(self.y)
        self.w = self.theta[:-1]
        self.b = self.theta[-1]
        
    def predict(self, test):
        self.test = test.copy()
        self.test = self.initialize_x0(self.test)
        self.test = np.asarray(self.test)
        
        return self.test.dot(self.theta) 


# iterative method, gradually approaches the solution
# iterative method, gradually approaches the solution
class GradientDescent:
    
    def __init__(self, learning_rate=0.01, n_iterations=2000, alpha=0, r=0):
        self.eta = learning_rate 
        self.n_iterations = n_iterations
        self.alpha = alpha  # regularization 
        self.r = r # l1 ratio   
        self.history = []
    
    def get_l1_l2_w(self):
        a, b = 0, 0
        for i in list(self.theta.squeeze()):
            a += i 
            b += i ** 2
        
        return (a, b)


    def computeCost(self):
        theta_sum, theta_squared_sum = self.get_l1_l2_w()
        cost = (1 / self.m) * self.X.T.dot(self.X.dot(self.theta) - self.y.T)
        elastic_net = (self.r * self.alpha * theta_sum) + (((1 - self.r) / 2) * self.alpha * theta_squared_sum)
        return cost + elastic_net 
        
    def initialize_x0(self, data):
        data = pd.DataFrame(data)
        data["b_x0"] = [1] * len(data)
        return data
    
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.X = self.initialize_x0(self.X)
        self.X = np.asarray(self.X)
        self.m = len(self.X)
        
        self.y = np.asarray(self.y).reshape(len(self.y) , 1).T        
            
        self.theta = np.asarray([0] * len(self.X.T)).T
        self.theta = self.theta.reshape(10, 1)
        
        for iteration in range(self.n_iterations):
            
            gradients = self.computeCost()
            self.theta = self.theta - (self.eta * gradients)
            self.history.append(gradients.sum())
            
        self.w = self.theta[:-1]
        self.b = self.theta[-1]
        
    def predict(self, test):
        self.test = test.copy()
        self.test = self.initialize_x0(self.test)
        self.test = np.asarray(self.test)
        
        return self.test.dot(self.theta)



class LogisticRegression:

    def __init(self):
        self.w = None
        self.b = None
        return

    def σ(self, z):
        s = 1/(1+np.exp(-z))
        return s

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim, 1))
        b = 0

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))

        return w, b

    def propagate(self, w, b, X, Y):
        m = X.shape[1]

        Z = np.dot(w.T, X) + b
        A = self.σ(Z)
        cost = -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

        dw = (1/m)*np.dot(X, (A-Y).T)
        db = (1/m)*np.sum(A-Y)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost


    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):

        costs = []

        for i in range(num_iterations):

            grads, cost = self.propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def predict(self, X):

        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        self.w = self.w.reshape(X.shape[0], 1)

        A = self.σ(np.dot(self.w.T, X) + self.b)

        for i in range(A.shape[1]):

            Y_prediction[0, i] = 1 if A[0, i] >= 0.5 else 0

        assert(Y_prediction.shape == (1, m))

        return Y_prediction
    
    def fit(self, X, y, num_iterations = 2000, learning_rate = 0.5):
        w, b = self.initialize_with_zeros(X.shape[0])
        parameters, grads, costs = self.optimize(w, b, X, y, num_iterations, learning_rate)

        self.w = parameters["w"]
        self.b = parameters["b"]