#!/usr/bin/env python
# coding: utf-8
​
# In[1]:
​
​
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import numpy as np
​
​
# In[2]:
​
​
# each point is length, width, type (0, 1)
​
data =[
[3,3,3,3,3,3,3,3,3],
[3,2,2,3,2,2,1,2,2],
[2,2,2,2,3,2,2,1,2],
[2,1,3,2,2,1,2,2,1],
[1,2,2,1,2,1,1,2,1],
[1,1,2,1,2,2,1,1,0],
[0,1,0,1,2,1,2,2,0],
[2,0,1,1,2,0,2,3,0],
[3,1,3,3,2,2,1,2,2],
[3,2,2,3,3,2,3,3,3],
[2,2,2,2,2,1,2,2,1],
[2,1,3,2,3,2,2,1,2],
[1,2,2,1,2,2,1,1,1],
[1,1,2,1,2,1,1,2,2],
[0,1,0,1,2,0,2,3,0],
[2,0,1,1,2,1,2,2,0],
[3,2,2,3,3,3,3,3,3],
[3,3,3,3,2,2,1,2,2],
[2,1,3,2,3,2,2,1,2],
[2,2,2,2,2,1,2,2,1],
[1,1,2,1,2,1,1,2,1],
[1,2,2,1,2,2,1,1,0],
[2,0,1,1,2,1,2,2,0],
[0,1,0,1,2,0,2,3,0],
[3,3,2,3,2,2,3,3,3],
[3,2,3,3,3,3,1,2,3],
[2,2,3,2,2,1,2,1,2],
[2,1,2,2,3,2,2,2,2],
[1,2,2,1,2,2,1,2,1],
[1,1,2,1,2,1,1,1,1],
[0,1,1,1,2,0,2,2,0],
[2,0,0,1,2,1,2,3,0],
[3,1,2,3,3,2,1,2,3],
[3,2,3,3,2,2,3,3,3],
[2,2,3,2,3,2,2,2,2],
[2,1,2,2,2,1,2,1,2],
[1,2,2,1,2,1,1,1,1],
[1,1,2,1,2,2,1,2,1],
[0,1,1,1,2,1,2,3,0],
[2,0,0,1,2,0,2,2,0],
[3,2,3,3,2,2,3,3,3],
[3,3,2,3,3,3,1,2,3],
[2,1,2,2,2,1,2,1,2],
[2,2,3,2,3,2,2,2,2],
[1,1,2,1,2,2,1,2,1],
[1,2,2,1,2,1,1,1,1],
[2,0,0,1,2,0,2,2,0],
[0,1,1,1,2,1,2,3,0]]
​
mystery_flower = [3,3,3,2,3,3,3,3]
​
​
# In[38]:
​
​
# scatter plot them
#def vis_data():
    #plt.grid()
​
    #for i in range(len(data)):
       # c = 'r'
        #if data[i][2] == 0:
            #c = 'b'
        #plt.scatter([data[i][0]], [data[i][1]], c=c)
​
    #plt.scatter([mystery_flower[0]], [mystery_flower[1]], c='gray')
​
#vis_data()
​
​
# In[3]:
​
​
# network
​
#       o  flower type
#      / \  w1, w2, b
#     o   o  length, width
​
​
# In[4]:
​
​
# activation function
​
def sigmoid(x):
    return 1/(1+np.exp(-x))
​
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))
​
​
# In[10]:
​
​
#X = np.linspace(-5, 5, 100)
​
#plt.plot(X, sigmoid(X), c="b") # sigmoid in blue
#fig = plt.plot(X, sigmoid_p(X), c="r") # sigmoid_p in red
​
​
# In[20]:
​
​
# train
​
def train():
    #random init of weights
    w1 = np.random.randn()
    w2 = np.random.randn()
    w3 = np.random.randn()
    w4 = np.random.randn()
    w5 = np.random.randn()
    w6 = np.random.randn()
    w7 = np.random.randn()
    w8 = np.random.randn()
    b = np.random.randn()
    
    iterations = 10000
    learning_rate = 0.1
    costs = [] # keep costs during training, see if they go down
    
        # get a random point
    for i in range(iterations):
        ri = np.random.randint(len(data))
        point = data[ri]
        
        z = point[0] * w1 + point[1] * w2 + point[2] * w3 + point[3] * w4 + point[4] * w5 + point[5] * w6 + point[6] * w7 + point[7] * w8 + b
        pred = sigmoid(z) # networks prediction
        
        target = point[8]
        
        # cost for current random point
        cost = np.square(pred - target)
        
        # print the cost over all data points every 1k iters
        if i % 100 == 0:
            c = 0
            for j in range(len(data)):
                p = data[j]
                p_pred = sigmoid(w1 * p[0] + w2 * p[1] + w3 * p[2] + w4 * p[3] + w5 * p[4] + w6 * p[5] + w7 * p[6] + w8 * p[7] + b)
                c += np.square(p_pred - p[8])
            costs.append(c)
        
        dcost_dpred = 2 * (pred - target)
        dpred_dz = sigmoid_p(z)
        
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_dw3 = point[2]
        dz_dw4 = point[3]
        dz_dw5 = point[4]
        dz_dw6 = point[5]
        dz_dw7 = point[6]
        dz_dw8 = point[7]
        dz_db = 1
        
        dcost_dz = dcost_dpred * dpred_dz
        
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_dw3 = dcost_dz * dz_dw3
        dcost_dw4 = dcost_dz * dz_dw4
        dcost_dw5 = dcost_dz * dz_dw5
        dcost_dw6 = dcost_dz * dz_dw6
        dcost_dw7 = dcost_dz * dz_dw7
        dcost_dw8 = dcost_dz * dz_dw8
        dcost_db = dcost_dz * dz_db
        
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        w3 = w1 - learning_rate * dcost_dw3
        w4 = w2 - learning_rate * dcost_dw4
        w5 = w1 - learning_rate * dcost_dw5
        w6 = w2 - learning_rate * dcost_dw6
        w7 = w1 - learning_rate * dcost_dw7
        w8 = w2 - learning_rate * dcost_dw8
        b = b - learning_rate * dcost_db
        
    return costs, w1, w2,w3,w4,w5,w6,w7,w8,b
        
costs, w1,w2,w3,w4,w5,w6,w7,w8,b = train()
​
fig = plt.plot(costs)
​
​
# In[22]:
​
​
# predict what the myster flower is!
​
z = w1 * mystery_flower[0] + w2 * mystery_flower[1] + w3 * mystery_flower[2] + w4 * mystery_flower[3] + w5 * mystery_flower[4] + w6 * mystery_flower[5] + w7 * mystery_flower[6] + w8 * mystery_flower[7] + b
pred = sigmoid(z)
​
print(pred)
print("close to 0 -> LEVEL-0, close to 1 -> LEVEL-1,close to 2 -> LEVEL-2,close to 3 -> LEVEL-3")
​
​
# In[42]:
​
​
# check out the networks predictions in the x,y plane
#for x in np.linspace(0, 6, 20):
    #for y in np.linspace(0, 3, 20):
        #pred = sigmoid(w1 * x + w2 * y + b)
        #c = 'b'
        #if pred > .5:
            #c = 'r'
        #plt.scatter([x],[y],c=c, alpha=.2)
        
# plot points over network predictions
# you should see a split, with half the predictions blue
# and the other half red.. nicely predicting each data point!
#vis_data()
​
​
# In[ ]:
​
​
​
​

Saved successfully!
