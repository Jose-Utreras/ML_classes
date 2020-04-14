
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_boston

from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()

def cost(w1, w2, b, X, t):
    '''
    Evaluate the cost function in a non-vectorized manner for
    inputs `X` and targets `t`, at weights `w1`, `w2` and `b`.
    '''
    costs = 0
    for i in range(len(t)):
        y_i = w1 * X[i, 0] + w2 * X[i, 1] + b
        t_i = t[i]
        costs += 0.5 * (y_i - t_i) ** 2
    return costs / len(t)

def cost_vectorized(w1, w2, b, X, t):
    w    = np.array([w1,w2])
    cost = 0.5*(np.dot(w,X.T)+b-t) ** 2
    return cost.sum()/len(t)

def solve_exactly(X, t):
    '''
    Solve linear regression exactly. (fully vectorized)

    Given `X` - NxD matrix of inputs
          `t` - target outputs
    Returns the optimal weights as a D-dimensional vector
    '''
    N, D = np.shape(X)
    A = np.matmul(X.T, X)
    c = np.dot(X.T, t)
    return np.matmul(np.linalg.inv(A), c)

# Vectorized gradient function
def gradfn(weights, X, t):
    '''
    Given `weights` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Return gradient of each weight evaluated at the current value
    '''
    N, D = np.shape(X)
    y_pred = np.matmul(X, weights)
    error = y_pred - t
    return np.matmul(x_in.T, error) / float(N)

def solve_via_gradient_descent(X, t, print_every=5000,
    niter=100000, alpha=0.005):
    '''
    Given `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    for k in range(niter):
        dw = gradfn(w, X, t)
        w = w - alpha*dw
        if k % print_every == 0:
            print('Weight after %d iteration: %s' % (k, str(w)))
    return w

boston_data = load_boston()
print(boston_data['DESCR'])

data = boston_data['data']
x_input = data[:, [2,5]]
y_target = boston_data['target']

plt.title('Industrialness vs Med House Price')
plt.scatter(x_input[:, 0], y_target)
plt.xlabel('Industrialness')
plt.ylabel('Med House Price')
plt.show()

plt.title('Avg Num Rooms vs Med House Price')
plt.scatter(x_input[:, 1], y_target)
plt.xlabel('Avg Num Rooms')
plt.ylabel('Med House Price')
plt.show()

tstart= time.time()
cost(3, 5, 20, x_input, y_target)
tend= time.time()
print('%.5f seconds' %(tend-tstart))

tstart= time.time()
cost_vectorized(3, 5, 20, x_input, y_target)
tend= time.time()
print('%.5f seconds' %(tend-tstart))


w1s = np.arange(-1.0, 1.0, 0.01)
w2s = np.arange(6.0, 10.0, 0.1)
z_cost = []
for w2 in w2s:
    z_cost.append([cost_vectorized(w1, w2, -22.89831573, x_input, y_target) for w1 in w1s])
z_cost = np.array(z_cost)
np.shape(z_cost)
W1, W2 = np.meshgrid(w1s, w2s)
CS = plt.contour(W1, W2, z_cost, 25)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Costs for various values of w1 and w2 for b=-22.89')
plt.xlabel("w1")
plt.ylabel("w2")
plt.plot([-0.33471389], [7.82205511], 'o') # this will be the minima that we'll find later
plt.show()

"""
EXACT solution
"""
N=len(x_input[:,0])
x_in = np.hstack((x_input,np.ones((N,1))))

tstart = time.time()
solve_exactly(x_in, y_target)
tend = time.time()
print('%.5f seconds' %(tend-tstart))

tstart = time.time()
np.linalg.lstsq(x_in, y_target)
tend = time.time()
print('%.5f seconds' %(tend-tstart))

weights=solve_via_gradient_descent( X=x_in, t=y_target, alpha=0.007)
fit=np.dot(x_in,weights)

x=np.concatenate((x_in[:,0],x_in[:,0]))
y=np.concatenate((x_in[:,1],x_in[:,1]))
z=np.concatenate((y_target,fit))
label1 = ['real']*len(y_target)
label2 = ['fit']*len(fit)
label = np.array(label1+label2,dtype=str)


dataset = pd.DataFrame({'Industrialness':x, 'Rooms': y,
            'Price' :z , 'Value' : label })

fig = px.scatter_3d(dataset, x='Industrialness', y='Rooms', z='Price',
              color='Value')
fig.update_traces(marker=dict(size=5))
fig.show()
