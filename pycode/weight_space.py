from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl

#weight-space view
#linear model
sigma = 1
#training data
X = np.concatenate(([[-5,2,5]],[[1,1,1]]), axis=0)
y = np.array([[-6],[1],[4]])
#Prior and Posterior distribution parameters
#set(0,1)
w_prior = (np.zeros([2]),np.eye(2))
#inverse,@表示矩阵向量的乘法
A = (X @ X.T)/sigma**2 + nl.inv(w_prior[1])
#满足如下高斯分布
w_posterior = ((nl.inv(A) @ X @ y).squeeze(),nl.inv(A))
#plotting
w_1,w_2 = np.meshgrid(np.linspace(-2,2,100),np.linspace(-2,2,100))
ws = np.dstack((w_1, w_2))
#生成多元正态分布数组
prior_rv = multivariate_normal(w_prior[0],w_prior[1])
#生成概率密度函数
prior_z = prior_rv.pdf(ws)

#似然
def likelihood(X, y,ws,sigma):
    n = X.shape[1]
    N = 1/(2*np.pi*sigma**2)**(n/2)
    return np.exp(-np.sum((y.T-ws@X)**2,axis=-1)/(2*sigma**2))/N
likelihood_z = likelihood(X,y,ws,1)
posterior_rv = multivariate_normal(w_posterior[0],w_posterior[1],allow_singular=True)
posterior_z = posterior_rv.pdf(ws)

plt.figure(figsize=(12,4))
#prior_z.T
plt.subplot(1,3,1)
plt.contour(w_1, w_2,prior_z.T, colors='k')
plt.title(r'$(w_1, w_2)$ Prior')
plt.xlabel(r'Slope $w_1$')
plt.ylabel(r'Intercept $w_2$')

plt.subplot(1,3,2)
plt.contour(w_1, w_2, likelihood_z.T, colors='k')
plt.title(r'$(w_1, w_2)$ Likelihood')
plt.xlabel(r'Slope $w_1$')
plt.ylabel(r'Intercept $w_2$')

plt.subplot(1,3,3)
plt.contour(w_1, w_2, posterior_z.T, colors='k')
plt.title(r'$(w_1, w_2)$ Posterior')
plt.xlabel(r'Slope $w_1$')
plt.ylabel(r'Intercept $w_2$')

plt.tight_layout()

plt.show()


# New data
Xs = np.concatenate((np.linspace(-6,6,100).reshape(1,100), np.ones([1,100])))

# Parameter inference for each instance
y_mean = (Xs.T @ nl.inv(A) @ X @ y).squeeze()
y_std = np.sqrt((Xs.T @ nl.inv(A) @ Xs).diagonal())

# Plotting
plt.figure()

plt.scatter(X[0,:], y, marker='x', color='r', zorder=10)
plt.plot(Xs[0,:], y_mean, color='g')
plt.fill_between(Xs[0,:], y_mean - y_std, y_mean + y_std, alpha=0.2, color='b')
plt.xlim([-6,6])

plt.xlabel(r'Input $x$')
plt.ylabel(r'Output $f(x)$')
plt.title(r'Bayesian Linear Model $f(\mathbf{x}) = w_1 + w_2 x$')

plt.show()

#Projections of Inputs into Feature Space
#投影到多项式空间的次数，φ
def phi(X,N):
    return np.concatenate([np.power(X,i) for i in range(N)],axis=1).T

#随机噪声标准差
n_data = 10
sigma = 1
N = 5
X = np.random.random(size=[n_data,1])*10-5
y = np.random.random(size=[n_data,1])*3-1.5
#先验和后验分布参数
w_prior = (np.zeros([N]),np.eye(N))
A = (phi(X,N)@phi(X, N).T)/sigma**2+nl.inv(w_prior[1])

prior_rv = multivariate_normal(w_prior[0], w_prior[1], allow_singular=True)
posterior_rv = multivariate_normal(w_posterior[0], w_posterior[1], allow_singular=True)

Xs = np.linspace(-6,6,100).reshape(100,1)
features = phi(Xs,N)

# Plotting
n_samples = 3
#rvs:从多元正态分布中随机抽取样本
prior_samples = features.T @ prior_rv.rvs(n_samples).T
posterior_samples = features.T @ posterior_rv.rvs(n_samples).T

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(Xs.squeeze(), prior_samples)
plt.xlim([-6,6])
plt.ylim([-2,2])

plt.title('Prior Random Samples')
plt.xlabel(r'Input $x$')
plt.ylabel(r'Output $f(x)$')

plt.subplot(1,2,2)
plt.scatter(X.squeeze(), y.squeeze(), marker='x', color='r', zorder=10)
plt.plot(Xs.squeeze(), posterior_samples)
plt.xlim([-6,6])
plt.ylim([-2,2])

plt.title('Posterior Random Samples')
plt.xlabel(r'Input $x$')
plt.ylabel(r'Output $f(x)$')

plt.tight_layout()
plt.show()

#回归

M = phi(Xs, N).T @ w_prior[1] @ phi(X, N)
Q = phi(Xs, N).T @ w_prior[1] @ phi(Xs, N)
K = phi(X, N).T @ w_prior[1] @ phi(X, N)

y_mean = (M @ nl.inv(K + sigma ** 2 * np.eye(n_data)) @ y).squeeze()
y_std = np.sqrt((Q - M @ nl.inv(K + sigma ** 2 * np.eye(n_data)) @ M.T).diagonal())

# Plotting
plt.figure()

plt.scatter(X.squeeze(), y.squeeze(), marker='x', color='r', zorder=10)
plt.plot(Xs.squeeze(), y_mean.squeeze(), color='g')
plt.fill_between(Xs.squeeze(), y_mean.squeeze() - y_std, y_mean + y_std, alpha=0.2, color='b')
plt.xlim([-6,6])
plt.ylim([-2,2])

plt.xlabel(r'Input $x$')
plt.ylabel(r'Output $f(x)$')
plt.title(r'Bayesian Polynomial Model $f(x) = \phi(x)^T \mathbf{w}$')

plt.show()