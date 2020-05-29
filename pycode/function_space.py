import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy.linalg as nl


#noise-free
def cov(X_1,X_2):
    return np.exp(- (X_1 - X_2.T) ** 2 / 2)

#训练设置
n_data = 4
sigma = 0.25
# Training data
X = np.random.random(size=[n_data,1]) * 10 - 5
y = np.random.random(size=[n_data,1]) * 3 - 1.5

#先验和后验分布参数
#测试数据
Xs = np.linspace(-6,6,100).reshape(100,1)
#0均值，1协方差
prior = (np.zeros(100), cov(Xs, Xs))
posterior = ((cov(Xs, X) @ nl.inv(cov(X, X)) @ y).squeeze(), cov(Xs, Xs) - cov(Xs, X) @ nl.inv(cov(X, X)) @ cov(X, Xs))

# Plotting
n_samples = 3
#先验信息，采样
prior_rv = multivariate_normal(prior[0], prior[1], allow_singular=True)
prior_samples = prior_rv.rvs(n_samples).T
#后验
posterior_rv = multivariate_normal(posterior[0], posterior[1], allow_singular=True)
posterior_samples = posterior_rv.rvs(n_samples).T
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
# Parameter inference for each instance
y_mean1 = posterior[0].squeeze()
y_std1 = np.sqrt((posterior[1]).diagonal())

# Plotting
plt.figure()

plt.scatter(X.squeeze(), y.squeeze(), marker='x', color='r', zorder=10)
plt.plot(Xs.squeeze(), y_mean1.squeeze(), color='g')
plt.fill_between(Xs.squeeze(), y_mean1.squeeze() - y_std1, y_mean1 + y_std1, alpha=0.2, color='b')
plt.xlim([-6,6])
plt.ylim([-2,2])

plt.xlabel(r'Input $x$')
plt.ylabel(r'Output $f(x)$')
plt.title(r'Regression with Noise-free Observations')

plt.show()


#加噪声模型的预测
# Prior and Posterior distribution parameters
posterior = ((cov(Xs, X) @ nl.inv(cov(X, X) + sigma ** 2 * np.eye(n_data)) @ y).squeeze(),
             cov(Xs, Xs) - cov(Xs, X) @ nl.inv(cov(X, X) + sigma ** 2 * np.eye(n_data)) @ cov(X, Xs))
#参数推断
y_mean2 = posterior[0].squeeze()
y_std2 = np.sqrt((posterior[1]).diagonal())

# Plotting
plt.figure()

plt.scatter(X.squeeze(), y.squeeze(), marker='x', color='r', zorder=10)
plt.plot(Xs.squeeze(), y_mean2.squeeze(), color='g')
plt.fill_between(Xs.squeeze(), y_mean2.squeeze() - y_std2, y_mean2 + y_std2, alpha=0.2, color='b')
plt.xlim([-6,6])
plt.ylim([-2,2])

plt.xlabel(r'Input $x$')
plt.ylabel(r'Output $f(x)$')
plt.title(r'Regression with Noisy Observations')

plt.show()