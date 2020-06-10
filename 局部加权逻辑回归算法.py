import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(6)
class local_polynomial_logistic_regression():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self, test_x, test_y, h):
        y = self.y.reshape(-1, 1)
        n, feature = self.x.shape
        test_n = test_x.shape[0]
        pred = np.zeros((test_n, ), dtype=np.float)
        for one in range(test_n):
            cur_x = test_x[one]
            beta = np.ones((1, feature+1), dtype=np.float)
            t = (self.x - cur_x) / h
            c = np.ones((n, 1))
            X = np.hstack((c, self.x - cur_x))
            for k in range(100):
                prev = beta.copy()
                distance = np.sum(t**2, axis=1)/np.sqrt(2*np.pi)
                K = np.diag(distance)
                Z = 1 / (1 + np.exp(beta @ X.T))
                beta += (1 / n) * 0.01 * ((K @ y).T - distance.reshape(1, -1) * Z) @ X
            if 1 / (1 + beta[0][0]) >= 0.5:
                pred[one] = 0
            else:
                pred[one] = 1
            print(one)
        return pred


if __name__ == '__main__':
    x1 = np.random.randn(200, 10)
    x2 = np.sin(x1)
    x = np.vstack((x1, x2))
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    y1 = np.ones((200))
    y2 = np.zeros((200))
    y = np.hstack((y1, y2))
    train_X, test_X, train_y, test_y = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    print(test_y.shape)
    '''
    plt.plot(x1[:, 0], x1[:, 1], 'ro')
    plt.plot(x2[:, 0], x2[:, 1], 'bo')
    plt.show()
    '''
    model = local_polynomial_logistic_regression(train_X, train_y)
    acc = []
    h_list = np.arange(0.01, 0.02, 0.002)
    for h in h_list:
        pred = model.train(test_X, test_y, 0.9)
        acc.append(np.mean(pred==test_y))
    max_index = np.argmax(acc)
    print(h_list[max_index], np.max(acc), acc)
    lr = LogisticRegression(penalty='l2', C=5.0)
    lr.fit(train_X, train_y)
    pr = lr.predict(test_X)
    print(np.mean(pr == test_y))




