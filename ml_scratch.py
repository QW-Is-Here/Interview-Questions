import numpy as np

# ---------------- Linear Regression (with Gradient Descent) ----------------
class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iter):
            y_pred = X @ self.W + self.b
            error = y_pred - y
            # MSE loss
            loss = np.mean(error ** 2)
            # Gradients
            dW = 2 * X.T @ error / n_samples
            db = 2 * np.mean(error)
            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.W + self.b

# ---------------- Logistic Regression (with Gradient Descent & BCE Loss) ----------------
class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iter):
            z = X @ self.W + self.b
            y_hat = 1 / (1 + np.exp(-z))
            # BCE loss: -[y*log(y_hat)+(1-y)*log(1-y_hat)]
            error = y_hat - y
            dW = X.T @ error / n_samples
            db = np.mean(error)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_proba(self, X):
        X = np.asarray(X)
        z = X @ self.W + self.b
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

# ---------------- KNN (using distance + broadcasting for tensorization) ----------------
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
    
    def predict(self, X):
        X = np.asarray(X)
        # Broadcasting for L2 distance: (test_N, feat) - (train_N, feat) --> (test_N, train_N, feat)
        dists = np.sqrt(((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        # for each test sample, find the k nearest neighbors
        idx_knn = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        y_knn = self.y_train[idx_knn]
        # majority vote (assumes y is int/label)
        from scipy.stats import mode
        pred, _ = mode(y_knn, axis=1, keepdims=False)
        return pred

# ---------------- K-Means (centroid update, fully vectorized) ----------------
class KMeans:
    def __init__(self, n_clusters=3, n_iter=100):
        self.n_clusters = n_clusters
        self.n_iter = n_iter

    def fit(self, X):
        X = np.asarray(X)
        N, F = X.shape
        # Randomly initialize centroids
        rng = np.random.default_rng()
        indices = rng.choice(N, self.n_clusters, replace=False)
        centroids = X[indices]
        for _ in range(self.n_iter):
            # Distance: (N, K)
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            # Update centroids: for each k, mean over all X[labels==k]
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])
            if np.allclose(centroids, new_centroids):  # early stop
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels_ = labels

    def predict(self, X):
        X = np.asarray(X)
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

# ---------------- Softmax + Cross Entropy ----------------
def softmax(logits):
    logits = np.asarray(logits)
    # for stability
    logits_max = np.max(logits, axis=1, keepdims=True)
    e_x = np.exp(logits - logits_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred_probs):
    # y_true: (N,) int or one-hot, y_pred_probs: (N, C)
    y_pred_probs = np.asarray(y_pred_probs)
    N = y_pred_probs.shape[0]
    if y_true.ndim == 1:
        # indices, not one-hot
        return -np.mean(np.log(y_pred_probs[np.arange(N), y_true] + 1e-12))
    else:
        return -np.mean(np.sum(y_true * np.log(y_pred_probs + 1e-12), axis=1))

#%%
import numpy as np

def linear_regression_2d(X, y, lr=0.01, steps=1000):
    """
    X: shape (N, 2)
    y: shape (N,)
    """
    N = X.shape[0]

    # parameters
    w = np.zeros(2)
    b = 0.0

    for _ in range(steps):
        # forward
        y_hat = X @ w + b        # (N,)

        # error
        err = y_hat - y          # (N,)

        # gradients
        dw = (2 / N) * (X.T @ err)   # (2,)
        db = (2 / N) * np.sum(err)

        # update
        w -= lr * dw
        b -= lr * db

    return w, b

#%%
from collections import Counter
import numpy as np

class KNNClassifier:
    def __init__(self, k=3, distance='euclidean', p=2):
        self.k = k
        self.distance = distance 
        self.p = p 

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            if self.distance == 'euclidean':
                distances = np.linalg.norm(self.X_train - x, axis=1)
            elif self.distance == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            elif self.distance == 'cosine':
                dot = np.dot(self.X_train, x)
                norm_x = np.linalg.norm(x)
                norm_X_train = np.linalg.norm(self.X_train, axis=1)
                distances = 1 - dot / (norm_X_train * norm_x)
            elif self.distance == 'minkowski':  # 新增
                distances = np.power(
                    np.sum(np.power(np.abs(self.X_train - x), self.p), axis=1),
                    1/self.p
                )

            indices = np.argsort(distances)[:self.k]
            y_pred.append(Counter(self.y_train[indices]).most_common(1)[0][0])
        return np.array(y_pred) 

#%% k means
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X):
        n_samples = X.shape[0]
        
        # 1. 随机初始化中心
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centers = X[idx]
        
        for _ in range(self.max_iters):
            # 2. 分配：每个点到最近中心
            distances = np.linalg.norm(X[:, None] - self.centers, axis=2)  # (N, K)
            self.labels = np.argmin(distances, axis=1)  # (N,)
            
            # 3. 更新：计算新中心
            new_centers = np.array([
                X[self.labels == i].mean(axis=0) for i in range(self.k)
            ])
            
            # 4. 检查收敛
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers
        
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
        return np.argmin(distances, axis=1)


#%% K-means++ (优化初始化)
class KMeansPlusPlus:
    """
    K-means++ 算法
    
    核心改进：智能选择初始中心点
    - 第一个中心：随机选择
    - 后续中心：以概率选择，距离已有中心越远的点被选中概率越高
    
    优点：
    1. 避免初始中心聚集在一起
    2. 加快收敛速度
    3. 得到更好的聚类结果
    4. 减少对初始化的敏感性
    
    时间复杂度：O(k * n * d) 初始化 + O(i * k * n * d) 迭代
    空间复杂度：O(k * d + n)
    """
    
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
    
    def _init_centers_plus_plus(self, X):
        """
        K-means++ 初始化策略
        
        算法步骤：
        1. 随机选择第一个中心
        2. 对于每个后续中心：
           - 计算每个点到最近已选中心的距离
           - 以距离的平方作为概率权重
           - 按概率随机选择下一个中心
        """
        n_samples = X.shape[0]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 步骤1: 随机选择第一个中心
        first_idx = np.random.randint(n_samples)
        centers = [X[first_idx]]
        
        # 步骤2: 选择剩余的 k-1 个中心
        for _ in range(self.k - 1):
            # 计算每个点到最近中心的距离
            distances = np.min([
                np.linalg.norm(X - center, axis=1) 
                for center in centers
            ], axis=0)
            
            # 距离的平方作为概率权重
            # 距离越远，被选中的概率越高
            probabilities = distances ** 2
            probabilities /= probabilities.sum()  # 归一化
            
            # 按概率选择下一个中心
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers.append(X[next_idx])
        
        return np.array(centers)
    
    def fit(self, X):
        """训练 K-means++ 模型"""
        # 1. K-means++ 初始化中心（关键改进！）
        self.centers = self._init_centers_plus_plus(X)
        
        # 2. 标准 K-means 迭代
        for iteration in range(self.max_iters):
            # 2.1 分配：每个点到最近中心
            distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            # 2.2 更新：计算新中心
            new_centers = np.array([
                X[self.labels == i].mean(axis=0) 
                for i in range(self.k)
            ])
            
            # 2.3 检查收敛
            if np.allclose(self.centers, new_centers):
                print(f"收敛于第 {iteration + 1} 次迭代")
                break
            
            self.centers = new_centers
        
        return self
    
    def predict(self, X):
        """预测新数据点的簇标签"""
        distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """训练并返回标签"""
        self.fit(X)
        return self.labels


#%% K-means++ 优化版（更高效的实现）
class KMeansPlusPlusOptimized:
    """
    K-means++ 优化实现
    使用向量化操作提高效率
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # 收敛阈值
        self.random_state = random_state
        self.inertia_ = None  # 簇内误差平方和
        self.n_iter_ = 0  # 实际迭代次数
    
    def _init_centers_plus_plus(self, X):
        """优化的 K-means++ 初始化"""
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化中心数组
        centers = np.empty((self.k, n_features))
        
        # 第一个中心：随机选择
        center_id = np.random.randint(n_samples)
        centers[0] = X[center_id]
        
        # 初始化距离数组（每个点到最近中心的距离）
        closest_dist_sq = np.linalg.norm(X - centers[0], axis=1) ** 2
        
        # 选择剩余的中心
        for c in range(1, self.k):
            # 按距离平方的概率选择
            probabilities = closest_dist_sq / closest_dist_sq.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            # 二分查找选中的点
            center_id = np.searchsorted(cumulative_probs, r)
            centers[c] = X[center_id]
            
            # 更新最近距离
            new_dist_sq = np.linalg.norm(X - centers[c], axis=1) ** 2
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
        
        return centers
    
    def _compute_inertia(self, X):
        """计算簇内误差平方和（用于评估聚类质量）"""
        distances = np.linalg.norm(X - self.centers[self.labels], axis=1)
        return np.sum(distances ** 2)
    
    def fit(self, X):
        """训练模型"""
        n_samples = X.shape[0]
        
        # K-means++ 初始化
        self.centers = self._init_centers_plus_plus(X)
        
        # 迭代优化
        for iteration in range(self.max_iters):
            # 分配
            distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            # 更新中心
            new_centers = np.array([
                X[self.labels == i].mean(axis=0) if np.any(self.labels == i)
                else self.centers[i]  # 如果簇为空，保持原中心
                for i in range(self.k)
            ])
            
            # 检查收敛（中心移动距离 < 阈值）
            center_shift = np.linalg.norm(new_centers - self.centers, axis=1).sum()
            self.centers = new_centers
            
            if center_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iters
        
        # 计算最终的 inertia
        self.inertia_ = self._compute_inertia(X)
        
        return self
    
    def predict(self, X):
        """预测"""
        distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """训练并预测"""
        self.fit(X)
        return self.labels
    
    def score(self, X):
        """返回负的 inertia（用于模型选择，越大越好）"""
        labels = self.predict(X)
        distances = np.linalg.norm(X - self.centers[labels], axis=1)
        return -np.sum(distances ** 2)


#%% K-means vs K-means++ 对比示例
if __name__ == "__main__":
    """
    演示 K-means++ 相比标准 K-means 的优势
    """
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    
    print("=" * 60)
    print("K-means vs K-means++ 对比")
    print("=" * 60)
    
    # 生成测试数据
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                           cluster_std=0.6, random_state=42)
    
    k = 4
    n_trials = 10  # 运行多次取平均
    
    # 对比标准 K-means 和 K-means++
    kmeans_inertias = []
    kmeanspp_inertias = []
    kmeans_iters = []
    kmeanspp_iters = []
    
    for trial in range(n_trials):
        # 标准 K-means
        kmeans = KMeans(k=k, max_iters=100)
        kmeans.fit(X)
        distances = np.min([
            np.linalg.norm(X - center, axis=1) 
            for center in kmeans.centers
        ], axis=0)
        kmeans_inertias.append(np.sum(distances ** 2))
        
        # K-means++
        kmeanspp = KMeansPlusPlusOptimized(k=k, max_iters=100, random_state=trial)
        kmeanspp.fit(X)
        kmeanspp_inertias.append(kmeanspp.inertia_)
        kmeanspp_iters.append(kmeanspp.n_iter_)
    
    print(f"\n运行 {n_trials} 次的统计结果：")
    print("-" * 60)
    print(f"标准 K-means:")
    print(f"  平均 Inertia: {np.mean(kmeans_inertias):.2f} ± {np.std(kmeans_inertias):.2f}")
    print(f"  最好 Inertia: {np.min(kmeans_inertias):.2f}")
    print(f"  最差 Inertia: {np.max(kmeans_inertias):.2f}")
    
    print(f"\nK-means++:")
    print(f"  平均 Inertia: {np.mean(kmeanspp_inertias):.2f} ± {np.std(kmeanspp_inertias):.2f}")
    print(f"  最好 Inertia: {np.min(kmeanspp_inertias):.2f}")
    print(f"  最差 Inertia: {np.max(kmeanspp_inertias):.2f}")
    print(f"  平均迭代次数: {np.mean(kmeanspp_iters):.1f}")
    
    print(f"\n改进：")
    print(f"  Inertia 降低: {(1 - np.mean(kmeanspp_inertias)/np.mean(kmeans_inertias)) * 100:.1f}%")
    print(f"  稳定性提升: 标准差降低 {(1 - np.std(kmeanspp_inertias)/np.std(kmeans_inertias)) * 100:.1f}%")
    print("=" * 60)
    
    # 可视化一次运行的结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 标准 K-means
    kmeans = KMeans(k=k, max_iters=100)
    kmeans.fit(X)
    axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', alpha=0.6)
    axes[0].scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0].set_title('标准 K-means')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # K-means++
    kmeanspp = KMeansPlusPlusOptimized(k=k, max_iters=100, random_state=42)
    kmeanspp.fit(X)
    axes[1].scatter(X[:, 0], X[:, 1], c=kmeanspp.labels, cmap='viridis', alpha=0.6)
    axes[1].scatter(kmeanspp.centers[:, 0], kmeanspp.centers[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[1].set_title(f'K-means++ (收敛于 {kmeanspp.n_iter_} 次迭代)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('kmeans_comparison.png', dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存为 kmeans_comparison.png")
    plt.show()

#%% logistic regression
import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # initialize weights and bias to zeros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent optimization
        for i in range(self.n_iters):
            # calculate predicted probabilities and cost
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            cost = (-1 / n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
            # calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        # calculate predicted probabilities
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        # convert probabilities to binary predictions
        return np.round(y_pred).astype(int)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# create sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# initialize logistic regression model
lr = LogisticRegression()

# train model on sample dataset
lr.fit(X, y)

# make predictions on new data
X_new = np.array([[6, 7], [7, 8]])
y_pred = lr.predict(X_new)

print(y_pred)  # [1, 1]
# %%
import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iters=1000, regularization='l2', reg_strength=0.1, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        n_batches = n_samples // self.batch_size
        for i in range(self.n_iters):
            batch_indices = np.random.choice(n_samples, self.batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            z = np.dot(X_batch, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            cost = (-1 / self.batch_size) * np.sum(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
            if self.regularization == 'l2':
                reg_cost = (self.reg_strength / (2 * n_samples)) * np.sum(self.weights ** 2)
                cost += reg_cost
            elif self.regularization == 'l1':
                reg_cost = (self.reg_strength / (2 * n_samples)) * np.sum(np.abs(self.weights))
                cost += reg_cost
            dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1 / self.batch_size) * np.sum(y_pred - y_batch)
            if self.regularization == 'l2':
                dw += (self.reg_strength / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.reg_strength / n_samples) * np.sign(self.weights)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return np.round(y_pred).astype(int)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
#%%
#%% knn classifier
from collections import Counter
import numpy as np

class KNNClassifier:
    def __init__(self, k=3, distance='euclidean', p=2):
        self.k = k
        self.distance = distance 
        self.p = p 

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            if self.distance == 'euclidean':
                distances = np.linalg.norm(self.X_train - x, axis=1)
            elif self.distance == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x), axis=1)
            elif self.distance == 'cosine':
                dot = np.dot(self.X_train, x)
                norm_x = np.linalg.norm(x)
                norm_X_train = np.linalg.norm(self.X_train, axis=1)
                distances = 1 - dot / (norm_X_train * norm_x)
            elif self.distance == 'minkowski':  # 新增
                distances = np.power(
                    np.sum(np.power(np.abs(self.X_train - x), self.p), axis=1),
                    1/self.p
                )

            indices = np.argsort(distances)[:self.k]
            y_pred.append(Counter(self.y_train[indices]).most_common(1)[0][0])
        return np.array(y_pred)

#%% singleton 
import threading

class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

##% kmeans clustering
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X - self.centroids[:, np.newaxis], axis=2)
            labels = np.argmin(distances, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X - self.centroids[:, np.newaxis], axis=2)
        return np.argmin(distances, axis=0)

##% LRU cache
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.queue = deque()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.queue.remove(key)
        self.cache[key] = value
        self.queue.append(key)
        if len(self.cache) > self.capacity:
            oldest = self.queue.popleft()
            del self.cache[oldest] 

##% cross entropy loss
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))
    
    def backward(self, y_true, y_pred):
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
