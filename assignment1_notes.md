http://cs231n.github.io/assignments2016/assignment1/

---

## kNN 分类器

kNN 分类器步骤：

1. 训练：不需要训练，简单地保存所有训练数据；
2. 测试：测试样本分别和所有训练样本计算距离，选取 k 个最近的训练样本的 label，投票选出预测值。

### 样本距离计算

第一个练习是计算两两样本之间的 L2 距离，逐步实现两层循环、一层循环、不用循环的方法：

```python
  def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        # dists[i, j] = np.linalg.norm((X[i] - self.X_train[j]))
    return dists
```

也可以用 `np.linalg.norm` 直接计算欧氏距离。


```python
  def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
    return dists
```

要减少循环，就要熟悉 NumPy 的 Broadcasting 机制：

> 对两个数组使用广播机制要遵守下列规则：
>   1. 如果数组的秩不同，使用1来将秩较小的数组进行扩展，直到两个数组的尺寸的长度都一样。
>   2. 如果两个数组在某个维度上的长度是一样的，或者其中一个数组在该维度上长度为1，那么我们就说这两个数组在该维度上是相容的。
>   3. 如果两个数组在所有维度上都是相容的，他们就能使用广播。

不用循环的方法不太好想，可以先把 dist 矩阵中的某个元素具体形式写出来。

```python
  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    dists = -2 * np.dot(X, self.X_train.T) + np.sum(self.X_train**2, axis=1) + np.sum(X**2, axis=1, keepdims=True)
    return np.sqrt(dists)

```

可以参考这个链接：

https://stackoverflow.com/questions/32856726/memory-efficient-l2-norm-using-python-broadcasting

```python
m = x.shape[0] # x has shape (m, d)
n = y.shape[0] # y has shape (n, d)
x2 = np.sum(x**2, axis=1).reshape((m, 1))
y2 = np.sum(y**2, axis=1).reshape((1, n))
xy = x.dot(y.T) # shape is (m, n)
dists = np.sqrt(x2 + y2 - 2*xy) # shape is (m, n)
```

计算完距离矩阵后，可视化，有个问题是：为什么有些列、行几乎都是白色的？

> 某一行特别亮表示这一行所代表的那个测试样本与5000个训练样本的距离都比较远，说明这个样本与所有的训练数据都不太像；相反，如果一列比较亮，则说明训练样本里面的这张图片与500个测试图片都不太像。

> 还有就是这些行、列对应的图像可能比较暗/亮（比如全黑或全白），一个观察是白色的行列相交时有些更亮了（全黑与全白），有些变黑了（都是全黑或都是全白）

使用距离矩阵来预测 label:

```python
  def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []

      min_k_dist_indices = dists[i].argsort()[0:k]
      closest_y = self.y_train[min_k_dist_indices]
      counts = np.bincount(closest_y)
      y_pred[i] = np.argmax(counts)
    return y_pred

```

- `np.argsort()` 返回排序后的下标 list
- `np.bincount()` 统计数组中元素出现的频数，结合 `np.argmax()` 实现投票

### 交叉验证（确定 knn 的超参数 k）

不同的 k ，预测的准确率也不同，如何选取 k ？可以利用 cross validation。

把训练集划分为若干份，每次取一份作为测试集；对事先选定的一些 k 分别进行 训练、测试 。最后选表现最好的 k 。

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

# 先把训练数据均分成若干份
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

# 对每个 k 值进行交叉验证
for k in k_choices:
  k_to_accuracies[k] = []
  for i in xrange(num_folds):
    X_train_cur = np.concatenate([fold for idx, fold in enumerate(X_train_folds)\
                                  if idx != i])
    y_train_cur = np.concatenate([fold for idx, fold in enumerate(y_train_folds)\
                                  if idx != i])
    X_val_cur = X_train_folds[i]
    y_val_cur = y_train_folds[i]

    knn = KNearestNeighbor()
    knn.train(X_train_cur, y_train_cur)
    y_val_pred = knn.predict(X_val_cur, k)
    num_correct = np.sum(y_val_cur == y_val_pred)
    k_to_accuracies[k].append(float(num_correct) / float(len(y_val_cur)))


# Print out the computed accuracies
for k in sorted(k_to_accuracies):
  for accuracy in k_to_accuracies[k]:
    print 'k = %d, accuracy = %f' % (k, accuracy)
```

最后我们取平均 accuracy 最好的 k。

---

