## 欧几里得距离：

两点间的（即直线）的距离

在欧几里得空间中，点 $x = (x_1, ..., x_n)$ 和 $y = (y_1,...,y_n)$ 之间的欧式距离为
$$
d(x,y) := \sqrt {(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2}
\\ = \sqrt {\sum_{i=1}^{n}(x_i - y_i)^2}
$$


两点之间线段最短



## 有几个需要考虑的问题

1. 把一个物体表示成向量
   1. 这也叫“特征工程” （Feature Engineering)
   2. 模型的输入一定是数量化的信息，我们需要把现实生活中的物体表示成 向量/矩阵/张量 形式
2. 标记好每一个物体的标签
3. 计算两个物体之间的距离/相似度
   1. 欧式距离
4. 选择合适的 K
   1. 决策边界（比如满分100分，60分以上为及格， 60分以下为不及格。（60分为决策边界）

### 交叉验证

把训练数据进一步分成训练集（Training Data) 和验证集（Validation Data)

选择在验证数据里最好的超参数组合，求出多个验证数据的准确率，然后求出平均值准确率，

k-fold Cross Validation

注意事项：

1. 交叉验证中，不能拿测试数据来进行训练, 调参，会导致过拟合。
2. 数据量越少，可以适当增加折数。



## KNN延伸

- 如何处理大数据量

  1. 预测阶段：需要计算**被预测样本** 和 **每一个训练样本** 之间的距离
  2. 时间复杂度： O(N), N 是样本总数
  3. 当 N 很大的时候，这个复杂度显然不能作为实时预测？？

  如何解决？

  1. KD-tree 树

     1. 对于样本格式 N 来说 O(logN) , 树的时间复杂度一般是 O(log(n))
     2. 对于数据维度 D （二维， 三维）来说， 指数复杂度。

  2. 利用类似哈希算法--- Locality Sensitivity Hashing   (**LSH**)

     

- 如何处理特征之间的相关性？



- 如何处理样本的重要性？

  - 权重计算

  

- 能不能利用 Kernel Trick ?



延伸阅读推荐：

- KD-tree: https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf
- Large Margin Nearest Neighbor: http://jmlr.org/papers/volume10/weinberger09a/weinberger09a.pdf
- Latent Semantic Hashing: [Book]Mining of Massive Dataset. Chapter 3.  Leskovec-Rajaraman-Ullman



## 总结

1. KNN 是一个非常简单的算法
2. 比较适合应用在低维空间
3. 预测的时候复杂度高，对于大数据需要处理（KD-tree, LSH)