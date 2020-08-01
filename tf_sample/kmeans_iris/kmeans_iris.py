# -*- coding: utf-8 -*-
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 加载数据集，是一个字典类似Java中的map
lris_df = datasets.load_iris()

# 挑选出前两个维度作为x轴和y轴，你也可以选择其他维度
x_axis = lris_df.data[:, 0]  # 花萼长度
y_axis = lris_df.data[:, 2]  # 花萼宽度

print('lris_df', lris_df.data.shape)
print('target', lris_df.target.shape)
print('x_axis', x_axis.shape)
print('y_axis', y_axis.shape)

# c指定点的颜色，当c赋值为数值时，会根据值的不同自动着色
plt.scatter(x_axis, y_axis, c=lris_df.target)
plt.savefig('iris-' + 'sample' + '.png')
plt.show()

# 这里已经知道了分3类，其他分类这里的参数需要调试
model = KMeans(n_clusters=3)
model.fit(lris_df.data)

# 用index=100的那条数据预测
flower_one = lris_df.data[100].reshape(1, -1)
prddicted_label = model.predict(flower_one)
print('flower_one', flower_one)
print('prddicted_label', prddicted_label)

# 预测全部150条数据
all_predictions = model.predict(lris_df.data)
print('all_predictions', all_predictions.shape)

# 打印出来对150条数据的聚类散点图
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.savefig('iris-' + 'predict' + '.png')
plt.show()
