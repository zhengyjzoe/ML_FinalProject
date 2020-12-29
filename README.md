# 机器学习课程报告——代码解释

### 文件介绍

代码中共有7个文件夹，其中包括：

1. data：存储原始图片，提取到的图片特征，使用Lasso降维后的数据，Resnet每一层的可视化（由于上传大小有限制，还请老师从邮箱中下载后再将该文件放入根目录）
2. evaluate：对训练得到的模型结果评估与结果可视化
3. models：含有BPNN，KNN，MLP，RF四个模型文件以及tools文件，内含pca实现以及混淆矩阵的计算与可视化
4. preprocessing：提取图片特征，预处理数据，Lasso降维实现，可视化Resnet
5. save：各种结果的保存
6. train_code：模型训练的主文件
7. train_lasso：求解Lasso问题的算法文件，同时存有Lasso的结果图片

### 代码运行

**以下复现过程中很可能覆盖已经存好的结果，如需复现请comment存储代码**

**参数均在文件开头或`__main__`中大写定义**

**报告仅使用训练集与验证集(将验证集视为测试集)数据，当需要进一步研究参数对模型准确率影响时需要用到第三个数据集，故此处留出这部分数据。**

#### 特征提取

终端进入preprocessing文件夹，运行

```
python3 generate_embeddings.py '../data/image/seg_train' -b 16 --res r50  --out ../data/feature_map
```

其中`r50`为所使用的Resnet，考虑到本项目并不追求不同提取方式对模型准确度的影响，仅使用r50对特征提取。

提取到的特征将存在`data/feature_map`中。

#### Lasso降维

**Step1**:进入`train_Lasso`，运行`fast_proximal.py`,`fast_smooth.py`文件，将在`save`中存储回归系数，并在`Plots`中存储训练曲线；

**Step2**: 运行`view_coef.py`，将生成系数值的KDE图；

**Step3**:运行`preprocessing/Lasso.py`，将在`data/data_reduction`中存储降维后的数据；

#### 模型训练

进入`train_code/train_my_model.py`。

**调整文件上方的参数**，由于`SVM`,`RF`运行效率较低，复现时建议comment模型保存语句。

模型训练中保存了`SVM`,`RF`的模型文件，并保存所有模型的预测结果，均在`save`对应的文件夹中，可根据名称找到。

#### 模型评估

进入`evaluate/evaluate.py`，调整参数后即可得到对应的结果。
