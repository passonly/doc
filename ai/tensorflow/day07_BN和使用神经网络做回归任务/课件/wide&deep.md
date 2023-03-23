# wide and deep模型

# 1.背景

Wide and deep 模型是 TensorFlow 在 2016 年 6 月左右发布的一类用于分类和回归的模型，并应用到了 Google Play 的应用推荐中。wide and deep 模型的核心思想是结合线性模型的记忆能力（memorization）和 DNN 模型的泛化能力（generalization），在训练过程中同时优化 2 个模型的参数，从而达到整体模型的预测能力最优。

记忆（memorization）即从历史数据中发现item或者特征之间的相关性。

泛化（generalization）即相关性的传递，发现在历史数据中很少或者没有出现的新的特征组合。

[原论文]([https://arxiv.org/pdf/1606.07792.pdf](https://arxiv.org/pdf/1606.07792.pdf))

# 2.原理

## 2.1 网络结构

![](//upload-images.jianshu.io/upload_images/1500965-e5896703dd49b541.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)![1500965e5896703dd49b541.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1644577462000/2738f2436e8d4a7494384b8e91333421.png)

可以认为：WideDeep = LR + DNN

# 3. 稀疏特征

- 离散值特征: 只能从N个值中选择一个
  - 比如性别, 只能是男女
  - one-hot编码表示的离散特征, 我们就认为是稀疏特征.
  - Eg: 专业= {计算机, 人文, 其他}, 人文 = [0, 1, 0]
  - Eg: 词表 = {人工智能,深度学习,你, 我, 他, 马士兵, ..} 他= [0, 0, 0, 0, 1, 0, ...]
  - 叉乘 = {(计算机, 人工智能), (计算机, 你)...}
  - 叉乘可以用来精确刻画样本, 实现记忆效果.
  - 优点:
    - 有效, 广泛用于工业界, 比如广告点击率预估(谷歌, 百度的主要业务), 推荐算法.
  - 缺点:
    - 需要人工设计.
    - 叉乘过度, 可能过拟合, 所有特征都叉乘, 相当于记住了每一个样本.
    - 泛化能力差, 没出现过就不会起效果
- 密集特征
  - 向量表达
    - Eg: 词表 = {人工智能, 你, 我, 他, 马士兵}
    - 他 = [0.3, 0.2, 0.6, ...(n维向量)]
    - 每个词都可以用一个密集向量表示, 那么词和词之间就可以计算距离.
  - Word2vec工具可以方便的将词语转化为向量.
    - 男 - 女 = 国王  - 王后
  - 优点:
    - 带有语义信息, 不同向量之间有相关性.
    - 兼容没有出现过的特征组合.
    - 更少人工参与
  - 缺点:
    - 过度泛化, 比如推荐不怎么相关的产品.![150096513fa11d119bb20b7.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1644577462000/cdfda2e841be46f38a0607010ad10524.png)