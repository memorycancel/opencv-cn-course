# tensorflow深度学习训练营

## 01 神经网络入门

What and Why 神经网络入门视频

<iframe width="1080" height="608" src="https://www.youtube.com/embed/_5XYLA2HLmo" title="Neural Networks - What They Are &amp; Why They Matter - A 30,000 Feet View for Beginners" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

 ![download1.png](01_getting_started_with_neural_networks/download1.png)

本教程面向希望进入机器学习和深度学习领域的绝对初学者。我们将简化许多细节，以便您能够掌握最基本的概念。

### 01-00 目录

Table of Contents

1. 理解神经网络为黑匣子
2. 理解神经网络的输出
3. 理解神经网络的输入
4. 如何理解神经网络训练？

### 01-02 理解神经网络为黑匣子

我们首先将神经网络视为黑匣子；你不知道里面有什么，但正如你在这个例子中看到的，我们有一个任意大小、格式或颜色的输入图像，网络的输出是 0 到 1 之间的三个数字，其中每个输出对应是每种类别的概率：输入图像可以是“猫”、“狗”或其他类别（我们简称为“其他”）。

### 01-03 理解神经网络的输出

我们通常将这些类别称为标签（Labels）或类标签（Class Labels）。这个特殊问题称为图像分类（image classification），其中输入是图像，输出是三个可能类别中每一个类别的可能性数值。需要明确的是网络的输出是三个数值（而不是标签本身）。

在此示例中，网络为第一个输出 0.97，为第二个输出 0.01，为第三个输出 0.02。请注意，三个输出之和为 1，因为它们代表概率。由于第一个输出的概率最高，我们说网络预测输入图像是猫。

一般，分配给输入图像的标签是通过从三个输出中选择与最大概率相关联的标签来计算的。因此，如果输出分别为 0.51、0.48 和 0.01，我们仍会将预测标签分配为 Cat，因为 0.51  仍然代表所有三个类别的最高概率。在这种情况下，网络对预测的信心较低。

如果输入图像是猫，完美神经网络将输出  (1,0,0)；如果输入图像是狗，则输出 (0,1,0)；如果输入图像是猫或狗以外的东西，则最终输出  (0,0,1)。事实上，即使训练有素的网络也无法给出如此完美的结果。实际上，执行图像分类的神经网络可能有数百个可能的类别（不仅仅是三个），但分配类标签的过程是相同的。请记住，神经网络可用于许多其他问题类型，但图像分类是一种非常常见的应用，非常适合作为入门级示例。

### 01-04 理解神经网络的输入

现在让我们看一下神经网络的输入，并考虑如何将这些信息表示为数值。您可能已经知道，灰度图像表示为像素值数组，其中每个像素值代表从纯黑到纯白的强度。

 ![download2.png](01_getting_started_with_neural_networks/download2.png)

彩色图像非常相似，只不过它们的每个像素具有三个分量，分别代表红色、绿色和蓝色的颜色强度。因此，在本例中，256 x 256 彩色图像由 196,608 个数字表示。考虑到这一点，让我们更新我们的图表，以更清楚地反映幕后发生的情况。

 ![download3.png](01_getting_started_with_neural_networks/download3.png)

在这里，我们展示了神经网络期望输入总共有近 200,000  个数字，但我们尚未指定该数据的形状。根据网络的类型，数据可以表示为一维向量；或更紧凑的形式：例如三个二维数组，其中每个数组为  256×256。但无论哪种情况，特定的网络设计都期望数据具有固定的大小和形状。

 ![](01_getting_started_with_neural_networks/download4.png)

这里要注意的是当设计神经网络时，它们为了接受特定大小和形状的输入而完成的。不同的图像分类网络根据其设计支持的应用程序需要不同大小的输入，这并不罕见。例如，由于与移动设备相关的资源有限，为移动设备设计的网络通常需要较小的输入图像。但这没关系，因为我们需要做的就是预处理图像以符合任何特定网络所需的尺寸和形状。

 ![](01_getting_started_with_neural_networks/download5.png)

### 01-05 如何理解神经网络训练？

现在让我们来谈谈如何理解神经网络的训练。关于神经网络要了解的主要内容是它们包含许多可调参数，您可以将其理解为黑匣子上的旋钮设置（在技术术语中，这些旋钮设置称为权重weights）。如果你有这样一个黑匣子，但不知道正确的旋钮设置，它基本上是没有用的，但好消息是，你可以通过有条不紊地训练神经网络来找到正确的设置。

 ![](01_getting_started_with_neural_networks/download6.png) 

训练过程类似于幼儿如何了解周围的世界。在日常生活中，孩子吸收大量的视觉信息，并通过反复试验，在父母的帮助下学会识别世界上的物体。训练神经网络来执行图像分类非常相似。它通常需要大量数据并需要多次迭代才能确定神经网络权重的最佳设置。

当你训练神经网络时，你需要向它展示你希望它学习的各种类别的数千个示例，例如猫的图像、狗的图像以及其他类型物体的图像。这种训练称为监督学习（**supervised learning**），因为您向神经网络提供一个类的图像，并明确告诉图像的类别。

 ![](01_getting_started_with_neural_networks/download7.png) 

下面附上一张，监督学习和非监督学习的图：

 ![supervisor_and_unsuper.jpeg](01_getting_started_with_neural_networks/supervisor_and_unsuper.jpeg) 

如果网络做出错误的预测，我们会计算与错误预测相关的误差，并且该误差用于调整网络中的权重，从而提高后续预测的准确性。

 ![](01_getting_started_with_neural_networks/download8.png) 

在下一个单元中，我们将更深入地研究神经网络的训练方式，包括如何对标记的训练数据进行建模、如何使用损失函数（loss functions ）以及用于更新神经网络权重的技术（称为梯度下降**gradient descent**）。



## 02 训练神经网络的基础知识

<iframe width="899" height="506" src="https://www.youtube.com/embed/4E2_rkP3owI" title="Deep Learning Using Keras – Training Neural Network" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

使用**Keras**进行深度学习，训练神经网络。

Keras是什么？ Keras是**一种用Python编写的高级神经网络应用程序编程接口（API）**，它是开源的。 它建立在CNTK、TensorFlow和Theano等框架之上，旨在通过深度神经网络实现快速实验。 Keras优先考虑其代码的灵活性和适应性，它不处理较低级别的计算，而是将其分配给后端库。

 ![](02_fundamentals_of_training_a_neural_network/download1.png)

在本单元中，我们将介绍针对图像分类问题训练神经网络所需的基本要素。我们仍然将内部网络架构视为黑匣子，以便我们可以专注于训练神经网络所需的其他基本组件和概念。

### 02-00 目录

1. 简介
2. 标记训练数据和 One-Hot 编码
3. 损失函数 Loss functions
4. 梯度下降 Gradient Descent（优化Optimizations）
5. 权重(Weights)更新计算示例
6. 完整的训练循环
7. 训练图(Plots)
8. 使用经过训练的模型进行推理(Inference)
9. 结论

### 02-01 简介

在上一单元中，我们介绍了神经网络的大概全貌，主要关注输入和输出以及如何解释图像分类问题的结果。我们还了解到，神经网络包含必须通过训练过程进行适当调整的权重。在这篇文章中，我们将更深入地研究神经网络的训练方式，而不涉及特定网络架构的细节。这将使我们能够在概念层面上理解模型训练过程。

1. 如何对标记过的（labeled）训练数据进行建模。
2. 如何使用损失函数来量化输入和预测输出之间的误差。
3. 如何使用梯度下降来更新网络中的权重。

### 02-02 标注训练数据和 One-Hot 编码

让我们仔细看看图像分类任务中标记的训练数据是如何表示的。带标签的训练数据由图像及其相应的现实（分类）标签组成。如果网络被设计为对来自三个类别（例如猫、狗、其他）的对象进行分类，我们将需要来自所有三个类别的训练样本。通常每个类别都需要数千个样本。

包含分类标签的数据集可以将标签表示为字符串（“Cat”、“Dog”、“Other”）或整数（0,1,2）。但是，在通过神经网络处理数据集之前，标签必须数字表示。当数据集包含整数标签（例如，0、1、2）来表示类时，会提供一个类标签文件，用于定义从类名称到数据集中的整数表示的映射。这允许在需要时将整数映射回类名。如下所示的类映射。

```text
Label    Description
  0          Cat
  1          Dog
  2          Other
```

这种类型的标签编码称为整数编码，因为使用唯一的整数对类标签进行编码。但是，当类标签之间没有关系时，建议使用One-Hot Encoding。  One-hot  编码是一种将分类标签表示为二进制向量（仅包含零和一）的技术。在此示例中，我们有三个不同的类（猫、狗和其他），因此我们可以使用长度为 3  的向量以数字方式表示每个类，其中其中一个条目为 1，其他条目均为 0。

```text
Cat   Dog  Other
 1     0     0
 0     1     0
 0     0     1
```

顺序是任意的，但它需要在整个数据集中保持一致。

我们首先考虑一个训练样本，如下图所示，它由输入图像和该图像的类标签组成。对于每个输入训练样本，网络将生成一个由三个数字组成的预测，表示输入图像对应于给定类别的概率。概率最高的输出决定了预测结果。在这种情况下，网络（错误地）预测输入图像是“狗”，因为网络的第二个输出具有最高的概率。请注意，网络的输入只是图像。每个输入图像的类标签用于计算损失，如下一节所述。

 ![](02_fundamentals_of_training_a_neural_network/download2.png)

### 02-03 损失函数 Loss Function

所有神经网络都使用**损失函数**来量化给定训练样本的预测输出与真实值之间的**误差**。正如我们将在下一节中看到的，损失函数可用于指导学习过程（即以提高未来预测准确性的方式更新网络权重network weights）。

量化网络输出与预期结果之间的误差的一种方法是计算误差平方和  (`SSE`)，如下所示。这也称为`损失`(`LOSS`)。在下面的示例中，我们通过计算实际标签值与预测标签值的差来计算单个训练样本的误差。然后对每一项进行平方，三项之和代表总误差，在本例中为 0.6638。
$$
SSE=(1−0.37)^2+(0−0.50)^2+(0−0.13)^2=0.6638
$$
在实践中训练神经网络时，在更新网络权重之前，会使用许多图像来计算损失。因此，下一个方程通常用于计算多个训练图像的均方误差 (MSE)。 MSE  只是所有使用的图像的 SSE 的平均值。用于更新权重的图像数量称为批量大小(**batch size**)（批量大小 32  通常是一个很好的默认值）。一批图像的处理称为一次“迭代”（**iteration**）。
$$
\large{\text{MSE} = \frac{1}{n}  \sum_{i=1}^{n}(y_{i} - y^{'}_{i})^2 = \text{mean(SSE)}}
$$
mean表示求平均值。

### 02-04 梯度下降Gradient Descent（优化）

现在我们已经熟悉了损失函数的概念，我们准备好介绍用于更新神经网络中权重的优化过程。幸运的是，有一种方法可以调整神经网络的权重，称为梯度下降(Gradient Descent)。为简单起见，我们将仅使用一个名为 W 的可调参数来说明这个概念，并且假设损失函数是凸函数，因此形状像碗，如图所示。

PS：在学习考研数学和深度学习时我发现凹凸函数网上的叫法不同，在高等数学中下图被叫做凹函数，但是外国教授却叫凸函数（convex function）。这是因为数分和高数对于凸凹函数的定义是反的。

 ![](02_fundamentals_of_training_a_neural_network/download3.png)

损失函数的值显示在垂直轴上，我们的单个可训练权重的值显示在水平轴上。假设当前的权重估计为 
$$
\normalsize{W_{e1}}
$$
参考左图，如果我们计算当前权重估计对应的点处的损失函数的斜率，We1
，我们可以看到斜率（梯度）为负。在这种情况下，我们需要增加权重以接近 Wo 指示的最佳值。所以我们需要沿着与梯度符号相反的方向移动。

另一方面，如果我们当前的权重估计，We1>Wo（如右图所示），梯度将为正，我们需要减少当前权重的值以更接近最佳值Wo。

请注意，在这两种情况下，我们仍然需要沿着与梯度符号相反的方向移动。

在继续之前，请注意，在这两个图中，我们绘制的代表梯度（斜率）的箭头都指向右侧。在一种情况下，箭头指向右下，而在另一种情况下，箭头指向右上。但不要对两个箭头都指向右侧的事实感到困惑，重要的是梯度的符号，

 ![](02_fundamentals_of_training_a_neural_network/download4.png)

请记住，直线的斜率定义为运行过程中的上升，当权重位于最佳值左侧时，函数的斜率为负，而当权重位于最佳值右侧时，函数的斜率为正。所以梯度的符号很重要。

 ![](02_fundamentals_of_training_a_neural_network/download5.png)

在上述两种情况下，我们都需要在与梯度符号相反的方向上调整权重。考虑到这些概念，我们可以证明以下方程可用于在正确的方向上更新权重，而不管权重的当前值相对于最佳值如何。

 ![](02_fundamentals_of_training_a_neural_network/download6.png)

考虑这个问题的最好方法是，梯度的符号决定了我们需要移动的方向。但是我们需要移动的量需要用一个称为学习率（**Learning Rate**）的参数来调整，该参数通常是一个**很小的数字**(小于 1)。学习率是我们在训练之前需要指定的东西，而不是网络学习的东西。像这样的参数通常称为超参数(**hyperparameters**)，以将它们与可训练参数trainable parameters （例如网络权重weights）区分开来。

实际上，损失函数有很多维度，通常不是凸函数，而是有很多峰和谷。一般情况下，损失函数的斜率称为梯度，是网络中所有权重的函数。但用于更新权重的方法在概念上与此处描述的相同。

  ![](02_fundamentals_of_training_a_neural_network/download7.png)

### 02-05 权重更新计算示例

为了使这一点更加具体，让我们进行一个更新权重的示例计算。这里，我们假设当前权重为 We1，其值为 0.38。我们还将假设学习率为  0.01，并且损失函数在 We1 点的斜率等于 -0.55。使用上面的更新方程，我们可以轻松计算出新的权重估计值，我们将其称为  We2。这个计算得到了简化，因为我们只在一个维度上工作，很容易扩展到多个维度。

PS: *A gradient is a vector, and slope is a scalar*. 上面算的是一维的，所以就是slope，扩展到多个维度就是gradient梯度，如上图所示，所以梯度是一个多维的坡度，而slope斜率是一维平坡。

  ![](02_fundamentals_of_training_a_neural_network/download8.png)

~~我们还没有讨论的一件事是如何实际计算损失函数相对于网络权重的梯度。~~幸运的是，这是通过一种称为反向传播**backpropagation**的算法来处理的，该算法内置于深度学习框架中，例如 TensorFlow、Keras 和 PyTorch，因此您不需要自己实现。

### 02-06 完整的训练闭环

现在我们已经涵盖了与训练神经网络相关的所有基本要素，我们可以在下图中总结该过程。

  ![](02_fundamentals_of_training_a_neural_network/download9.png)

这里，左边是输入图像，右边是网络的输出，我们将其称为 y′。我们使用真实标签 y 以及网络的预测输出来计算损失。请注意，我们没有具体显示网络的多个输出，但应该理解的是 y′ 和 y是向量，其长度等于网络正在训练的类的数量。

计算损失后，我们可以计算损失相对于权重的梯度，然后可以将其用于更新网络中的权重。**这是一个重要的图表，它高度概括了神经网络的训练过程。**

### 02-07 训练曲线 Plots

现在我们已经知道如何更新网络中的权重，值得强调的是，训练神经网络是一个迭代过程，通常需要将整个训练集多次通过网络。

每次整个训练数据集通过网络时，我们将其称为训练时期（**training epoch**）。训练神经网络通常需要许多训练周期，直到损失随着额外训练而停止减少。正如您在下面的第一张图中所看到的，随着训练的进行，损失减少的速度逐渐减小，这表明模型正在接近其学习能力。

  ![](02_fundamentals_of_training_a_neural_network/download10.png)

绘制训练准确度图也很常见，正如您所期望的那样，随着损失的减少，准确度往往会增加，如第二张图所示。

  ![](02_fundamentals_of_training_a_neural_network/download11.png)

有许多与训练神经网络相关的重要细节，我们在第一篇文章中没有介绍，但随着本系列的进展，我们将继续介绍有关该主题的更多高级概念。

注意：我们尚未讨论的一个重要主题是数据分割（**data splitting**）。这涉及到验证数据集的概念，**用于在训练过程中评估训练模型的质量（正确率）**。这是一个重要且核心的主题，将在后续文章中介绍。

### 02-08 使用经过训练的模型执行推理

现在我们已经介绍了如何训练神经网络的过程，有必要谈谈我们将如何使用它。一旦我们有了训练有素的网络，我们就可以为其提供未知内容的图像，并使用该网络来预测该图像属于哪个类别。这反映在下图中，我们所需要的只是想要分类的未知内容的图像。对未知数据进行预测（make prediction）通常称为使用网络进行推理（perform inference）。

 ![](02_fundamentals_of_training_a_neural_network/download12.png)

### 02-09 结论

让我们总结一下与训练神经网络相关的要点。

+ 训练神经网络以执行图像分类等监督学习任务需要标记过的训练数据(labled)。
+ 在大多数情况下，建议对分类数据使用 One-Hot 标签编码。（[1,0,0],[0,1,0],[0,0,1]）
+ 训练神经网络需要一个损失函数，用于量化网络输出和预期输出之间的误差。
+ 损失函数的梯度是使用称为反向传播backpropagation的算法计算的，该算法内置于 TensorFlow 和 PyTorch 等深度学习框架中。
+ 梯度下降以迭代方式用于更新神经网络的权重。
+ 训练图像的子集（批量大小batch size）用于执行权重更新。这称为训练时期内epoch的迭代。
+ 训练时期包括通过网络处理整个训练数据集。因此，训练时期的迭代次数等于训练图像的数量除以批量大小。
+ 每个训练周期代表训练过程的完整过程，直到损失函数稳定。注意：在实践中，我们不仅仅依靠训练损失来评估训练模型的质量。还需要验证损失，我们将在后续文章中介绍这一点。



## 03 线性回归建模

<iframe width="899" height="506" src="https://www.youtube.com/embed/yuAZQJ5BnJk" title="Linear Regression Tutorial using Tensorflow and Keras" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### 学习如何使用 Keras 进行线性回归建模

在研究深度神经网络之前，我们将介绍简单（线性）神经网络的基本组成部分。我们将从线性回归的主题开始。由于线性回归可以建模为神经网络，因此它提供了一个很好的示例来介绍神经网络的基本组件。回归Regression是监督学习的一种形式，旨在对一个或多个输入变量（特征）与连续（目标）变量之间的关系进行建模。我们假设输入变量 x 和目标变量 y  之间的关系可以表示为输入的加权和（即模型的参数是线性的）。简而言之，线性回归旨在学习一种**将一个或多个输入特征映射到单个数值目标值的函数**。

 ![](03_modeling_linear_regression/download1.png)

### 03-00目录

1. 了解数据集
2. 线性回归模型
3. 神经网络概念和术语
4. 在 Keras 中建模神经网络
5. 结论

```py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers

import tensorflow as tf
import matplotlib.pyplot as plt

SEED_VALUE = 42

# Fix seed to make training deterministic.
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
```

### 03-01 了解数据集

#### 03-01-01 加载波士顿住房数据集

在本文中，我们将使用波士顿住房数据集。该数据集包含美国人口普查局收集的有关马萨诸塞州波士顿住房的信息。它已在文献中广泛用于基准算法，并且由于其数据量小也适合演示目的。该数据集包含 14 个独特属性，其中包括给定郊区房屋的中值（千美元价格）。我们将使用该数据集作为示例，说明如何开发一个模型，使我们能够根据数据集中的单个属性（房屋中的平均房间数）来预测房屋的中位价格。

Keras 提供了 load_data() 函数来加载该数据集。数据集通常分为训练`train`和测试`test`集，load_data() 函数为每个数据集返回一个元组。每个元组包含一个二维特征数组（例如 X_train）和一个向量，该向量包含数据集中每个样本的关联目标值（例如 y_train）。例如，X_train 中的行代表数据集中的各种样本，列代表各种特征。在本笔记本中，我们将仅利用训练数据来演示如何训练模型。然而，在实践中，使用测试数据来了解训练后的模型在未见过的数据上的表现非常重要。

```python
# Load the Boston housing dataset.
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(X_train.shape)
print("\n")
print("Input features: ", X_train[0])
print("\n")
print("Output target: ", y_train[0])
```

```text
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz
57026/57026 [==============================] - 0s 0us/step
(404, 13)

Input features:  [  1.23247   0.        8.14      0.        0.538     6.142    91.7
   3.9769    4.      307.       21.      396.9      18.72   ]
Output target:  15.2
```

#### 03-01-02 从数据集中提取特征

在此笔记本中，我们将仅使用数据集中的单个特征，因此为了简单起见，我们将特征数据存储在一个新变量中。

```python
boston_features = {
    "Average Number of Rooms": 5,
}

X_train_1d = X_train[:, boston_features["Average Number of Rooms"]]
print(X_train_1d.shape)

X_test_1d = X_test[:, boston_features["Average Number of Rooms"]]
# (404,)
```

#### 03-01-03 绘制特征点

在这里，我们绘制了房屋的中位价格与单一特征（“平均房间数 Average Number of Rooms'”）的关系。

```python
plt.figure(figsize=(15, 5))
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price [$K]")
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color="green", alpha=0.5)
# <matplotlib.collections.PathCollection at 0x7f21bbde9d60>
```

 ![](03_modeling_linear_regression/download2.png)

### 03-02 线性回归模型

让我们首先清楚地了解我们正在努力实现的目标。下图显示了单个自变量（房间数量）和因变量（房屋中位价格）的训练数据。我们希望使用线性回归来为这些数据开发一个可靠的模型。在此示例中，模型只是一条由斜率 (m) 和 y 截距 (b) 定义的直线。

 ![](03_modeling_linear_regression/download3.png)

### 03-03 神经网络概念和术语

下图显示了如何将该模型表示为简单（单神经元）网络。我们将使用这个简单的示例来介绍神经网络组件和术语。输入数据 (x)  由单个特征（房间的平均数量）组成，预测输出 (y′)  是标量（房屋的预测中位价格）。请注意，数据集中的每个数据样本代表波士顿郊区的统计数据。模型参数（m 和  b）在训练过程中迭代学习。您可能已经知道，模型参数可以通过封闭形式的**普通最小二乘法** (Ordinary Least Squares-OSL)  来计算。然而，我们也可以使用称为梯度下降**Gradient Descent**的数值技术迭代地解决这个问题，这是神经网络训练的基础。此处我们不介绍梯度下降的细节，但重要的是要了解它是一种用于调整模型参数的迭代技术。

 ![](03_modeling_linear_regression/download4.png)

该网络仅包含一个神经元，该神经元接受单个输入 (x) 并产生单个输出  (y′)，即房屋的预测（平均）价格。单个神经元有两个可训练参数，即线性模型的斜率slope (m) 和 y 轴截距y-intercept   (b)。这些参数通常分别称为权重weight和偏差bias。在回归问题中，模型通常具有多个输入特征，其中每个输入都有一个关联的权重  (wi)，但在本例中，我们将仅使用单个输入特征来预测输出。因此，一般来说，一个神经元通常具有多个权重（w1、w2、w3 等）和一个偏置项  (b)。在此示例中，您可以将神经元视为 mx+b 的数学计算，它产生预测值 y′。

下面显示了同一模型的稍微更正式的图表。在这里，我们引入了反馈循环**feedback loop** 的概念，它显示了模型参数（w和b) 在训练过程中更新。最初，模型参数被初始化为小的随机值。在训练过程中，当训练数据通过网络传递时，模型的预测值 (y′)  会与数据集中给定样本的真实值 (y)  进行比较。该差异用作计算损失，然后用作网络中的反馈，以改进预测的方式调整模型参数。这个过程涉及两个步骤，称为梯度下降和反向传播。在这个阶段，了解其工作原理的数学细节并不重要，但重要的是要了解训练模型的迭代过程。

 ![](03_modeling_linear_regression/download5.png)

我们使用的损失函数可以有多种形式。在这个例子中，我们将使用均方误差（MSE），这是回归问题中非常常见的损失函数。
$$
J = \frac{1}{m}\sum_{i=1}^{m} (y_{i}' - y_{i})^2
$$
**基本思想是我们希望最小化该函数的值**，该函数表示我们的模型和训练数据集之间的误差。在上式中，m是训练样本的数量。

### 03-04 在 Keras 中建模神经网络

上一节中的网络图代表了最简单的神经网络。该网络有一个由输出 wx+b 的单个神经元组成的单层。对于每个训练样本，预测输出 y′

与训练数据的实际值进行比较，并计算损失。然后可以使用损失来微调（更新）模型参数。

与训练神经网络相关的所有细节均由 Keras 处理，总结如下工作流程：

1. 使用 Keras 中的预定义层构建/定义网络模型。
2. 使用 model.compile() 编译模型
3. 使用 model.fit() 训练模型
4. 预测输出 model.predict()

#### 03-04-01 定义 Keras 模型

```python
model = Sequential()

# Define the model consisting of a single neuron. 包含一个单神经元
model.add(Dense(units=1, input_shape=(1,)))

# Display a summary of the model architecture.
model.summary()
```

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 1)                 2                               
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
```

#### 03-04-02 编译模型

```python
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005), loss="mse")
```

#### 03-04-03 训练模型

```python
history = model.fit(
    X_train_1d, 
    y_train, 
    batch_size=16, 
    epochs=101, 
    validation_split=0.3,
)
```

```text
Epoch 1/101
18/18 [==============================] - 1s 18ms/step - loss: 218.3039 - val_loss: 266.6791
Epoch 2/101
18/18 [==============================] - 0s 5ms/step - loss: 200.4328 - val_loss: 248.9743
...
...
...
Epoch 100/101
18/18 [==============================] - 0s 3ms/step - loss: 53.9164 - val_loss: 74.0994
Epoch 101/101
18/18 [==============================] - 0s 3ms/step - loss: 53.8947 - val_loss: 74.0549
```

#### 03-04-04 绘制训练结果

```python
def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

plot_loss(history)
```

 ![](03_modeling_linear_regression/download6.png)

上面的损耗曲线相当典型。首先，请注意有两条曲线，一条用于训练损失，一条用于验证损失。两者最初都很大，然后稳步下降，最终趋于平稳，在大约 30 个 epoch 后没有进一步改善。由于模型仅在训练数据上进行训练，因此训练损失低于验证损失也是相当典型的。

#### 03-04-05 使用模型进行预测

我们现在可以使用Keras 中的predict() 方法进行单个预测。在此示例中，我们将值列表传递给模型（表示平均房间数），模型为每个输入返回房屋价格的预测值。

```python
# Predict the median price of a home with [3, 4, 5, 6, 7] rooms.
x = [3, 4, 5, 6, 7]
y_pred = model.predict(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")
```

```text
1/1 [==============================] - 0s 99ms/step
Predicted price of a home with 3 rooms: $11.0K
Predicted price of a home with 4 rooms: $14.4K
Predicted price of a home with 5 rooms: $17.9K
Predicted price of a home with 6 rooms: $21.3K
Predicted price of a home with 7 rooms: $24.8K
```

#### 03-04-06 绘制模型和数据

```python
# Generate feature data that spans the range of interest for the independent variable.
x = np.linspace(3, 9, 10)

# Use the model to predict the dependent variable.
y = model.predict(x)
# 1/1 [==============================] - 0s 44ms/step

def plot_data(x_data, y_data, x, y, title=None):
    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()

```

训练集：

```python
plot_data(X_train_1d, y_train, x, y, title='Training Dataset')
```

 ![](03_modeling_linear_regression/download7.png)

测试集：

```python
plot_data(X_test_1d, y_test, x, y, title='Test Dataset')
```

 ![](03_modeling_linear_regression/download8.png)

### 03-05 结论

本文我们在简单神经网络的背景下介绍了线性回归模型。我们展示了如何使用 Keras 来建模和训练网络以学习线性模型的参数以及如何可视化模型预测。




## 04 使用 MLP 进行 MNIST 数字分类

### 在 Keras 中使用前馈网络进行图像分类

下面我们将介绍与涉及两个以上类别的一般分类问题相关的几个新概念。当类的数量超过两个时，这有时被称为多项式回归或 softmax  回归。具体来说，我们将了解如何使用前馈多层感知器（Multilayer Perceptron Network）网络对 MNIST 数据集中的手写数字进行分类。 ~~MLP  并不是处理图像数据的首选方式，但这可以作为引入一些新概念的一个很好的例子~~。 MNIST 手写数字数据集包含在 Tensorflow  中，可以轻松导入和加载，如下所示。使用此数据集和简单的前馈网络，我们将演示一种如何处理图像数据并构建对数字 [0,9] 进行分类的网络的方法。

 ![](04_MNIST_digit_classification_using_MLP/download1.png)

### 04-00 目录

1. 加载 MNIST 数据集
2. 数据集预处理
3. 模型架构
4. 模型实现
5. 模型评估
6. 结论

准备导入必须的库和准备种子数据

```python
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["image.cmap"] = "gray"

from tensorflow.keras.datasets import fashion_mnist

SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
```

### 04-01 加载并分割 MNIST 数据集

MNIST 数据集包含 70,000 张图像，分为 60,000 张用于训练，10,000 张用于测试。保留一部分数据用于验证可以通过进一步划分训练数据来完成。如下所示，我们从训练数据中提取 10,000 个样本用于验证。

```python
(X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()

X_valid = X_train_all[:10000]
X_train = X_train_all[10000:]

y_valid = y_train_all[:10000]
y_train = y_train_all[10000:]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#11490434/11490434 [==============================] - 2s 0us/step
#(50000, 28, 28)
#(10000, 28, 28)
#(10000, 28, 28)

plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.axis(True)
    plt.imshow(X_train[i], cmap="gray")
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

```

 ![](04_MNIST_digit_classification_using_MLP/download2.png)

### 04-02 数据集预处理

#### 04-02-01 输入特征变换和归一化

由于我们现在使用图像作为输入，因此我们需要找到一些逻辑方法将图像数据表示为一组特征。对于该数据集来说，一种实际上效果相当好的简单方法是假设像素强度是特征。将图像数据转换为我们可以处理的一组特征的一种方法是将 2D 数组展平为 1D 数组。 28x28 输入图像因此成为包含 784 个特征的一维数组。请注意，我们还将像素强度标准化为 [0, 1]  范围内。这在处理图像数据时很常见，这有助于更有效地训练模型。另外，需要明确的是，使用像素强度作为特征是一种幼稚低级的方法，我们在这里故意使用这种方法是为了让事情变得简单。正如我们将在后续文章中看到的，我们将了解卷积神经网络 (CNN)，它使用更先进的技术来表示和处理神经网络中的图像数据。

```python
X_train = X_train.reshape((X_train.shape[0], 28 * 28))
X_train = X_train.astype("float32") / 255

X_test = X_test.reshape((X_test.shape[0], 28 * 28))
X_test = X_test.astype("float32") / 255

X_valid = X_valid.reshape((X_valid.shape[0], 28 * 28))
X_valid = X_valid.astype("float32") / 255
```

#### 04-02-02 标签编码选项

在处理分类数据时，在通过机器学习算法处理数据之前，需要将目标标签表示为数值。标签编码是将类标签从字符串转换为数值的过程。对于如何对每个类别的标签进行数字编码，我们有几个选项。我们可以使用序数整数编码，其中为每个类分配一个整数，或者我们可以使用一种称为 one-hot  编码的技术，该技术使用单独的二进制向量对每个类标签进行编码。根据数据集的不同，一种方法可能优于另一种方法，但在大多数情况下，通常使用  one-hot 编码。由于这是一篇介绍性文章，我们将简要演示每种编码的样子，以便您熟悉这两种表示形式。

##### A 整数标签编码

包含分类标签的数据集可以在内部将标签表示为字符串或整数。然而，在通过神经网络处理数据集之前，标签必须数字表示。当数据集包含分类数据的整数标签时，会提供一个类标签文件，该文件定义从类名称到数据集中的整数表示的映射，以便在需要时可以将整数映射回类名称。作为一个具体示例，请考虑下面显示的 Fashion MNIST 数据集的字典映射。

```text
Label   Description
0       T-shirt/top
1       Trouser
2       Pullover
3       Dress
4       Coat
5       Sandal
6       Shirt
7       Sneaker
8       Bag
9       Ankle boot
```

Fashion MNIST  数据集本身包含整数标签，我们可以通过加载数据集并打印出一些标签来验证这一点，如下面代码单元的输出所示。这种类型的标签编码称为整数编码，因为使用唯一的整数对类（字符串）标签进行编码。但是，当类标签彼此没有关系时，通常建议改用 One-Hot Encoding，这将在下一节中介绍。

```python
# Load the Fashion MNIST dataset.
((X_train_fashion, y_train_fashion), (_, _)) = fashion_mnist.load_data()

# The labels in the Fashion MNIST dataset are encoded as integers.
print(y_train_fashion[0:9])

#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
#29515/29515 [==============================] - 0s 1us/step
#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
#26421880/26421880 [==============================] - 3s 0us/step
#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
#5148/5148 [==============================] - 0s 0us/step
#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
#4422102/4422102 [==============================] - 1s 0us/step

#[9 0 0 3 0 2 7 2 5]
```

##### B One-Hot标签编码

One-hot 编码是一种将分类标签表示为 one-hot 编码向量的技术。因此，我们可以使用 Keras 中的  to_categorical()  函数作为预处理步骤，将每个标签表示为二进制向量，而不是将类标签表示为唯一的整数。在这种情况下，每个标签都转换为二进制向量，其中向量的长度等于类的数量。除了与整数标签相对应的元素之外，所有条目都设置为零。

```python
y_train_onehot = to_categorical(y_train_fashion[0:9])
print(y_train_onehot)
```

```text
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
```

注意：由于 MNIST 数字数据集中的标签具有直接对应于类标签的整数标签（即整数 4 对应于类标签  4），因此技术上不需要类映射文件。此外，由于整数标签具有自然排序，因此我们可以直接使用整数标签。但由于最常使用 one-hot  编码，我们将继续以这种方式对标签进行编码，如下所示。

```python
# Convert integer labels to one-hot encoded vectors.
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test  = to_categorical(y_test)
```

### 04-03 模型架构

#### 04-03-01 深度神经网络架构

下面显示的网络架构有多层。一个输入层、两个隐藏层和一个输出层。关于此架构有几点需要注意。

1. 输入数据：图像输入数据从二维数组 [28x28] 预处理（展平）为长度 [784x1] 的一维向量，其中该输入向量中的元素是归一化像素强度。网络的输入有时被称为输入“层”，但从技术上讲，它不是网络中的层，因为没有与之相关的可训练参数。
2. 隐藏层：我们有两个隐藏层，其中包含一定数量的神经元（我们需要指定）。这些层中的每个神经元都有一个非线性激活函数（例如 ReLU、Sigmoid 等）。
3. 输出层：现在，我们在输出层中有 10 个神经元来表示 10 个不同的类别（数字：0 到 9），而不是回归示例中的单个神经元。
4. 密集层：网络中的所有层都是完全连接的，这意味着给定层中的每个神经元都与前一层中的每个神经元完全连接（或密集）。与每层关联的权重以粗体表示，以指示这些矩阵包含网络中相邻层之间的所有连接的每个权重。
5. Softmax 函数：输出层中每个神经元的值通过 softmax 函数传递，以生成数据集中十个数字中每个数字的概率得分。
6. 网络输出：网络输出（y′) 是一个长度为 10 的向量，包含每个输出神经元的概率。预测类标签只需要传递 (y′)通过argmax函数来确定预测标签的索引。
7. 损失函数：使用的损失函数是交叉熵损失，这通常是分类问题的首选损失函数。它是根据地面实况标签（y) 和网络的输出概率 (y′)。注意 y 和 y′都是长度等于类数的向量。

尽管该图看起来与线性回归示例中的单层感知器有很大不同，但就训练和预测期间发生的处理而言，它基本上非常相似。我们仍然根据网络的预测输出和输入的真实标签来计算损失。反向传播用于计算损失相对于网络权重的梯度。优化器（实现梯度下降）用于更新神经网络中的权重。

 ![](04_MNIST_digit_classification_using_MLP/download3.png)

### 04-04 模型实现

在这里，我们使用 Keras 定义模型架构，该架构具有两个密集层（每个密集层有 128 个神经元）和一个有 10  个神经元的输出层。输出层中的每个神经元对应于数据集中的类别标签（0 到  9），其中每个神经元的输出表示输入图像对应于与该神经元关联的类别的概率。例如，如果第 5 个神经元的输出为 0.87，则意味着输入图像为 4  的概率为 87%（因为第一类为 0，因此第 5 个神经元代表数字 4）。

请注意，第一个隐藏层的输入形状为 [784,1]，因为  28x28 图像被展平为长度为 784 的向量。每个隐藏层中的神经元都具有称为“ReLU”的激活函数，它代表“整流线性单元”  。然后，输出层中的神经元通过“softmax”函数，该函数对原始输出进行转换（标准化），可以将其解释为如上所述的概率。

我们不会在这篇文章中介绍 softmax 函数或交叉熵损失函数的细节，因为这些是更高级的主题，但简单地说，softmax  函数对网络的输出进行归一化并将其转换为概率。交叉熵损失函数计算预测输出概率和地面真实标签之间的损失。预测的输出概率距离目标标签越远，损失就越高。

#### 04-04-01 定义模型

```python
# Instantiate the model.
model = tf.keras.Sequential()

# Build the model.
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10,  activation="softmax"))

# Display the model summary.
model.summary()
```

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               100480    
                                                                 
 dense_1 (Dense)             (None, 128)               16512     
                                                                 
 dense_2 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
```

#### 04-04-02 编译模型

此步骤定义将在训练循环中使用的优化器**Optimizer**和损失函数**Loss Function**。我们也可以在这里指定要跟踪的任何其他指标。

优化器：这里，我们将使用 Keras 中的 RMSProp 优化器。

损失函数：如上所述，分类问题的首选损失函数是交叉熵。但根据标签的编码方式，我们需要指定交叉熵损失函数的正确形式。如果标签是one-hot编码的，那么你应该将损失函数指定为categorical_crossentropy，如果标签是整数编码的，那么你应该使用sparse_categorical_crossentropy。进行二元分类时，应该使用binary_crossentropy作为损失函数。由于我们在本例中使用 one-hot 编码，因此我们将损失函数指定为 categorical_crossentropy。

指标**Metrics**：最后，我们还指定准确性作为训练期间记录的附加指标，以便我们可以在训练完成后绘制它。训练损失和验证损失会自动记录，因此无需指定。

```python
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

#### 04-04-03 训练模型

为了训练模型，我们调用 Keras 中的 fit()  方法。请注意，由于我们将原始训练数据分为训练数据集和验证数据集，因此我们需要使用validation_data  =（X_valid，y_valid））显式指定验证数据集。回想一下本系列之前关于线性回归的文章，我们还可以选择使用validation_split参数来自动保留训练数据的随机部分用作验证数据。因此，在这里，我们演示如何显式使用单独的验证数据集。

```python
training_results = model.fit(X_train, 
                             y_train, 
                             epochs=21, 
                             batch_size=64, 
                             validation_data=(X_valid, y_valid));
```

```text
Epoch 1/21
782/782 [==============================] - 12s 6ms/step - loss: 0.2833 - accuracy: 0.9173 - val_loss: 0.1755 - val_accuracy: 0.9450
Epoch 2/21
782/782 [==============================] - 3s 3ms/step - loss: 0.1203 - accuracy: 0.9634 - val_loss: 0.1276 - val_accuracy: 0.9622
...
...
...
Epoch 20/21
782/782 [==============================] - 3s 3ms/step - loss: 0.0033 - accuracy: 0.9989 - val_loss: 0.1924 - val_accuracy: 0.9743
Epoch 21/21
782/782 [==============================] - 3s 3ms/step - loss: 0.0034 - accuracy: 0.9988 - val_loss: 0.1595 - val_accuracy: 0.9780
```

### 04-05 绘制训练结果

下面介绍一个方便的函数，用于绘制训练和验证损失以及训练和验证准确性。它有一个必需的参数，绘制指标列表。

```python
def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]
        
    for idx, metric in enumerate(metrics):    
        ax.plot(metric, color=color[idx])
    
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, 20])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)   
    plt.show()
    plt.close()
```

可以从 fit 方法返回的历史对象中访问损失和准确性指标。我们使用预定义的字典键访问指标，如下所示。

```python
# Retrieve training results.
train_loss = training_results.history["loss"]
train_acc  = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc  = training_results.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 0.5],
    metric_name=["Training Loss", "Validation Loss"],
    color=["g", "b"],
)

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.9, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
)
```

 ![](04_MNIST_digit_classification_using_MLP/download4.png)

 ![](04_MNIST_digit_classification_using_MLP/download5.png)

### 04-05 模型评估

#### 04-05-01 对样本测试图像进行预测

我们现在可以预测所有测试图像的结果，如下面的代码所示。在这里，我们调用predict()方法来预测，然后从测试集中选择特定索引并打印出每个类别的预测分数。您可以通过将测试索引设置为各种值来试验下面的代码，并查看最高分数如何与正确值相关联。

```python
predictions = model.predict(X_test)
index = 0  # up to 9999
print("Ground truth for test digit: ", y_test[index])
print("\n")
print("Predictions for each class:\n")
for i in range(10):
    print("digit:", i, " probability: ", predictions[index][i])
```

```text
313/313 [==============================] - 1s 1ms/step
Ground truth for test digit:  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]


Predictions for each class:

digit: 0  probability:  9.819607e-24
digit: 1  probability:  2.4064698e-18
digit: 2  probability:  1.4520596e-13
digit: 3  probability:  2.4951994e-13
digit: 4  probability:  1.5394617e-26
digit: 5  probability:  9.713211e-23
digit: 6  probability:  4.6183826e-30
digit: 7  probability:  1.0
digit: 8  probability:  1.8647681e-26
digit: 9  probability:  1.4221963e-17
```

#### 04-05-01 混淆矩阵

混淆矩阵是一种非常常见的度量，用于总结分类问题的结果。该信息以表格或矩阵的形式呈现，其中一个轴代表每个类别的真实标签，另一个轴代表来自网络的预测标签。表中的条目表示实验中的实例数（有时表示为百分比而不是计数）。在 TensorFlow 中生成混淆矩阵是通过调用函数 tf.math.confusion_matrix()  来完成的，该函数采用两个必需参数，即真实标签列表和关联的预测标签。

```python
# Generate predictions for the test dataset.
predictions = model.predict(X_test)

# For each sample image in the test dataset, select the class label with the highest probability.
predicted_labels = [np.argmax(i) for i in predictions]
# 313/313 [==============================] - 0s 1ms/step
```

```python
# Convert one-hot encoded labels to integers.
y_test_integer_labels = tf.argmax(y_test, axis=1)

# Generate a confusion matrix for the test dataset.
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

# Plot the confusion matrix as a heatmap.
plt.figure(figsize=[15, 8])
import seaborn as sn

sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14})
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
```

 ![](04_MNIST_digit_classification_using_MLP/download6.png)

### 04-06 结论

我们介绍了一种简单的方法，用于对图像数据进行建模，以便在密集连接的网络中分类。在下一个课程，我们将了解专门用于处理图像数据的卷积神经网络（CNN）。以下链接包含几个著名的 CNN 架构的非常好的交互式基于网络的动画，这是开始熟悉它们的好地方。

https://tensorspace.org/html/playground/lenet.html

## 05 CNN 基础知识

 ![](05_CNN_fundamentals/download1.png)
