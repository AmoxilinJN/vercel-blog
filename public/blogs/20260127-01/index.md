## rtdetr-r18

### train_batch

![](/blogs/20260127-01/e89737fc9dc5a62a.jpg)

![](/blogs/20260127-01/370a2e3b2d8d6a72.jpg)

![](/blogs/20260127-01/39bc4dd09a87f310.jpg)

### Loss曲线和mAP曲线

![](/blogs/20260127-01/aa391de828e617ba.png)

### 混淆矩阵

![](/blogs/20260127-01/d30d47a92c142b2a.png)

![](/blogs/20260127-01/d799c4d880fd1372.png)

### PR曲线

![](/blogs/20260127-01/c7f2b61c787ef28a.png)

![](/blogs/20260127-01/54434b4d683fcae6.png)

![](/blogs/20260127-01/c00b8ff9810d1536.png)

### F1曲线

![](/blogs/20260127-01/46e5a7f2ba1974bc.png)

### 实际预测效果

![](./detect/exp/000001.jpg)

![000003](./detect/exp/000003.jpg)

![000023](./detect/exp/000023.jpg)

![000035](./detect/exp/000035.jpg)

![000089](./detect/exp/000089.jpg)

## rtdetr-starnet

### train_batch

![](./train/starnet/train_batch0.jpg)

![](./train/starnet/train_batch1.jpg)

![train_batch2](./train/starnet/train_batch2.jpg)

### Loss曲线和mAP曲线

![](./train/starnet/results.png)

### 混淆矩阵

![](./train/starnet/confusion_matrix.png)

![confusion_matrix_normalized](./train/starnet/confusion_matrix_normalized.png)

###  PR曲线

![](./train/starnet/P_curve.png)

![PR_curve](./train/starnet/PR_curve.png)

![R_curve](./train/starnet/R_curve.png)

### F1曲线

![](./train/starnet/F1_curve.png)

### 实际预测效果

![待测](./detect/starnet/000001.jpg)

![000003](./detect/starnet/000003.jpg)

![000023](./detect/starnet/000023.jpg)

![000035](./detect/starnet/000035.jpg)

![000089](./detect/starnet/000089.jpg)

## rtdetr-r50-ASF

### train_batch

![](./train/r50-ASF/train_batch0.jpg)

![](./train/r50-ASF/train_batch1.jpg)

![train_batch2](./train/r50-ASF/train_batch2.jpg)

### Loss曲线和mAP曲线

![](./train/r50-ASF/results.png)

### 混淆矩阵

![](./train/r50-ASF/confusion_matrix.png)

![confusion_matrix_normalized](./train/r50-ASF/confusion_matrix_normalized.png)

### PR曲线

![](./train/r50-ASF/P_curve.png)

![PR_curve](./train/r50-ASF/PR_curve.png)

![R_curve](./train/r50-ASF/R_curve.png)

### F1曲线

![](./train/r50-ASF/F1_curve.png)

### 实际预测效果

![待测](./detect/r50-ASF/000001.jpg)

![000003](./detect/r50-ASF/000003.jpg)

![000023](./detect/r50-ASF/000023.jpg)

![000035](./detect/r50-ASF/000035.jpg)

![000089](./detect/r50-ASF/000089.jpg)

![000093](./detect/r50-ASF/000093.jpg)

## rtdetr-r101

### train_batch

![](./train/r101/train_batch0.jpg)

![](./train/r101/train_batch1.jpg)

![train_batch2](./train/r101/train_batch2.jpg)

### Loss曲线和mAP曲线

![](./train/r101/results.png)

### 混淆矩阵

![](./train/r101/confusion_matrix.png)

![confusion_matrix_normalized](./train/r101/confusion_matrix_normalized.png)

### PR曲线

![](./train/r101/P_curve.png)

![PR_curve](./train/r101/PR_curve.png)

![R_curve](./train/r101/R_curve.png)

### F1曲线

![](./train/r101/F1_curve.png)

### 实际预测效果

![待测](./detect/r101/000001.jpg)

![000003](./detect/r101/000003.jpg)

![000023](./detect/r101/000023.jpg)

![000035](./detect/r101/000035.jpg)

![000089](./detect/r101/000089.jpg)

![000093](./detect/r101/000093.jpg)

## AKConv

![](./train/AKConv/training_metrics.png)

![](./detect/AKConv/000001.jpg)

![000003](./detect/AKConv/000003.jpg)

![000023](./detect/AKConv/000023.jpg)

![000035](./detect/AKConv/000035.jpg)

![000089](./detect/AKConv/000089.jpg)

![000093](./detect/AKConv/000093.jpg)

> 以上训练皆未加载预权重

## 模型对比

### mAP 曲线对比

![](./train/comparison_result.png)

### size 和 FPS 对比

- r18: size:77.0M (bs:1)Latency:0.03013s +- 0.00556s fps:33.2
- AKConv: size:59.5M (bs:1)Latency:0.07989s +- 0.01048s fps:12.5
- r50-ASF: size:168.1M (bs:1)Latency:0.05301s +- 0.00987s fps:18.9
- starnet: size:46.9M (bs:1)Latency:0.03384s +- 0.00364s fps:29.5
- r101: size:292.8M (bs:1)Latency:0.05755s +- 0.00771s fps:17.4

## 数据分析

在性能上，r50-ASF 表现较好，其 mAP50-95 达到 0.612，相比  r18 提升了近 91%，而且 Precision (0.936) 和 Recall (0.921) 极其均衡并维持在很高水平，说明 ASF 模块在特征提取和多尺度融合方面起到了关键作用，能够极好地捕捉目标细节并减少漏检，而且速度也比 AKConv 和 r101 快，性价比很高

在效率上，StarNet  展现了极佳的效率/精度平衡，不仅超越了改进版的 AKConv，甚至击败了参数量更大的 r101 (0.513)。通过分析 Loss 数据发现，StarNet 的验证集回归损失 ( val/giou_loss: 0.340, val/l1_loss: 0.169 ) 是所有模型中最低的。这表明 StarNet 虽然召回率 (0.75) 稍低，但对检出目标的边框回归非常精准，同时速度也非常快，模型也非常小，因此非常适合对速度有一定要求且希望保持高精度的边缘侧部署

对于 r101 的表现，我认为大概率是训练集太少导致的，而非模型性能问题，因此暂不做分析

对于 AKConv，相比 r18 有显著提升，其 Precision (0.927) 很高，但 mAP50-95 受限于 Recall 等因素，因此表现中规中矩

r18 作为轻量级基准，验证了上述改进模型的有效性。虽然速度最快，但是在精度和模型大小上还是不如 starnet，总体不如 starnet

## 模型算法原理

### RT-DETR-r18

RT-DETR-r18 采用了经典的 ResNet-18 作为主干网络（Backbone），它的设计核心在于在保持 Transformer 全局感受野优势的同时，通过混合编码器（Hybrid Encoder）解决推理速度慢的问题

RT-DETR-r18 的整体架构可以分为三个部分：Backbone (主干)、Neck (混合编码器) 和 Head (Transformer 解码器)

#### Backbone: ResNet-18

主干网络负责从输入图像中提取多尺度特征
*   ResNet-18 是一个 18 层的深度残差网络。它由 4 个主要阶段（Stage）组成，每个阶段包含 2 个 BasicBlock
    *   BasicBlock：每个 BasicBlock 包含两个 $3\times3$ 卷积层，通过残差连接 (Residual Connection)，输入 $x$ 直接加到输出上（$F(x) + x$），这解决了深层网络梯度消失的问题，使得网络更容易训练

*   它输出三个尺度的特征图：$P_3$ (8倍下采样), $P_4$ (16倍下采样), $P_5$ (32倍下采样)，通道数分别为：128, 256, 512
*   ResNet-18 结构简单、推理速度极快，但相比 ResNet-50/101，其深层语义提取能力相对较弱

#### Neck: 混合编码器 (Efficient Hybrid Encoder)

传统的 DETR 使用多层 Transformer Encoder 处理所有尺度特征，计算量巨大，而 RT-DETR 设计了混合编码器来解耦尺度内和尺度间的特征交互

##### AIFI (Attention-based Intra-scale Feature Interaction)

*   原理：AIFI 模块仅在最高层级特征 ($P_5$) 上应用 Transformer Encoder（自注意力机制）
    *   *最高层级特征 (P5)*：这是神经网络“读”图读到最后提取出的特征，虽然它的分辨率最低（图像被缩小了32倍，很模糊），但它包含了最丰富的语义信息（比如它能“知道”这是只猫，而不是仅仅看到猫的毛发纹理），是模型识别物体类别的关键依据
    *   *Transformer Encoder（自注意力机制）*：这就好比我们在看一张图时，眼睛会不自觉地在画面不同位置之间反复扫视，建立联系（比如看到车轮自然联想到车身），它能让模型在处理某个像素时，参考整张图中所有其他像素的信息，从而获得全局的理解能力

*   $P_5$ 特征图分辨率最低，但包含最丰富的语义信息。在这里使用自注意力可以以极低的计算成本捕获图像的全局上下文信息（这是 CNN 难以做到的），从而更好地区分物体和背景
*   传统 DETR 在所有尺度（包括高分辨率的 $P_3$）上都做自注意力，导致计算量呈平方级增长。RT-DETR 仅在 $P_5$ 做，大幅提升了速度

##### CCFM (Cross-scale Channel-sharing Feature Fusion Module)

*   原理：CCFM 负责多尺度特征的融合。它类似于 YOLO 系列中的 PANet 结构，包含自顶向下和自底向上的路径
    *   *多尺度融合*：就像为了看清近处的小字要摘下眼镜凑近看（高分辨率特征），而为了看清远处的山要戴上眼镜看轮廓（低分辨率特征），多尺度融合就是把这两种“视觉”结合起来，让模型既能看清小物体，又能识别大物体

*   RT-DETR 并没有使用 Transformer 来做多尺度融合，而是使用了类似 YOLO 的 RepC3 模块（基于 CSPNet 和 RepVGG），通过重参数化 (Reparameterization)，在训练时使用多分支结构增强特征提取，在推理时将多分支合并为一个 $3\times3$ 卷积，从而在不损失精度的情况下显著提升推理速度
    *   *CSPNet*：这是一种聪明的网络设计技巧，它把特征图一分为二，一部分继续处理，另一部分直接连到后面，这样既减少了重复计算量，又让梯度传播更顺畅
    *   *RepVGG*：这是一个“训练时复杂，推理时简单”的黑科技，它允许模型在训练时使用多条路径来学得更好，但在使用时把这些路径融合成一条简单的直路，从而在不牺牲精度的情况下让推理速度飞快


#### Head: Transformer Decoder & Query Selection

RT-DETR 的解码器部分沿用了 DETR 的范式，但做了关键优化，如下：

*   IoU-aware Query Selection (IoU 感知查询选择)：传统 DETR 使用随机初始化或固定的 Object Queries，而 RT-DETR 直接从 Encoder 输出的特征图中，挑选出置信度最高的前 $K$ 个特征点（Top-K）作为初始 Queries，通过引入 IoU 约束，确保被选中的特征点不仅分类分数高，而且定位也尽可能准确（即与 GT 的 IoU 高）
    *   *Object Queries (物体查询向量)*：可以把它们想象成一群被派出去的“侦探”，每个侦探手里拿着不同的线索板，去图像特征中寻找是否有符合自己线索的目标，如果有，就把它框出来并告诉我们那是什么
    *   *IoU 约束*：IoU 是衡量两个框重合程度的指标，IoU 约束就是强迫模型在挑选“侦探”（Queries）时，不能只选那些“以此为荣”喊得响的（分类分高），必须选那些真正能把框画得准（和真实物体重合度高）的，以此保证选出的框既准又稳

*   解码过程：这 300 个 Queries 在 Decoder 层中与图像特征进行 Cross-Attention（交叉注意力），不断修正目标的位置和类别，最终直接输出 300 个预测框，无需 NMS（非极大值抑制）后处理
    *   *Cross-Attention（交叉注意力）*：这是“侦探”（Queries）和图像特征互动的过程，侦探拿着自己的线索去图像特征里找对应的线索，只关注和自己目标相关的信息（比如找猫的侦探只关注毛茸茸的特征），忽略无关背景
    *   *NMS（非极大值抑制）*：这是一个筛选过程，当模型对同一个物体预测了好多重叠的框时，NMS 就像一个严厉的考官，只保留得分最高的那一个框，把其余重叠的、多余的框统统删掉，确保每个物体只保留一个最佳结果


### RT-DETR-r50-ASF

RT-DETR-r50-ASF 在标准 RT-DETR-r50 的基础上，引入了 ASF (Attentional Scale Feature) 模块，旨在解决多尺度目标检测中的特征融合难题。该模型不仅继承了 ResNet-50 强大的特征提取能力，还通过“尺度注意力”机制，让模型能够动态选择最适合当前目标的特征尺度

#### 与 r18  的核心差异

|                | r18                    | r50-ASF                | 效果                                                   |
| :------------- | :--------------------- | :--------------------- | :----------------------------------------------------- |
| **主干网络**   | ResNet-18 (BasicBlock) | ResNet-50 (Bottleneck) | 更深的网络、更丰富的语义特征，提升对复杂目标的识别能力 |
| **特征融合**   | 普通 Concat            | Zoom_cat               | 以中尺度为核心，通过上/下采样同时聚合大/小尺度信息     |
| **注意力机制** | 仅 AIFI (单层)         | ScalSeq (ASF)          | 引入 3D 卷积处理尺度维度，实现跨尺度的动态特征选择     |

#### ResNet-50 Bottleneck

*   相比 r18 的 BasicBlock（两个 3x3 卷积），r50 使用了 Bottleneck 结构（1x1 -> 3x3 -> 1x1），第一个 1x1 卷积用于降维，减少计算量；中间的 3x3 卷积在低维空间进行特征处理；最后一个 1x1 卷积升维，恢复通道数
*   这种设计允许网络在保持计算成本可控的前提下，构建得更深（50层 vs 18层），从而提取出更具判别力的高层语义特征

#### Zoom_cat (聚焦式融合)

在特征融合阶段，r50-ASF 使用了 Zoom_cat 替代传统的 Concat，它同时接收三个尺度的特征（Large, Medium, Small），并将它们统一对齐到 Medium (中尺度)：
1.  Large (P3)：通过池化（Max+Avg）下采样，保留纹理细节
    - *池化（Max+Avg）下采样*：这是把大图变小的过程，就像把高清大图缩略显示。“Max”是只留最亮的那个点（保留最显著特征），“Avg”是算平均亮度（保留背景概貌），把两者结合起来下采样，既能保留关键纹理，又不会丢失整体信息，比单纯扔掉像素高明得多
2.  Small (P5)：通过插值上采样，提供全局语义
    - *插值上采样*：这是把小图变大的过程，因为小图变大后像素不够用，需要“脑补”中间的像素。插值就是根据周围已知像素的值，按一定数学规则（如双线性插值）估算出中间空缺像素的值，让模糊的小图平滑地放大
3.  Medium (P4)：保持不变

这种“中间对齐”策略比传统的“逐级上采样”更高效，它让模型在处理中等大小目标（最常见的情况）时，能同时“看清细节”和“理解背景”

#### ScalSeq (ASF 核心 - 尺度序列注意力)

ScalSeq 模块位于 Neck 的末端，负责生成最终的增强特征

*   ScalSeq 将不同尺度的特征图视为一个 3D 数据立方体：
    1.  堆叠 (Stacking)：将 P3, P4, P5 特征图对齐后，在新的维度（Scale Dimension）上堆叠，形成 `(Batch, Channel, 3, H, W)` 的张量
    2.  3D 卷积 (3D Conv)：使用 `(1, 1, 1)` 的 3D 卷积核在“尺度轴”上进行滑动。这使得模型能够学习不同尺度之间的关系
    3.  跨尺度优选 (Scale Pooling)：通过 3D 最大池化 (`MaxPool3d`)，在三个尺度中自动选择响应值最高的那个特征
        - *3D 最大池化 ( MaxPool3d )*：普通的池化是在一张图的长和宽上找最大值，而 3D 池化是在“长、宽、深”三个维度上找。在 ASF 模块中，这里的“深”指的是尺度维度 ，意味着模型会在大、中、小三个尺度的特征图中，为每一个像素位置自动挑选出那个反应最强烈的特征值，实现跨尺度的优选
*   直观理解：
    *   对于一个小物体，P3（高分辨率）的响应可能最强，模型会自动选择 P3 的特征
    *   对于一个大物体，P5（高语义）的响应更强，模型会选择 P5
    *   这实现了一种像素级的动态尺度选择，避免了传统 FPN 中“小目标被深层特征淹没”或“大目标缺乏语义信息”的问题
        - *FPN (特征金字塔网络)*：这是一种经典的架构，形状像金字塔。它把顶层提取到的高级语义特征（强语义但模糊）像瀑布一样传下来，与底层的高分辨率特征（弱语义但清晰）融合，让模型在每一层级都既有清晰的视野又有丰富的语义，是现代检测器的标配

### RT-DETR-StarNet

RT-DETR-StarNet 采用了 StarNet (CVPR 2024) 作为主干网络，摒弃了复杂的 LayerScale、复杂的注意力机制或多余的归一化层，仅通过简单的元素级乘法(Element-wise Multiplication)就实现了极高的性能

#### 与 r18 的核心差异

|                | r18              | StarNet        | 效果                                                         |
| :------------- | :--------------- | :------------- | :----------------------------------------------------------- |
| **基本单元**   | BasicBlock (CNN) | Star Block     | 从传统的“加法堆叠”转向“乘法映射”，特征表达维度更高           |
| **卷积核大小** | 3x3              | 7x7 Depthwise  | 大卷积核带来更大的有效感受野，能更好地捕捉大目标和上下文信息 |
| **激活机制**   | ReLU (线性激活)  | Star Operation | 通过 `ReLU(x1) * x2` 实现隐式的高维特征映射，无需增加额外计算量 |

#### Star Block

*   结构流程 ：
    1.  大核深度卷积 (7x7 DWConv)：首先使用 7x7 的卷积进行空间特征提取。相比 r18 的 3x3，它能“看”得更广，这对于检测任务（尤其是大物体和背景理解）至关重要
    2.  双路投影 (Dual Projection)：特征图被两个 1x1 卷积 (`f1`, `f2`) 映射为两个分支 $X_1$ 和 $X_2$
    3.  Star Operation ：
        $$ Y = \text{ReLU6}(X_1) \times X_2 $$
        *   这不仅是一个门控机制（类似 GLU），更可以看作是一种将特征映射到极高维非线性空间的手段
            - *门控机制和 GLU*：想象一个水龙头，水流（特征信息）经过时，开关（门）决定放多少水过去。GLU (Gated Linear Unit) 就是这样一种机制，它用一个分支产生“开关信号”（通常是0到1之间的数），去控制另一个分支的信息流量。StarNet 借用了这个思路，通过乘法让网络自己学会“什么信息该留，什么信息该丢”
        *   两个函数的乘积包含它们泰勒展开式中所有项的乘积，这意味着网络可以隐式地学习到极其复杂的特征交互，而不需要显式地增加网络深度或宽度
            - *特征交互*：在神经网络里，特征不能只各走各的路，必须“交流”才能产生高级智慧（比如“红色”特征和“圆形”特征交互，才能认出“苹果”）。StarNet 通过简单的元素级乘法（Element-wise Multiplication），让两个不同维度的特征直接相乘，这在数学上相当于把它们映射到了一个极其复杂的高维空间，让特征之间发生了剧烈的化学反应，从而极其高效地实现了深度融合
    4.  输出投影：最后通过一个 1x1 卷积 (`g`) 融合特征并输出

> 附各模型计算量与参数量
>
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SMPCGLU.yaml: 0.00 GFLOPs 12,835,628 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SHSA-EPGO-CGLU.yaml: 0.00 GFLOPs 13,700,276 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MambaVision.yaml: 0.00 GFLOPs 16,136,624 Params
> ultralytics/cfg/models/rt-detr/rtdetr-VSS.yaml: 0.00 GFLOPs 19,316,560 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-TransMamba.yaml: 0.00 GFLOPs 16,982,792 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-TVIM.yaml: 0.00 GFLOPs 14,399,216 Params
> ultralytics/cfg/models/rt-detr/rtdetr-GDSAFusion.yaml: 0.00 GFLOPs 23,105,392 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-GLVSS.yaml: 0.00 GFLOPs 15,604,642 Params
> ultralytics/cfg/models/rt-detr/rtdetr-mamba-T.yaml: 0.00 GFLOPs 9,170,920 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-GroupMamba.yaml: 0.00 GFLOPs 12,827,826 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-GroupMambaBlock.yaml: 0.00 GFLOPs 13,523,570 Params
> ultralytics/cfg/models/rt-detr/rtdetr-mamba-B.yaml: 0.00 GFLOPs 23,731,984 Params
> ultralytics/cfg/models/rt-detr/rtdetr-mamba-L.yaml: 0.00 GFLOPs 57,925,248 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SAVSS.yaml: 0.00 GFLOPs 14,276,528 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-VSS.yaml: 0.00 GFLOPs 41,064,812 Params
> ultralytics/cfg/models/rt-detr/rtdetr-LDConv.yaml: 0.00 GFLOPs 19,668,724 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-AKConv.yaml: 0.00 GFLOPs 37,432,396 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-EPGO.yaml: 0.00 GFLOPs 19,944,273 Params
>上述由于部分python模块未导入，未测出GFLOP，下次可导入后重测
> ultralytics/cfg/models/rt-detr/rtdetr-timm.yaml: 23.96 GFLOPs 9,642,960 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RepNCSPELAN.yaml: 26.82 GFLOPs 9,156,528 Params
> ultralytics/cfg/models/rt-detr/rtdetr-EfficientViT.yaml: 27.58 GFLOPs 10,804,048 Params
> ultralytics/cfg/models/rt-detr/rtdetr-fasternet.yaml: 28.83 GFLOPs 10,909,524 Params
> ultralytics/cfg/models/rt-detr/rtdetr-EfficientFormerv2.yaml: 29.79 GFLOPs 11,901,056 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-MSGA.yaml: 30.16 GFLOPs 11,772,365 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-HAFB.yaml: 30.33 GFLOPs 12,219,832 Params
> ultralytics/cfg/models/rt-detr/rtdetr-starnet.yaml: 32.16 GFLOPs 12,089,264 Params
> ultralytics/cfg/models/rt-detr/rtdetr-convnextv2.yaml: 32.21 GFLOPs 12,401,824 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RepNCSPELAN-CAA.yaml: 32.71 GFLOPs 10,454,960 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MutilBackbone-DAF.yaml: 34.36 GFLOPs 14,085,368 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DRBNCSPELAN.yaml: 35.28 GFLOPs 13,784,912 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DBBNCSPELAN.yaml: 36.14 GFLOPs 14,207,856 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Conv3XCNCSPELAN.yaml: 36.14 GFLOPs 14,207,856 Params
> ultralytics/cfg/models/rt-detr/rtdetr-OREPANCSPELAN.yaml: 36.14 GFLOPs 14,207,856 Params
> ultralytics/cfg/models/rt-detr/rtdetr-repvit.yaml: 36.68 GFLOPs 13,404,152 Params
> ultralytics/cfg/models/rt-detr/rtdetr-lsknet.yaml: 37.89 GFLOPs 12,663,134 Params
> ultralytics/cfg/models/rt-detr/rtdetr-mobilenetv4.yaml: 39.82 GFLOPs 11,411,728 Params
> ultralytics/cfg/models/rt-detr/rtdetr-WTConv.yaml: 40.65 GFLOPs 12,922,448 Params
> ultralytics/cfg/models/rt-detr/rtdetr-lsnet.yaml: 41.47 GFLOPs 19,778,416 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-wConv.yaml: 41.64 GFLOPs 13,998,896 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-PFDConv.yaml: 41.68 GFLOPs 12,779,388 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FDConv.yaml: 41.90 GFLOPs 14,191,928 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-PSFSConv.yaml: 41.99 GFLOPs 12,592,048 Params
> ultralytics/cfg/models/rt-detr/rtdetr-mambaout.yaml: 42.18 GFLOPs 16,002,056 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FourierConv.yaml: 42.26 GFLOPs 17,268,688 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DRB.yaml: 42.79 GFLOPs 13,801,680 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-ConvAttn.yaml: 42.89 GFLOPs 16,482,432 Params
> ultralytics/cfg/models/rt-detr/rtdetr-PConv.yaml: 43.18 GFLOPs 14,101,200 Params
> ultralytics/cfg/models/rt-detr/rtdetr-PConv-Rep.yaml: 43.18 GFLOPs 14,101,680 Params
> ultralytics/cfg/models/rt-detr/rtdetr-pkinet.yaml: 43.22 GFLOPs 12,821,648 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-ELGCA-CGLU.yaml: 43.26 GFLOPs 12,923,364 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-gConv.yaml: 43.57 GFLOPs 13,003,760 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-CAMixer.yaml: 43.78 GFLOPs 13,276,638 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-HFERB.yaml: 43.83 GFLOPs 13,077,168 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MobileMamba.yaml: 43.86 GFLOPs 17,217,922 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-DIMB.yaml: 43.89 GFLOPs 13,309,156 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FasterFDConv.yaml: 44.16 GFLOPs 13,307,094 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-StripCGLU.yaml: 44.22 GFLOPs 13,407,524 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FasterSFSConv.yaml: 44.32 GFLOPs 13,213,424 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MSMHSA-CGLU.yaml: 44.42 GFLOPs 13,603,748 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RGCSPELAN.yaml: 44.80 GFLOPs 13,945,968 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-EBlock.yaml: 44.93 GFLOPs 13,862,640 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-LSBlock.yaml: 45.24 GFLOPs 13,416,952 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-DBlock.yaml: 45.72 GFLOPs 13,733,104 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SFSConv.yaml: 46.31 GFLOPs 13,858,288 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-AddutuveBlock-CGLU.yaml: 46.46 GFLOPs 14,068,708 Params
> ultralytics/cfg/models/rt-detr/rtdetr-faster-CGLU.yaml: 46.54 GFLOPs 15,504,672 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-FDConv.yaml: 46.64 GFLOPs 22,068,526 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-GlobalFilter.yaml: 46.78 GFLOPs 16,938,852 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout.yaml: 46.93 GFLOPs 13,886,702 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-RCB.yaml: 46.97 GFLOPs 13,987,376 Params
> ultralytics/cfg/models/rt-detr/rtdetr-iRMB-Cascaded.yaml: 47.14 GFLOPs 15,383,920 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-DIMB-HyperACE.yaml: 47.15 GFLOPs 15,331,435 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-IEL.yaml: 47.20 GFLOPs 13,917,512 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-ELGCA.yaml: 47.20 GFLOPs 13,965,808 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-AP.yaml: 47.27 GFLOPs 13,433,192 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AKConv.yaml: 47.31 GFLOPs 15,401,328 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DualConv.yaml: 47.64 GFLOPs 15,968,720 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-LSConv.yaml: 47.67 GFLOPs 14,090,678 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-Strip.yaml: 47.97 GFLOPs 14,419,248 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ContextGuided.yaml: 47.98 GFLOPs 16,618,952 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FCM.yaml: 48.01 GFLOPs 6,588,700 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CSP-PMSFA.yaml: 48.05 GFLOPs 14,193,232 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-EfficientVIM.yaml: 48.26 GFLOPs 14,566,326 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-EfficientVIM-CGLU.yaml: 48.27 GFLOPs 14,588,138 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CGRFPN.yaml: 48.52 GFLOPs 19,330,448 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DRBC3.yaml: 48.63 GFLOPs 18,243,408 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CSP-MutilScaleEdgeInformationEnhance.yaml: 48.73 GFLOPs 14,558,128 Params
> ultralytics/cfg/models/rt-detr/rtdetr-EMBSFPN-SC.yaml: 48.97 GFLOPs 18,040,801 Params
> ultralytics/cfg/models/rt-detr/rtdetr-EMBSFPN.yaml: 48.97 GFLOPs 18,040,801 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-SFSC.yaml: 48.97 GFLOPs 14,524,110 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-DynamicFilter.yaml: 49.15 GFLOPs 15,024,436 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-LEGM.yaml: 49.38 GFLOPs 14,534,048 Params
> ultralytics/cfg/models/rt-detr/rtdetr-bifpn-GLSA.yaml: 49.44 GFLOPs 17,783,621 Params
> ultralytics/cfg/models/rt-detr/rtdetr-iRMB.yaml: 49.46 GFLOPs 16,515,920 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MSM.yaml: 49.67 GFLOPs 16,695,872 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CSP-MutilScaleEdgeInformationSelect.yaml: 49.73 GFLOPs 14,731,042 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MogaBlock.yaml: 49.79 GFLOPs 14,622,470 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Faster-Rep.yaml: 49.85 GFLOPs 16,886,960 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Faster.yaml: 49.85 GFLOPs 16,886,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-AddutuveBlock.yaml: 50.19 GFLOPs 15,072,752 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SHSA.yaml: 50.26 GFLOPs 13,617,328 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SHSA-CGLU.yaml: 50.29 GFLOPs 13,642,672 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SHSA-EPGO.yaml: 50.37 GFLOPs 13,691,828 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DGCST.yaml: 50.41 GFLOPs 18,598,224 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RetBlockC3.yaml: 50.52 GFLOPs 18,620,240 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DGCST2.yaml: 50.58 GFLOPs 18,632,272 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FFCM.yaml: 50.89 GFLOPs 16,660,704 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FDConvC3.yaml: 50.89 GFLOPs 26,082,552 Params
> ultralytics/cfg/models/rt-detr/rtdetr-iRMB-DRB.yaml: 51.03 GFLOPs 16,642,896 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MobileMamba.yaml: 51.16 GFLOPs 14,278,640 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FMB.yaml: 51.30 GFLOPs 15,257,232 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-RepNCSPELAN.yaml: 51.36 GFLOPs 19,705,548 Params
> ultralytics/cfg/models/rt-detr/rtdetr-gConvC3.yaml: 51.46 GFLOPs 18,813,264 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FAT.yaml: 51.61 GFLOPs 15,310,776 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-vHeat.yaml: 51.68 GFLOPs 15,720,368 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MAN-FasterCGLU.yaml: 51.75 GFLOPs 19,610,324 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Faster-EMA.yaml: 51.79 GFLOPs 16,996,240 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Faster-Rep-EMA.yaml: 51.79 GFLOPs 16,996,720 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-ETB.yaml: 51.80 GFLOPs 14,939,516 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-mambaout-UniRepLK.yaml: 51.94 GFLOPs 15,323,998 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-VSSD.yaml: 52.29 GFLOPs 15,071,248 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MAN-Faster.yaml: 52.75 GFLOPs 19,922,000 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FDT.yaml: 52.80 GFLOPs 15,448,312 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-FCA.yaml: 52.90 GFLOPs 15,516,336 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MBRConv3.yaml: 52.90 GFLOPs 15,512,240 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-LFEM.yaml: 53.33 GFLOPs 15,049,100 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-RAB.yaml: 53.53 GFLOPs 13,413,826 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-HDRAB.yaml: 53.54 GFLOPs 15,726,600 Params
> ultralytics/cfg/models/rt-detr/rtdetr-slimneck.yaml: 53.59 GFLOPs 19,399,888 Params
> ultralytics/cfg/models/rt-detr/rtdetr-fadc.yaml: 53.63 GFLOPs 20,112,750 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HSFPN.yaml: 53.66 GFLOPs 18,215,504 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CA-HSFPN.yaml: 53.68 GFLOPs 18,183,424 Params
> ultralytics/cfg/models/rt-detr/rtdetr-PST.yaml: 53.80 GFLOPs 19,716,304 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CFPT.yaml: 53.84 GFLOPs 18,315,056 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ELA-HSFPN.yaml: 54.04 GFLOPs 20,029,392 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-ESC.yaml: 54.19 GFLOPs 18,110,608 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CFPT-P3456.yaml: 54.20 GFLOPs 19,700,816 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SFHF.yaml: 54.91 GFLOPs 18,006,408 Params
> ultralytics/cfg/models/rt-detr/rtdetr-wConv.yaml: 54.93 GFLOPs 19,973,968 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SRFD.yaml: 55.33 GFLOPs 19,727,088 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MAN-Star.yaml: 55.53 GFLOPs 20,771,984 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SOEP-PST.yaml: 55.63 GFLOPs 19,852,144 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HWD.yaml: 55.98 GFLOPs 19,319,120 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MFM.yaml: 55.98 GFLOPs 19,810,640 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MFMMAFPN.yaml: 56.00 GFLOPs 22,881,424 Params
> ultralytics/cfg/models/rt-detr/rtdetr-LFEC3.yaml: 56.31 GFLOPs 19,853,136 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MAFPN.yaml: 56.67 GFLOPs 23,027,600 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RFPN.yaml: 56.70 GFLOPs 19,681,616 Params
> ultralytics/cfg/models/rt-detr/rtdetr-PSConv.yaml: 56.81 GFLOPs 19,515,984 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CAA-HSFPN.yaml: 56.85 GFLOPs 18,663,248 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FreqFFPN.yaml: 56.96 GFLOPs 19,948,436 Params
> ultralytics/cfg/models/rt-detr/rtdetr-IELC3.yaml: 57.01 GFLOPs 19,917,072 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-EDFFN.yaml: 57.19 GFLOPs 19,861,584 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-SEFFN.yaml: 57.19 GFLOPs 19,853,392 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DyT.yaml: 57.29 GFLOPs 19,974,482 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MBRConv3C3.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-LPE.yaml: 57.29 GFLOPs 20,076,880 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Conv3XC.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DBBC3.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFIRepBN.yaml: 57.29 GFLOPs 19,975,506 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DPCF.yaml: 57.29 GFLOPs 19,974,994 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Conv3XCC3.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FourierConv.yaml: 57.29 GFLOPs 22,083,920 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DBB.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-WDBB.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DeepDBB.yaml: 57.29 GFLOPs 19,974,480 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Ortho.yaml: 57.29 GFLOPs 20,061,520 Params
> ultralytics/cfg/models/rt-detr/rtdetr-fsa.yaml: 57.29 GFLOPs 22,653,974 Params
> ultralytics/cfg/models/rt-detr/rtdetr-attention.yaml: 57.29 GFLOPs 20,631,400 Params
> ultralytics/cfg/models/rt-detr/rtdetr-KAN.yaml: 57.29 GFLOPs 43,565,920 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ContextGuideFPN.yaml: 57.30 GFLOPs 20,105,552 Params
> ultralytics/cfg/models/rt-detr/rtdetr-KANC3.yaml: 57.31 GFLOPs 27,050,880 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DySample.yaml: 57.32 GFLOPs 19,990,928 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DEConv.yaml: 57.34 GFLOPs 19,978,320 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-Mona.yaml: 57.36 GFLOPs 20,062,032 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-SHSA.yaml: 57.36 GFLOPs 19,798,608 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CascadedGroupAttention.yaml: 57.37 GFLOPs 19,804,628 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-SHSA-EPGO.yaml: 57.39 GFLOPs 19,831,633 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-TSSA.yaml: 57.39 GFLOPs 19,842,648 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-MSLA.yaml: 57.40 GFLOPs 19,804,308 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SDI.yaml: 57.42 GFLOPs 20,002,128 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-HiLo.yaml: 57.43 GFLOPs 19,940,944 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-timm.yaml: 57.45 GFLOPs 18,793,260 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AdditiveTokenMixer.yaml: 57.46 GFLOPs 20,050,768 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-MSMHSA.yaml: 57.46 GFLOPs 20,010,064 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-EfficientAdditive.yaml: 57.50 GFLOPs 19,974,736 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DAttention.yaml: 57.50 GFLOPs 19,977,936 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DPB.yaml: 57.51 GFLOPs 19,975,304 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-EDFFN-Mona-DyT.yaml: 57.53 GFLOPs 20,117,650 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFFN-Mona-DyT.yaml: 57.53 GFLOPs 20,109,458 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-FMFFN.yaml: 57.56 GFLOPs 20,149,136 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola.yaml: 57.56 GFLOPs 20,142,992 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-CGLU.yaml: 57.56 GFLOPs 20,149,640 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CARAFE.yaml: 57.59 GFLOPs 20,122,776 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DHSA.yaml: 57.61 GFLOPs 20,116,056 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-DTAB.yaml: 57.61 GFLOPs 18,880,576 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ETB.yaml: 57.67 GFLOPs 20,118,768 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CTrans.yaml: 57.71 GFLOPs 29,623,248 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-SEFN.yaml: 57.75 GFLOPs 21,430,352 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FDT.yaml: 57.78 GFLOPs 20,323,090 Params
> ultralytics/cfg/models/rt-detr/rtdetr-BIMAFPN.yaml: 57.82 GFLOPs 20,207,009 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SMAFB-CGLU.yaml: 57.89 GFLOPs 21,018,848 Params
> ultralytics/cfg/models/rt-detr/rtdetr-slimneck-ASF.yaml: 57.92 GFLOPs 19,704,528 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFN.yaml: 58.01 GFLOPs 21,598,864 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-PKI.yaml: 58.05 GFLOPs 17,761,904 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFN-Mona.yaml: 58.08 GFLOPs 21,686,416 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Pola-SEFN-Mona-DyT.yaml: 58.08 GFLOPs 21,686,418 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA.yaml: 58.17 GFLOPs 20,811,738 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DWRC3-DRB.yaml: 58.23 GFLOPs 21,149,520 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ASF-P2.yaml: 58.31 GFLOPs 15,050,672 Params
> ultralytics/cfg/models/rt-detr/rtdetr-EUCB-SC.yaml: 58.38 GFLOPs 20,111,184 Params
> ultralytics/cfg/models/rt-detr/rtdetr-EUCB.yaml: 58.38 GFLOPs 20,111,184 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HSPAN.yaml: 58.45 GFLOPs 20,741,456 Params
> ultralytics/cfg/models/rt-detr/rtdetr-Star.yaml: 58.49 GFLOPs 20,178,000 Params
> ultralytics/cfg/models/rt-detr/rtdetr-PACAPN.yaml: 58.54 GFLOPs 19,977,040 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA-SEFN.yaml: 58.62 GFLOPs 22,267,610 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA-SEFN-Mona-DyT.yaml: 58.69 GFLOPs 22,355,164 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AIFI-ASSA-SEFN-Mona.yaml: 58.69 GFLOPs 22,355,162 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DTAB.yaml: 59.00 GFLOPs 21,847,796 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CGAFusion.yaml: 59.55 GFLOPs 20,528,345 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CDFA.yaml: 59.60 GFLOPs 22,803,552 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MAN.yaml: 59.62 GFLOPs 22,068,560 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HS-FPN.yaml: 59.71 GFLOPs 19,739,728 Params
> ultralytics/cfg/models/rt-detr/rtdetr-GlobalEdgeInformationTransfer1.yaml: 59.76 GFLOPs 21,210,768 Params
> ultralytics/cfg/models/rt-detr/rtdetr-slimneck-WFU.yaml: 59.88 GFLOPs 22,958,800 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ReCalibrationFPN-P345.yaml: 59.89 GFLOPs 20,581,200 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RFCBAMConv.yaml: 60.07 GFLOPs 20,419,120 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SWC.yaml: 60.17 GFLOPs 16,032,720 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RFAConv.yaml: 60.23 GFLOPs 20,340,816 Params
> ultralytics/cfg/models/rt-detr/rtdetr-goldyolo.yaml: 60.29 GFLOPs 22,358,736 Params
> ultralytics/cfg/models/rt-detr/rtdetr-p6-CTrans.yaml: 60.38 GFLOPs 49,538,032 Params
> ultralytics/cfg/models/rt-detr/rtdetr-RFCAConv.yaml: 60.45 GFLOPs 20,435,800 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-PPA.yaml: 60.56 GFLOPs 18,053,982 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HyperACE.yaml: 60.76 GFLOPs 21,963,991 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SDFM.yaml: 60.79 GFLOPs 21,192,272 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CSP-FreqSpatial.yaml: 60.93 GFLOPs 20,571,440 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DWRC3.yaml: 61.01 GFLOPs 24,631,632 Params
> ultralytics/cfg/models/rt-detr/rtdetr-DySnake.yaml: 61.15 GFLOPs 27,958,680 Params
> ultralytics/cfg/models/rt-detr/rtdetr-WaveletPool.yaml: 61.22 GFLOPs 19,212,192 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CAB.yaml: 61.36 GFLOPs 20,795,760 Params
> ultralytics/cfg/models/rt-detr/rtdetr-rmt.yaml: 61.46 GFLOPs 21,457,616 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HyperCompute-MFM.yaml: 61.56 GFLOPs 22,024,528 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-SMAFB.yaml: 61.77 GFLOPs 28,492,976 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ASF.yaml: 61.79 GFLOPs 20,254,544 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ASF-Dynamic.yaml: 61.84 GFLOPs 20,295,664 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ContextGuidedDown.yaml: 62.06 GFLOPs 22,426,512 Params
> ultralytics/cfg/models/rt-detr/rtdetr-AggregatedAtt.yaml: 62.70 GFLOPs 23,300,560 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SOEP-MFM-RFPN.yaml: 62.74 GFLOPs 20,110,928 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HyperCompute.yaml: 63.23 GFLOPs 22,375,760 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SOEP-MFM.yaml: 63.32 GFLOPs 20,403,792 Params
> ultralytics/cfg/models/rt-detr/rtdetr-GlobalEdgeInformationTransfer3.yaml: 63.52 GFLOPs 21,908,624 Params
> ultralytics/cfg/models/rt-detr/rtdetr-GlobalEdgeInformationTransfer2.yaml: 63.52 GFLOPs 21,908,624 Params
> ultralytics/cfg/models/rt-detr/rtdetr-goldyolo-asf.yaml: 63.52 GFLOPs 22,589,648 Params
> ultralytics/cfg/models/rt-detr/rtdetr-WFU.yaml: 63.58 GFLOPs 23,533,392 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FDPN-DASI.yaml: 63.71 GFLOPs 21,271,248 Params
> ultralytics/cfg/models/rt-detr/rtdetr-iRMB-SWC.yaml: 63.97 GFLOPs 17,582,672 Params
> ultralytics/cfg/models/rt-detr/rtdetr-GLSA.yaml: 64.02 GFLOPs 22,058,105 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ReCalibrationFPN-P3456.yaml: 64.35 GFLOPs 34,948,464 Params
> ultralytics/cfg/models/rt-detr/rtdetr-LoGStem.yaml: 64.42 GFLOPs 20,005,184 Params
> ultralytics/cfg/models/rt-detr/rtdetr-bifpn.yaml: 64.63 GFLOPs 20,401,628 Params
> ultralytics/cfg/models/rt-detr/rtdetr-TransNeXt.yaml: 64.87 GFLOPs 21,146,400 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SOEP-RFPN.yaml: 64.94 GFLOPs 20,297,552 Params
> ultralytics/cfg/models/rt-detr/rtdetr-mpcafsa.yaml: 65.42 GFLOPs 26,032,948 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CAFMFusion.yaml: 65.46 GFLOPs 21,514,483 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SOEP.yaml: 65.52 GFLOPs 20,590,416 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CSFCN.yaml: 65.80 GFLOPs 21,205,856 Params
> ultralytics/cfg/models/rt-detr/rtdetr-FDPN.yaml: 66.42 GFLOPs 22,337,488 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HAFB-1.yaml: 67.85 GFLOPs 22,888,784 Params
> ultralytics/cfg/models/rt-detr/rtdetr-HAFB-2.yaml: 70.15 GFLOPs 23,804,240 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-JDPM.yaml: 70.84 GFLOPs 20,246,180 Params
> ultralytics/cfg/models/rt-detr/rtdetr-msga.yaml: 71.39 GFLOPs 22,585,445 Params
> ultralytics/cfg/models/rt-detr/rtdetr-PSFM.yaml: 71.54 GFLOPs 22,540,502 Params
> ultralytics/cfg/models/rt-detr/rtdetr-C2f-MBRConv5.yaml: 72.93 GFLOPs 20,886,192 Params
> ultralytics/cfg/models/rt-detr/rtdetr-MBRConv5C3.yaml: 73.02 GFLOPs 23,120,208 Params
> ultralytics/cfg/models/rt-detr/rtdetr-p2.yaml: 78.81 GFLOPs 18,649,264 Params
> ultralytics/cfg/models/rt-detr/rtdetr-ReCalibrationFPN-P2345.yaml: 85.03 GFLOPs 19,542,832 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r34.yaml: 89.15 GFLOPs 31,227,972 Params
> ultralytics/cfg/models/rt-detr/rtdetr-CSwomTramsformer.yaml: 90.22 GFLOPs 30,585,968 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DGCST.yaml: 94.51 GFLOPs 35,039,084 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-m.yaml: 96.41 GFLOPs 36,415,340 Params
> ultralytics/cfg/models/rt-detr/rtdetr-SwinTransformer.yaml: 97.31 GFLOPs 36,414,826 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DRB.yaml: 100.80 GFLOPs 30,986,284 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-WTConv.yaml: 101.25 GFLOPs 31,413,868 Params
> ultralytics/cfg/models/rt-detr/rtdetr-l-GhostHGNetV2.yaml: 101.50 GFLOPs 30,983,148 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-PConv.yaml: 101.67 GFLOPs 31,512,364 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-PConv-Rep.yaml: 101.67 GFLOPs 31,513,308 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-RetBlockC3.yaml: 102.58 GFLOPs 36,657,516 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-PACAPN.yaml: 102.64 GFLOPs 36,417,900 Params
> ultralytics/cfg/models/rt-detr/rtdetr-l-RepHGNetV2.yaml: 103.62 GFLOPs 32,073,516 Params
> ultralytics/cfg/models/rt-detr/rtdetr-l.yaml: 103.78 GFLOPs 32,148,396 Params
> ultralytics/cfg/models/rt-detr/rtdetr-l-attention.yaml: 103.85 GFLOPs 33,193,700 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-ContextGuided.yaml: 106.73 GFLOPs 34,217,720 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-faster-CGLU.yaml: 108.39 GFLOPs 34,048,136 Params
> ultralytics/cfg/models/rt-detr/rtdetr-VanillaNet.yaml: 110.48 GFLOPs 21,811,376 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DualConv.yaml: 110.60 GFLOPs 34,884,268 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-iRMB-Cascaded.yaml: 112.43 GFLOPs 33,846,700 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Faster-Rep.yaml: 115.00 GFLOPs 36,543,196 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Faster.yaml: 115.00 GFLOPs 36,542,252 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-iRMB-DRB.yaml: 116.98 GFLOPs 36,136,556 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Faster-Rep-EMA.yaml: 118.89 GFLOPs 36,741,564 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Faster-EMA.yaml: 118.89 GFLOPs 36,740,620 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-iRMB.yaml: 120.32 GFLOPs 35,876,460 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-fadc.yaml: 124.45 GFLOPs 42,325,913 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-SRFD.yaml: 127.94 GFLOPs 41,871,116 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DeepDBB.yaml: 129.89 GFLOPs 42,118,508 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Conv3XC.yaml: 129.89 GFLOPs 42,118,508 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DBB.yaml: 129.89 GFLOPs 42,118,508 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-WDBB.yaml: 129.89 GFLOPs 42,118,508 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50.yaml: 129.89 GFLOPs 42,118,508 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Ortho.yaml: 129.90 GFLOPs 44,633,452 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-KAN.yaml: 129.91 GFLOPs 84,582,800 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DEConv.yaml: 129.98 GFLOPs 42,126,060 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-attention.yaml: 130.23 GFLOPs 44,036,340 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-Star.yaml: 132.10 GFLOPs 42,518,764 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-RFCBAMConv.yaml: 132.60 GFLOPs 42,613,452 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-RFAConv.yaml: 132.76 GFLOPs 42,526,316 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-RFCAConv.yaml: 132.98 GFLOPs 42,627,828 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-SWC.yaml: 133.43 GFLOPs 35,373,996 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-AggregatedAtt.yaml: 134.35 GFLOPs 46,094,156 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-iRMB-SWC.yaml: 139.41 GFLOPs 37,933,036 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-ASF.yaml: 142.57 GFLOPs 43,234,156 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-DySnake.yaml: 143.41 GFLOPs 51,776,984 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r50-bifpn.yaml: 158.99 GFLOPs 44,086,136 Params
> ultralytics/cfg/models/rt-detr/rtdetr-x.yaml: 222.82 GFLOPs 65,632,092 Params
> ultralytics/cfg/models/rt-detr/rtdetr-r101.yaml: 247.43 GFLOPs 74,819,948 Params