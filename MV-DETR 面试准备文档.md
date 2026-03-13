# MV-DETR 面试准备文档

> 适用岗位：L5 MLE（Computer Vision / ML Systems）
> 项目代号：MV-DETR（Multi-View Detection Transformer）
> 代码位置：`caper/cv/barcode_localization/mmdetection/`
> 主要作者：David Dong (huaiyuya)

---

## 一、项目简介（30秒版本）

我们在 Instacart Caper 购物车上部署了一套实时商品检测系统。购物车配备4路摄像头，从不同角度拍摄货架区域。当前系统对每路相机独立做 2D 检测，然后通过 epipolar 几何约束做跨视角关联。

这套方案有三个核心问题：
1. **遮挡**：某个相机看不到的物体只能靠另一个相机检测，但独立检测后的关联极易失败
2. **标定漂移**：生产环境中相机位置会偏移，硬编码的 epipolar 阈值会失效
3. **稠密场景**：多件相似商品紧密堆叠时，threshold-based 关联产生大量错误对应

MV-DETR 的核心思路是：**把跨视角关联从后处理的规则系统变成模型内部的 attention 机制**。每个 object query 代表一个伪3D物体，在一次 forward pass 中同时 attend 4路相机，直接输出所有视角的 bbox 和可见性，天然保证跨视角一致性。

当前阶段完成了单视角验证（SVDetr3D），正在推进多视角版本（MVDetr3D）。

---

## 二、系统架构

### 2.1 整体流程图

```
输入: (B, 4, 1, H, W)  ← 4路相机灰度图，grayscale
         │
         ▼ extract_feat()  [共享权重]
┌──────────────────────────────────┐
│  SimpleNet Backbone              │  ← 冻结，复用已有2D检测模型权重
│  → (B×4, C, Hi, Wi)             │
│  NanoDetFPN Neck                 │  ← 冻结
│  → 4 levels: (B×4, 64, Hi, Wi)  │
└──────────────────────────────────┘
         │  reshape: (B×4, ...) → (B, 4, 64, Hi, Wi)
         ▼
  mlvl_feats: list of 4 tensors, each (B, 4, 64, Hi, Wi)
         │
         ▼ MVDetr3DTransformer
┌─────────────────────────────────────────────────────────┐
│  query_embedding: Embedding(50, 128)                    │
│    split → query_pos (50, 64) + query (50, 64)          │
│                                                         │
│  reference_points = Linear(64→8)(query_pos).sigmoid()   │
│    reshape → (B, 50, 4_views, 2)                        │
│    含义: 每个query在每个view的初始位置猜测               │
│                                                         │
│  ┌── Decoder Layer × 6 ─────────────────────────────┐  │
│  │                                                   │  │
│  │  1. Self-Attention                               │  │
│  │     query间建模物体间关系                         │  │
│  │     MultiheadAttention(embed=64, heads=8) + norm  │  │
│  │                                                   │  │
│  │  2. Cross-Attention: MVDetr3DCrossAtten           │  │
│  │     对每个view分别用 grid_sample 在                │  │
│  │     reference_point 位置稀疏采特征                 │  │
│  │     reference: (B, 50, 4, 2)                      │  │
│  │     每view每level采1点 → attention weight加权融合  │  │
│  │     position_encoder: MLP(2→64→64) 编码坐标       │  │
│  │     + norm                                        │  │
│  │                                                   │  │
│  │  3. FFN(64→128→64) + norm                        │  │
│  │                                                   │  │
│  │  4. Box Refine (with_box_refine=True):            │  │
│  │     Δ = reg_branch[layer](hs)                     │  │
│  │     new_ref = sigmoid(Δ[:,:,:,:2]                 │  │
│  │                + inverse_sigmoid(old_ref))        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │  6层 intermediate outputs
         ▼ MVDetr3DHead (每层独立预测分支)
┌──────────────────────────────────────────────────────┐
│  cls_branch[l]:   Linear(64→64)→LN→ReLU→Linear(64→2)│
│  → cls_scores: (B, 50, 2)                           │
│                                                      │
│  reg_branch[l]:   Linear(64→64)→ReLU→Linear(64→16)  │
│  → bbox_preds: (B, 50, 4, 4) cxcywh [per-view]      │
│    cx,cy += inverse_sigmoid(reference) → sigmoid     │
│                                                      │
│  view_branch[l]:  Linear(64→64)→ReLU→Linear(64→4)   │
│  → view_scores: (B, 50, 4)  ← 每view的可见性预测    │
└──────────────────────────────────────────────────────┘
         │
         ▼ 推理输出（最后一层）
  per-query: score + bboxes×4 + visibility×4
```

### 2.2 两阶段验证策略

| 阶段 | 名称 | 目标 | 成功标准 |
|------|------|------|---------|
| Phase 1 | Code Verification | overfit小数据集验证实现正确性 | loss收敛，输出合理 |
| Phase 2 | SVDetr3D（单视角） | 验证 Transformer decoder 能否替代 NanoDet head | 2D mAP ≥ NanoDet baseline |
| Phase 3 | MVDetr3D（多视角） | 验证跨视角关联可学习 | Association Precision/Recall ≥ threshold-based |
| Phase 4 | 量化+部署 | TRT 转换，满足延迟要求 | 延迟增幅 ≤ 40% |

---

## 三、核心设计

### 3.1 伪3D（Pseudo-3D）设计

每个 query 代表现实世界中的**一个物体实例**，但不预测显式的 3D 坐标，而是预测该物体在4个相机中的2D投影：

```
query_q 对应"货架上第3件饮料":
  cls_score: 0.92 (barcode类)
  cam_0: bbox=(120,80,50,80), visible=True
  cam_1: bbox=(340,90,45,75), visible=True
  cam_2: bbox=(??),           visible=False  ← 遮挡
  cam_3: bbox=(560,85,48,78), visible=True
```

**为什么叫"伪"**：表现得像3D（跨视角一致、感知遮挡），但内部没有世界坐标 (x,y,z)。这是针对购物车场景的务实选择——业务上不需要3D坐标，只需要知道哪个相机看到了哪个物品。

### 3.2 Reference Points 机制

**作用**：指导 cross-attention 做稀疏采样，避免 dense attention 扫全图。

**形状**：`(B, num_query, num_views, 2)`

- 同一物体在4个相机里的投影位置不同，所以每个 query 为每个 view 维护独立的 (cx, cy)
- 初始化：`Linear(64→8)(query_pos).sigmoid()`，随机初始化后靠数据学习
- 6层迭代精化：

```python
# 在 logit 空间做残差加法，保证结果在 [0,1]
new_ref = sigmoid(inverse_sigmoid(old_ref) + Δ[:,:,:,:2])
```

**为什么用 inverse_sigmoid**：直接相加可能超出 [0,1]，在 logit 空间加法后经 sigmoid 压缩，保证坐标始终合法。

### 3.3 Cross-Attention 采样细节

```python
# MVDetr3DCrossAtten.feature_sampling()
ref_cam = (reference_points - 0.5) * 2  # [0,1] → [-1,1]，grid_sample格式

for level in range(4):  # 4个FPN层
    sampled = F.grid_sample(feat[level], ref_cam)
    # 在 reference_point 精确位置取1个特征点

# 4个level的采样特征 → attention_weight加权 → output_proj
attention_weights = Linear(64, 4*1)(query)  # 4levels × 1point
```

每个 query 对每个 view 只采 **4点**（4 level × 1 point），极度轻量。

---

## 四、Loss 设计

### 4.1 四种 Loss

```
L_total = L_cls + L_view + L_bbox + L_iou
```

| Loss | 类型 | weight | 监督信号 | 特殊设计 |
|------|------|--------|---------|---------|
| `loss_cls` | FocalLoss (sigmoid) | 2.0 | 物体类别 | γ=2, α=0.25 |
| `loss_view` | BCE (sigmoid) | 1.0 | 每view可见性 | 正负样本都参与 |
| `loss_bbox` | L1Loss | 2.0 | per-view bbox坐标 | **visibility加权** |
| `loss_iou` | GIoULoss | 2.0 | per-view bbox形状 | **visibility加权** |

### 4.2 Visibility-Weighted Bbox Loss（关键设计）

```python
for view_idx in range(4):
    weight = gt_view_visible[:, view_idx].float()  # 不可见=0，可见=1
    loss_bbox += L1(pred_view, gt_view, weight) / num_pos
    loss_iou  += GIoU(pred_xyxy, gt_xyxy, weight) / num_pos
```

**设计意图**：遮挡视角没有有效 GT bbox，强行计算 bbox loss 会引入错误梯度。用 visibility 掩码让模型只在有真值的视角学习定位，不可见视角只被 `loss_view` 约束（学会预测"不可见"）。

### 4.3 Intermediate Supervision（6层都算 Loss）

```
Layer 0 → d0.loss_cls, d0.loss_view, d0.loss_bbox, d0.loss_iou
Layer 1 → d1.loss_cls, ...
...
Layer 4 → d4.loss_cls, ...
Layer 5 → loss_cls, loss_view, loss_bbox, loss_iou  (主loss，monitor)
```

**原因**：Transformer decoder 本来收敛慢，只有最后层有梯度时，梯度穿过5层会消失。每层都有 loss 相当于给每层直通梯度通道，大幅加速收敛。每层独立匹配（不共享 assignment），因为不同层 reference points 精度不同。

---

## 五、Hungarian Matching 实现

### 5.1 Cost Matrix 构建

**shape: (num_queries=50, num_gts=M)**，由4项累加：

```
total_cost = cls_cost + l1_cost + giou_cost + view_cost
```

#### 分类 cost（视角无关，算一次）
```python
# Focal Loss cost
pos_cost = -log(p) * α * (1-p)^γ
neg_cost = -log(1-p) * (1-α) * p^γ
cls_cost[q, g] = pos_cost[q, gt_label[g]] - neg_cost[q, gt_label[g]]
# 语义: query q 把 GT g 的类别预测准确，需要付出多少代价
```

#### Bbox cost（对可见view累加，不可见view清零）
```python
for view in range(4):
    l1_cost  = torch.cdist(pred_bboxes[:,v,:], gt_bboxes[:,v,:], p=1)  # (50, M)
    giou_cost = -bbox_overlaps(pred_xyxy, gt_xyxy, mode='giou')        # (50, M)

    # 不可见view的cost清零，不影响匹配决策
    l1_cost   *= gt_view_visible[:, v].float()   # broadcast: (50,M) * (M,)
    giou_cost *= gt_view_visible[:, v].float()

    total_cost += l1_cost + giou_cost
```

#### View visibility cost（MV独有）
```python
for view in range(4):
    pred_v = pred_view_scores[:, v].sigmoid()   # (50,)
    gt_v   = gt_view_visible[:, v].float()      # (M,)
    bce = -(gt_v * log(pred_v) + (1-gt_v) * log(1-pred_v))  # (50, M)
    view_cost += bce
# 语义: query q 对 GT g 的可见性预测是否正确
```

### 5.2 Hungarian Algorithm

```python
cost_matrix = total_cost.detach().cpu().numpy()  # detach: 匹配不需要梯度
row_inds, col_inds = linear_sum_assignment(cost_matrix)  # scipy, O(n³)

# 编码结果
assigned_gt_inds = zeros(50)
assigned_gt_inds[row_inds] = col_inds + 1   # 0=背景, 1~M=正样本(1-indexed)
```

### 5.3 Cost Weight 设置逻辑

```python
match_costs = [
    FocalLossCost(weight=2.0),    # 分类: 主导信号
    BBoxL1Cost(weight=0.5),       # L1: 对4view累加4次，等效weight=2.0
    IoUCost(giou, weight=2.0),    # GIoU: 值域[-1,1]量级小，weight补偿
]
view_cost_weight = 1.0
```

---

## 六、训练稳定性设计

### 6.1 已踩坑 & 解决方案（v1 config）

| 问题 | 现象 | 原因 | 解决 |
|------|------|------|------|
| LR爆炸 | epoch 2 loss爆炸 | auto_scale_lr在8GPU×96BS下把LR放大8倍至8e-4 | 禁用auto_scale_lr，固定LR=1e-4 |
| 梯度爆炸 | loss震荡不收敛 | value-based clip(35)对Transformer不够稳定 | 改norm-based clip(max_norm=0.1) |
| 特征退化 | mAP低于NanoDet基线 | backbone LR与decoder相同，破坏预训练特征 | backbone LR=0.1×base_lr |

### 6.2 双头结构（Aux Head）

```
backbone
  ├── main_neck → SVDetr3DHead (Transformer decoder，主任务)
  └── aux_neck  → SimpleConvHead (NanoDet-style，辅助任务)
                  [concat: 128ch = 64(main) + 64(aux)]
```

- `detach_epoch=10`：前10 epoch aux和main共享backbone梯度，帮助encoder特征适应
- 10 epoch之后 aux path detach，避免NanoDet的anchor-based梯度干扰Transformer的学习

**设计意图**：NanoDet head 是成熟的anchor-based检测头，在训练初期给 backbone 提供稳定的监督信号，帮助 Transformer decoder 从一个好的特征起点开始学习，而不是从随机初始化的backbone特征上硬训。

### 6.3 其他训练配置

```python
optimizer = AdamW(lr=1e-4, weight_decay=0.05)
scheduler = LinearWarmup(1000 iter) + CosineAnnealing(50 epochs)
ema = ExpMomentumEMA(momentum=2e-4)
batch_size = 96 × 8 GPU = 768
load_from = "NanoDet rt11d2v2 best checkpoint"  # 预训练权重
```

---

## 七、评估指标体系

### 7.1 工程指标

| 指标 | 含义 | 目标 |
|------|------|------|
| 2D mAP | 各相机独立检测精度 | ≥ NanoDet baseline |
| 2D Precision/Recall | 单视角假阳/假阴 | ≤ baseline FP/FN |
| MV-mAP | 多视角联合检测精度（4视角2D bbox） | ≥ baseline + 5pp (goal) |
| Association Precision | 每个query匹配的4个view GT是否属于同一物体 | +20% relative (goal) |
| Association Recall | 每个GT物体是否有query覆盖其所有可见view | +20% relative (goal) |

### 7.2 业务指标

| 指标 | 定义 | 改善机制 |
|------|------|---------|
| Frictionless Rate | 无人工干预完成结账的session比例 | 更准的检测→更准的跟踪→更少结账错误 |
| NOF Rate | 商品无法匹配到商品数据库的比例 | 减少漏检 |
| Activity Signal Accuracy | add/remove事件准确率 | 更稳定的跨视角关联→更少tracker碎片化 |

**业务链路**：better detection → more reliable OmniTracker → fewer false add/remove events → higher frictionless rate

---

## 八、面试 Q&A

### A. 项目动机 & 设计选择

**Q1: 为什么要做多视角联合检测，当前方案有什么问题？**

> 当前方案是"单视角检测 + 后处理关联"的两阶段流程。有两个根本性限制：
>
> 第一，**遮挡无法在后处理阶段恢复**。如果一个商品在某个相机被完全遮挡，该相机产生漏检，后处理看不到这个检测框就无法关联，只能依赖其他相机的检测结果。但一旦某个相机的检测置信度低，threshold-based 关联就会失败，产生重复计数或漏计。
>
> 第二，**epipolar 约束对标定敏感**。生产环境中购物车物理变形、相机松动会导致实际相机位置偏离标定参数，硬编码的几何阈值失效，关联精度下降。
>
> MV-DETR 通过 attention 机制在 feature 层做融合，在做检测决策之前就已经聚合了4路相机的信息，从根本上解决了这两个问题。

---

**Q2: 为什么选 DETR 范式而不是改进现有的 NanoDet？**

> 核心原因是 NanoDet 是 anchor-based 的逐像素检测头，本质上是对单张图的 feature map 做密集预测，很难自然地扩展到多视角联合推理。
>
> DETR 的 object query 范式天然适合多视角：每个 query 代表一个物体实例，通过 cross-attention 可以同时 attend 多张图的 feature map，query 的表示空间可以自由地编码跨视角信息，输出也可以自然地扩展到 per-view bbox。
>
> 此外，DETR 的 Hungarian matching 保证 one-to-one 分配，和多视角场景中"一个物体对应一组跨视角检测"的语义完全对齐，不需要 NMS，不需要额外的关联逻辑。

---

**Q3: 为什么不做真3D检测（直接预测世界坐标），而是选择伪3D？**

> 真3D检测（如自动驾驶中的 DETR3D）需要：精确的相机内外参、点云或深度图、以及3D bbox标注（费时费力）。
>
> Instacart 购物车场景的业务需求是"哪个商品被放入/取出了"，不需要知道商品的世界坐标，只需要每个相机中的检测框用于后续的 OmniTracker 跟踪。伪3D（预测4组2D bbox + visibility）完全满足业务需求，且标注成本更低，对标定漂移更鲁棒。这是一个 pragmatic 的设计选择。

---

### B. 损失函数深挖

**Q4: loss_view 的设计意图是什么？去掉它会怎样？**

> `loss_view` 有两个作用：
>
> **显式作用**：让模型学会预测每个物体在每个相机中是否可见，这个可见性分数在推理时用于过滤无效的 bbox 输出（被遮挡的相机不应该输出 bbox）。
>
> **隐式作用**：在 Hungarian matching 中，view cost 是 cost matrix 的一个组成部分。它引导 query 优先匹配那些"对可见性预测正确"的 GT，让模型从训练初期就被迫学习"这个物体在哪些相机出现"的空间关系。
>
> 如果去掉它，模型仍然会被 visibility-weighted bbox loss 间接约束，但收敛会更慢，且推理时没有显式的可见性分数，需要用额外的阈值逻辑来过滤。

---

**Q5: visibility-weighted bbox loss 和直接 ignore 不可见 view 有什么区别？**

> 实现上 weight=0 和 ignore 效果相同——不可见 view 的 bbox 预测不产生梯度。
>
> 但这个设计的关键不在于如何实现，而在于对"不可见 view 的 bbox 预测该怎么处理"这个问题的明确回答：**我们不强迫模型在遮挡视角做出有意义的 bbox 预测，但也不完全不管它**——`loss_view` 会约束模型对不可见视角输出低置信度，从而在推理时自动过滤掉这些无效的 bbox 输出。两者配合形成完整的遮挡处理机制。

---

**Q6: 为什么对6层 decoder 都计算 loss？每层用同一套 Hungarian matching 结果吗？**

> **6层都算 loss**：Transformer decoder 收敛慢，只有最后层有 loss 时梯度需穿过5层，容易消失。每层都有 loss 相当于给每层一个"直通"的梯度通道，大幅加速收敛，也有正则化效果。
>
> **每层独立做 Hungarian matching**：不共享同一套结果。原因是不同层的预测质量差异很大——浅层 reference points 还很不准确，如果强行用深层的 assignment 结果来监督浅层，会引入错误梯度。每层独立匹配能让每层的 assignment 与该层的预测能力相匹配，训练更稳定。代价是每个 forward pass 要运行6次 `linear_sum_assignment`，但购物车场景 GT 数量小（M<20），这个开销可接受。

---

### C. Reference Points 深挖

**Q7: Reference points 的形状是什么，如何初始化，如何更新？**

> **形状**：`(B, num_query, num_views, 2)`。同一物体在4个相机中的投影位置不同，所以每个 query 为每个 view 维护独立的归一化 (cx, cy)。
>
> **初始化**：由可学习的 `query_pos` 经 `Linear(64→8).sigmoid()` 生成。初始时分布较均匀，训练中逐渐学习到有意义的初始位置分布（例如学会"物体倾向于出现在画面中央偏下"）。
>
> **迭代更新**（每层 decoder）：
> ```python
> new_ref = sigmoid(inverse_sigmoid(old_ref) + delta[:,:,:,:2])
> ```
> 用 inverse_sigmoid 做残差加法，保证结果始终在 [0,1]。6层后从粗糙猜测精化到物体真实位置。

---

**Q8: Cross-attention 中 reference points 具体怎么用？**

> Cross-attention 中用 `F.grid_sample` 在 reference point 位置从 feature map 上采样特征。流程：
>
> 1. 把 reference point 从 [0,1] 转换到 [-1,1]（grid_sample 的坐标格式）
> 2. 对4个 FPN level 的 feature map 各采1个点（num_points=1）
> 3. 用 attention_weights（由 query 预测的 4×1=4 个权重）加权求和
> 4. 加上 position_encoder(MLP(2→64)) 编码的坐标位置特征
> 5. 经 output_proj 输出
>
> 相比 dense attention 扫全图，每个 query 只采 4个点（4 level × 1 point），计算量极小。

---

### D. 工程与部署

**Q9: 训练时发现 LR 爆炸的问题，你们是怎么定位和解决的？**

> 问题复现：在8GPU上训练，epoch 2 时 loss 突然爆炸。
>
> 定位：检查 optimizer config 发现配置了 `auto_scale_lr(enable=True, base_batch_size=96)`，框架会根据实际 total batch size（8 GPU × 96 = 768）自动将 LR 从 1e-4 线性 scale 到 8e-4。对 CNN 这是常规操作，但 Transformer 对 LR 非常敏感，8e-4 已超出稳定范围。
>
> 解决：禁用 auto_scale_lr，固定 LR=1e-4；同时把 gradient clipping 从 value-based（clip_value=35）改为 norm-based（max_norm=0.1），这是 DETR 系列的标准配置；再把 backbone/neck 的 LR 降为 0.1×，避免破坏预训练特征。
>
> 经验教训：Transformer 训练的 LR 调节逻辑和 CNN 不同，不能直接套用 linear scaling rule，需要单独设定并仔细验证。

---

**Q10: TRT 部署有哪些挑战？**

> 主要有两个：
>
> **第一**，deformable attention 中的 `F.grid_sample` 算子在旧版 TRT 上支持有限，特别是动态 shape 场景。可能需要升级 TRT 版本，或者实现自定义 plugin。
>
> **第二**，`linear_sum_assignment`（Hungarian algorithm）是纯 Python scipy 调用，推理时不需要（推理时没有 matching，直接 top-k），但需要确保 export 时这段代码路径被正确剔除，只导出 `forward` 的预测部分。
>
> 缓解策略：先在 Phase 4 做 Post-training Quantization（PTQ），如果精度损失过大再考虑 Quantization-Aware Training（QAT）。冻结的 backbone/neck 量化风险低，主要风险在 Transformer decoder。

---

### E. 反思与改进

**Q11: 如果让你改进这个系统，你会从哪里入手？**

> 三个方向：
>
> **1. 注入相机几何先验**：当前 reference points 完全靠数据从随机初始化学起，需要大量数据。可以用已知的相机标定参数（即使不精确）初始化 reference points——把 view_0 的预测通过 epipolar 几何投影到其他 view，作为其他 view reference points 的初始值。这相当于把几何先验作为软约束，标定准确时加速收敛，标定漂移时仍靠 attention 修正，两全其美。
>
> **2. 解冻 backbone 阶段性 fine-tune**：Phase 3 验证成功后，以 0.1× LR 解冻 backbone，让特征表示向多视角任务优化。当前冻结是出于数据效率的考虑，但最终性能上限可能被冻结 backbone 限制。
>
> **3. 跨视角数据增强**：随机遮挡某个 view 的部分区域（模拟遮挡场景），强迫模型在缺失信息的情况下靠其他 view 做出正确预测，提高对遮挡的鲁棒性。

---

**Q12: 这个项目最大的技术风险是什么？**

> 最大的风险是**数据需求不确定**。Transformer 架构通常需要大量数据才能充分收敛，而多视角标注（需要跨视角的 object correspondence 和 per-view visibility 标签）的生产成本很高，当前 pipeline 吞吐量有限。
>
> 我们没有先验知识知道需要多少数据才能达到目标精度，因此采用了监控 scaling law 的策略：在 100K → 250K → 500K 数据检查点分别评估性能，观察性能增益是否收敛，以此判断数据是否足够以及是否需要调整架构。
>
> 如果纯数据驱动方式数据需求过大，备选方案是引入相机标定参数作为辅助输入，减少模型需要从数据中学习的几何关系，降低学习复杂度。

---

## 九、关键数字速查

| 参数 | 值 |
|------|---|
| num_query | 50 |
| num_views | 4 |
| embed_dims | 64 |
| num_decoder_layers | 6 |
| num_feature_levels | 4 |
| num_points per level | 1 |
| backbone | SimpleNet (scale=0.5, ~1.5M params) |
| neck | NanoDetFPN (out_channels=64) |
| optimizer | AdamW (lr=1e-4, wd=0.05) |
| gradient clip | norm, max_norm=0.1 |
| backbone lr mult | 0.1× |
| batch size | 96 × 8 GPU |
| max epochs | 50 |
| loss_cls weight | 2.0 (FocalLoss, γ=2, α=0.25) |
| loss_bbox weight | 2.0 (L1) |
| loss_iou weight | 2.0 (GIoU) |
| loss_view weight | 1.0 (BCE) |
| match cost cls | 2.0 |
| match cost L1 | 0.5 |
| match cost GIoU | 2.0 |
| detach_epoch | 10 |
| EMA momentum | 2e-4 |
| 延迟要求 | ≤ 40% 增幅 vs 当前2D检测 |
| association precision goal | +20% relative |
| association recall goal | +20% relative |
