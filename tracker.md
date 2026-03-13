# tracker CV — 面试准备手册

> 适用岗位：北美 L5 MLE（搜广推 / CV / 系统方向）
> 覆盖系统：OmniTrack（3D 追踪 + 事件检测）+ RII（实时商品识别）
> 所有答案均可用真实代码支撑

---

## 一、项目简介（3 分钟版本）

我在 Instacart Caper 负责智能购物车的计算机视觉系统，核心工作是两个紧密协作的系统：

**OmniTrack**：实时追踪用户手中的商品在购物车 3D 空间中的运动，检测商品的 ADD（放入）和 REMOVE（取出）事件。系统运行在 Jetson AGX/ORIN_NX 边缘设备上，使用 4 个购物车摄像头，通过多层 EKF（扩展卡尔曼滤波器）将 2D 检测融合成 3D 轨迹，再通过事件驱动的状态机判断用户意图。

**RII（Real-time Item Identification）**：在 REMOVE 事件触发后，用视觉检索判断被取出的是购物车里哪件商品。基于 CLIP ViT-B/16 视觉编码器，经 LoRA 微调后部署为 TensorRT 推理引擎。建索引时用 K-means（k=4）压缩每件商品的 embedding，查询时暴力 cosine similarity 搜索。

两个系统的接口：OmniTrack 发布 `(REMOVE_event, timestamp)` → RII 取时间窗口内的 query embedding → Redis 发布 `(barcode, confidence)` → 上游业务系统扣减购物车。

---

## 二、系统架构

### 2.1 OmniTrack 数据流

```
4 相机 YOLO 检测（2D bbox，每帧）
        |
  EKF Tracker 2D（per-camera，像素坐标）
  状态：[x, y, vx, vy, w, h]
        |
  EKF Tracker 9D（多相机融合，世界坐标）
  状态：[x, y, z, vx, vy, vz, w, h, d]
  初始化：立体三角化 or 单目深度估计
        |
  TrajectoryModelStateMachine
  z → zone (OUTSIDE / AT_RIM / IN_CART)
  zone 转换序列 + vz 方向 → ADD / REMOVE
        |
  REMOVE 事件 (timestamp T)
        |
  ItemObserver → RII re-identification
```

### 2.2 RII 数据流

```
离线（GCP）：
  catalog images + GPT-5 描述
  生产 session 数据（{barcode: [session_path]}）
        |
  CLIP LoRA 微调（image-text contrastive，r=16）
        |
  merge_and_unload → ONNX（opset 14）→ TensorRT engine
        |
  部署到 Jetson

实时（Jetson）：
  barcode 扫码 → 采集时间窗口 embedding → cv::kmeans(k=4)
                                            → VirtualCartItem.clusterCentroids
  REMOVE 事件 → 采集 query embedding
                → cosine sim vs 所有商品的 4 个 centroid
                → rank → Redis publish
```

### 2.3 关键数字速查

| 指标 | 值 |
|---|---|
| CLIP 输出维度 | 512-dim（ViT-B/16，768→512 visual_projection） |
| LoRA 参数 | r=16, alpha=32, dropout=0.1, LR=5e-4 |
| 可训练参数 | ~1.6M / 87M total = 1.8% |
| K-means | k=4，cv::kmeans，10 restarts，KMEANS_PP_CENTERS |
| 每件商品内存 | 4 × 512 × 4 bytes = 8KB |
| EKF 9D 状态维度 | 9D：[x,y,z,vx,vy,vz,w,h,d] |
| rim buffer | upperCm=4, lowerCm=-2，共 6cm |
| ADD/REMOVE 确认时间 | 50ms |
| Action Persistence | 1000ms，decay=0.7 |
| Velocity 阈值 | ADD: vz < -5 cm/s，REMOVE: vz > +5 cm/s |
| 评估规模 | N_SELECTED_ITEMS=50, N_RUNS=40, MAX_ITEMS=2000 |

---

## 三、OmniTrack 设计实现

### 3.1 EKF 三层架构

#### EKF 2D — per-camera 2D 追踪
- **状态**：`[x, y, vx, vy, w, h]`，像素坐标
- **H 矩阵**：线性（直接映射，不需要投影）
- **特点**：observation smoothing（指数移动平均，alpha 可配），velocity decay
- **用途**：per-camera tracklet，为 9D 提供 2D 观测

#### EKF 3D — 多相机 3D 融合（较早版本）
- **状态**：`[x, y, z, vx, vy, vz]`，世界坐标（cm）
- **Q 模型**：白噪声加速度（Singer Model），含 pos-vel 交叉项
  ```
  Q(pos,pos) = σ_a² · dt⁴/4
  Q(vel,vel) = σ_a² · dt²
  Q(pos,vel) = σ_a² · dt³/2   ← 交叉项
  ```
- **Update**：Stacked update（所有相机一次性），支持 pseudo-measurements（速度约束、限速约束）
- **两种模式**：STEREO_VISION（立体校准矩阵）/ GEO_UTILS（相机内外参+射线），运行时可切换
- **Jacobian**：数值前向差分，eps=1e-3cm，只扰动位置维度

#### EKF 9D — 当前主要生产版本
- **状态**：`[x, y, z, vx, vy, vz, w, h, d]`，9D（含 3D 尺寸）
- **新增能力**：测量包含完整 bbox `[cx, cy, w, h]`，每次更新 4 个约束而非 2 个
- **velocity decay**：`F[v,v] = exp(-β·dt)`，建模 stop-start 手部运动
- **Q 模型**：独立随机游走（对角），σ_dim 自适应（立体初始化 vs 单目初始化）
- **Update**：Sequential（逐相机），每次 4×4 LDLT，更数值稳定
- **Jacobian**：中心差分，eps 自适应（`eps = eps_base × max(1, |state_i|)`）
- **初始化**：
  - 立体（≥2 相机）：三角化中心 + 三角化四个角点估计尺寸 → 低协方差
  - 单目：搜索深度使矩形框对边 3D 长度差最小 → 高协方差，等待立体修正
- **自适应 dimension damping**：
  - 单目更新：强阻尼（mono_dim_damping）
  - 立体更新×stereo 初始化：中等阻尼
  - 立体更新×mono 初始化，收敛前：允许较大修正
  - 立体更新×mono 初始化，收敛后：锁定
- **边界约束**：cart boundary clamp（x/y/z 都有上下界）
- **Barcode-only 特殊处理**：R_dim=1e6，K.row(w/h/d)=0，2-DOF gating

### 3.2 数值稳定性机制

**Joseph form 协方差更新**（三个 EKF 均使用）：
```
标准形式：P = (I-KH)P          ← 浮点累积导致非对称/非正定
Joseph form：P = (I-KH)P(I-KH)ᵀ + KRKᵀ  ← 天然对称，半正定
```

**矩阵求解层次（EKF 3D 最完整）**：
```
Cholesky LLT → 失败 → 加 jitter(ε·I) → 失败 → SVD 伪逆 → 失败 → 跳过更新
```

**协方差正则化（EKF 3D）**：
```
1. 对称化：P = 0.5(P + Pᵀ)
2. LLT 检查：失败 → 特征值分解 → clamp 负特征值 → 重构
```

### 3.3 轨迹状态机

#### 三 zone 空间划分
```
z 轴（购物车世界坐标，向上为正）：

z > 4cm        → OUTSIDE   （物品在购物车外）
-2cm ≤ z ≤ 4cm → AT_RIM    （6cm 缓冲区，吸收 EKF z 轴噪声）
z < -2cm       → IN_CART   （物品在购物车内）
```

#### ADD/REMOVE 检测（各两种 Pattern）

| Pattern | 触发条件 | 置信度 | 适用场景 |
|---|---|---|---|
| ADD Full | OUTSIDE→AT_RIM→IN_CART，vz<0，IN_CART≥50ms | 全置信 | 小物品（牛奶、薯片） |
| ADD Partial | OUTSIDE→AT_RIM，vz<-5cm/s，AT_RIM≥50ms | ×0.9 | 大物品（整箱水，中心不进 IN_CART） |
| REMOVE Full | IN_CART→AT_RIM→OUTSIDE，vz>0，OUTSIDE≥50ms | 全置信 | 正常取出 |
| REMOVE Partial | IN_CART→AT_RIM，vz>+5cm/s，AT_RIM≥50ms | ×0.9 | 只抬到车沿 |

#### 其他状态

| 状态 | 检测条件 |
|---|---|
| STATIONARY | IN_CART + 200ms 无显著移动 + 低速度 |
| STRUGGLING | AT_RIM > 5s 且有运动，或 rimCrossingCount > 3 |
| SHUFFLE | 30s 内 rim 穿越 ≥ 4 次 |
| UNKNOWN | 无 pattern 匹配，返回 0.85 概率 |

优先级：ADD > REMOVE > STATIONARY > STRUGGLING > SHUFFLE > UNKNOWN

#### 置信度计算
```
confidence = raw × zone_weight × transition_boost

raw = 0.6
    + 0.1 × min(1, |v| / 10)           # 速度越大越确定
    + 0.1 × min(1, timeInState / 500ms) # 停留越久越确定

zone_weight: IN_CART=1.2, AT_RIM=1.0, OUTSIDE=0.6

transition_boost: 有 rim 穿越记录 → 1.3×，否则 1.0×

cap: min(0.95, final)
minConfidenceForAction = 0.28（发布阈值）
```

#### 防抖机制

**Decision Buffering**（事件确认前）：
- 要求连续 2 帧（minConfirmationFrames=2）检测到相同 action 才发布
- 防止单帧误触发，代价：约 60ms 延迟

**Action Persistence**（事件确认后）：
```
确认后记录：lastConfirmedAction, timestamp, confidence
后续帧：decayedConf = lastConf × 0.7 × (1 - age / 1000ms)
  > actionHoldMinConfidence(0.20) → 继续返回该 action
  否则 → 重置为 UNKNOWN
```
防止 REMOVE→UNKNOWN→REMOVE 振荡，给下游 1s 稳定消费窗口。

---

## 四、RII 设计实现

### 4.1 训练

**LoRA 配置**：
```python
LoraConfig(
    r=16, lora_alpha=32,      # effective_scale = alpha/r = 2.0
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.1,
    modules_to_save=["visual_projection"]  # full retrain，不用 LoRA
)
# 应用于 vision_model，text encoder 冻结作为 anchor
# 1.2M（adapter）+ 0.4M（visual_projection）= 1.6M 可训练参数
```

**为什么 LoRA 而非全量**：
- 购物车数据量有限，全量微调过拟合风险高
- LoRA 可用更大 LR（5e-4 vs 全量 1e-5），B 矩阵初始化为零，大 LR 不破坏预训练表征
- adapter 文件 ~10MB vs 全量 checkpoint 数百 MB，Jetson 部署友好
- `merge_and_unload()` 在推理时零 latency 开销

**visual_projection 为何用 modules_to_save 而非 LoRA**：
768→512 的线性层直接决定 embedding 几何空间。LoRA 假设原始方向对，只做低秩扰动。这里需要完全重塑 embedding 空间以适配购物车商品检索，需要全参数自由度。

**文本描述的角色**：
- 训练时：GPT-5 生成描述 → 冻结 text encoder → text embedding 作为 supervision anchor
- 推理时：完全不用 text encoder，只导出 visual encoder

### 4.2 索引建立（barcode 触发）

```
barcode 扫描（时间 T）
    → getClassificationsInWindow(T±window)
    → filter: confidence < threshold → 丢弃
    → VirtualCartItem.extendWithEmbeddings()

basket sync（用户确认商品）
    → performClustering(k=4)
      - len(embeddings) < 4 → 每个 embedding 直接作为 centroid
      - 正常：cv::kmeans(data, k=4, KMEANS_PP_CENTERS, 10 restarts)
    → clusterCentroids = [c0, c1, c2, c3]
    → embeddings.clear() + shrink_to_fit()  # 释放原始 embedding，只保留 8KB/item
```

**k=4 的直觉**：对应包装商品的 4 个主视角（正面/背面/侧面/顶部），不是相机数量。

### 4.3 REMOVE Re-identification

```
REMOVE 事件（时间 T）
    → 采集 query embeddings（时间窗口内）
    → 质量过滤：
        - len(qualityEmbeddings) < minRequired → 跳过
        - z-axis movement < zMovementThresholdMm → 跳过（未真正移动）
        - dominantType == PRODUCE → 跳过（生鲜无条形码）
    → 暴力搜索（brute-force）：
        for each queryEmb:
            for each basketItem (isInCurrentBasket==true):
                bestSim = max(cosine_sim(queryEmb, centroid)
                              for centroid in item.clusterCentroids)
                if bestSim >= minSimilarityThreshold:
                    itemSimilarities[barcode].append(bestSim)
    → 聚合：avgSim = mean(itemSimilarities[barcode])
    → rank → top-1 → markItemAsRemoved()
    → if avgSim > reIDConfidenceThreshold:
        publish(REMOVED_ITEM_CHANNEL, {barcode, confidence, trace_id})
```

**为什么暴力搜索而非 FAISS**：购物车 < 20 件商品，每件 4 个 centroid = 最多 80 个向量。暴力搜索延迟可忽略，无需引入外部 ANN 库依赖。

### 4.4 模型部署

```python
# 导出流程
base_model.vision_model = PeftModel.from_pretrained(base_model.vision_model, adapter_path)
base_model.vision_model = base_model.vision_model.merge_and_unload()
image_encoder = model.visual  # 只导出 visual encoder
torch.onnx.export(
    image_encoder,
    dummy_input,                          # (1, 3, 224, 224)
    output_path,
    opset_version=14,
    dynamic_axes={"image": {0: "batch_size"}},
)
# Jetson 上：trtexec --onnx=model.onnx --saveEngine=model.engine
```

---

## 五、评估与验证

### 5.1 离线评估协议

```
organize_data_by_sessions()
  → get_first_session_per_item()   # index set（label=1）
  → last session                   # query set（label=0）

build_item_indices(KMeans k=4)     # 离线模拟线上建索引
compute_similarities(query, cluster_centers)
  → cosine_similarity([query], centers)[0]
  → max over 4 centroids            # item-level score per frame
  → mean over all query frames      # session-level aggregation

Metrics: Top-1 / Top-5 / Top-10 / MRR
```

### 5.2 大规模生产数据评估

```python
# eval_threshold_topk.py
# shape: (N_frames, total_centers) where total_centers = N_items × 4
all_similarities = cosine_similarity(retrieval_features, all_cluster_centers)

# Per-item max
item_similarities[:, item_idx] = np.max(item_center_sims, axis=1)

# Item-level aggregation
aggregated = np.mean(item_similarities, axis=0)

# Threshold 优化
# TP = in top-K AND above threshold
# FP = not in top-K AND above threshold
# 目标：precision-first（false positive 代价更高）

# 规模：N_SELECTED_ITEMS=50, N_RUNS=40, MAX_ITEMS=2000
```

### 5.3 Basket Size Ablation

```python
# simulate_multi_basket_size_evaluation.py
two_camera_mode = True  # 只保留 camera 0 和 3
basket_size sweep: 1..N items，100 trials per size
  → Top-1 accuracy ± CI（2.5th / 97.5th percentile）

# 对齐线上真实分布（通常 5-15 件商品）
```

### 5.4 Threshold 调优

```
Precision-recall-coverage 曲线，per (Top-K, threshold) pair

模型版本感知：
  v1 → reid_confidence_threshold_v1, min_reid_candidate_similarity_v1
  v2 → reid_confidence_threshold_v2, min_reid_candidate_similarity_v2
  （从 model filename 后缀自动检测，C++ 代码）
```

### 5.5 为什么 Precision-First

RII 结果直接更新购物车扣减逻辑：
- **False Positive 代价**：把 A 取出，系统认为是 B → 扣减错误商品 → 结账错误，用户可感知，严重
- **False Negative 代价**：取出但未识别 → 购物车状态未更新 → 可通过其他方式兜底，代价相对小

**宁可不报，不能报错**。

---

## 六、关键设计决策速查

| 决策 | 理由 |
|---|---|
| OmniTrack 事件驱动 vs 帧驱动 | 帧驱动导致 90% UNKNOWN，状态转换触发的事件信号更稳定 |
| 3-zone vs 6-zone | 6-zone 维护复杂，AT_RIM 缓冲区已能处理测量噪声，简化后调参直觉 |
| EKF 9D vs 3D | 9D 将 bbox 尺寸纳入状态，测量方程更完整，三角化精度更高 |
| velocity decay in 9D | 手部 stop-start 运动不满足 constant velocity，decay 防止鬼影漂移 |
| sequential vs stacked update（9D） | 9D 每相机 4D 测量，stacked 矩阵大；sequential 固定 4×4 更稳定 |
| Joseph form 协方差更新 | 标准形式浮点累积破坏正定性；Joseph form 天然对称正定 |
| LoRA r=16, alpha=32 | r=16 中间值（r=4 欠拟合，r=64 接近全量）；alpha/r=2.0 加速 adapter 收敛 |
| visual_projection full retrain | 需要完全重塑 embedding 空间，LoRA 低秩约束太强 |
| 只导出 visual encoder | text encoder 边缘设备成本高；推理全程 image-to-image |
| k=4 K-means centroids | 对应商品 4 主视角；k=4 × 512 × 4B = 8KB/item，Jetson 内存可控 |
| 清除原始 embedding | 8KB vs 100KB+，Jetson 内存约束；debug mode 保留原始 embedding |
| 暴力 cosine 搜索 | < 20 items × 4 centroids = 80 vectors，无需 FAISS 复杂度 |
| Precision-first threshold | False positive 直接导致结账错误，代价高于漏报 |

---

## 七、Q&A — OmniTrack

### 系统设计

**Q: 为什么需要 3D tracking？用 2D detection 不够吗？**

ADD/REMOVE 检测依赖 z 轴（高度）信息。购物车 rim 是固定高度，物品是否越过 rim 无法从单相机 2D 坐标判断——2D 的 y 轴混合了深度和高度。另外 4 个相机各自独立的 2D tracklet 需要在 3D 世界坐标合并，才能获得统一的位置和速度供 state machine 使用。EKF 通过三角化将多相机 2D 观测融合成 `[x,y,z,vx,vy,vz]`，z 坐标直接对应 rim 高度判断，vz 方向确认 ADD/REMOVE 意图。

---

**Q: 3-zone 的 AT_RIM 缓冲区为什么是 6cm 宽？**

EKF z 轴不确定性来源：三角化误差（~1cm）+ 相机外参校准误差（~1-2cm）+ 检测器 bbox 误差（~1cm）。3σ 约 3-4cm。6cm 宽的缓冲区覆盖这个不确定性，让物品在静止时稳定在单个 zone 内，不会因为测量抖动在边界反复穿越产生假转换。

诊断依据：如果 `rimCrossingCount` 在正常操作中异常升高，说明 z 轴噪声变大（可能是相机重新校准），需要调大 rim buffer 或重新标定 EKF 的 R 矩阵。所有配置通过 proto config 热更新，不需要重新部署。

---

**Q: 为什么 ADD/REMOVE 各有两种 Pattern（Full/Partial）？**

Full Pattern 要求物品中心点完整穿越三个 zone，对小物品（牛奶、薯片）完全适用。但**大物品**（整箱水、大西瓜）放入购物车时，相机只能看到物品上半部分，中心点从 OUTSIDE 降到 AT_RIM 就停止，永远不会进入 IN_CART。

Partial Pattern 针对这种情况：只要在 AT_RIM 且 vz < -5 cm/s（强向下速度），就判定 ADD。速度阈值是关键 guard，防止自然下沉或抖动误触发。置信度乘以 0.9，略低于 Full Pattern，体现了信息不完整的代价。

---

**Q: Action Persistence 和 Decision Buffering 都需要吗？各自解决什么问题？**

两者解决不同阶段的问题：

Decision Buffering（确认前）：要求连续 2 帧检测到相同 action，防止单帧检测器噪声触发事件。代价是约 60ms 延迟。

Action Persistence（确认后）：REMOVE 确认后，在 1s 内以指数衰减置信度维持该状态（`decayedConf = lastConf × 0.7 × (1 - age/1000ms)`）。防止 EKF 追踪短暂丢帧导致 REMOVE→UNKNOWN→REMOVE 振荡，给下游 RII 稳定的时间窗口消费事件。

两者组合：buffering 控制"触发门槛"，persistence 控制"持续时间"，缺一不可。

---

### EKF 设计

**Q: EKF 3D 和 EKF 9D 的核心区别？为什么升级到 9D？**

EKF 3D 把物体当点，测量只用 bbox 中心 (cx,cy)，丢弃了 bbox 尺寸信息。

EKF 9D 加入物体 3D 尺寸 `[w,h,d]` 进入状态向量，测量方程变成完整 bbox `[cx,cy,w,h]`。每次更新有 4 个约束而非 2 个，三角化时还可以通过四个角点估计尺寸，深度精度更高。

升级 9D 的根本原因：`projectBoxTo2D(center, dims, cameraId)` 需要 center 和 dims 才能预测 bbox，如果不把 dims 纳入状态，每次 Jacobian 计算都需要额外假设，导致模型不一致性。

---

**Q: EKF 9D 为什么用 velocity decay，3D 用白噪声加速度模型？哪个更好？**

EKF 3D 的 Singer Model（白噪声加速度）是物理上更严谨的推导，Q 矩阵有 pos-vel 交叉项（dt³/2 项），反映位置和速度误差的相关性。

EKF 9D 选 velocity decay（`F[v,v] = exp(-β·dt)`）的原因：购物车场景里手的运动是 stop-start 的，constant velocity 假设在停顿时会继续预测物品漂移（"鬼影"）。Decay 让速度在没有观测时自然衰减到零，更符合真实手部运动特征。

两种方法在精度上相近，decay 更简单直觉，Singer Model 更物理严谨。9D 选 decay 也是因为同时引入了尺寸状态，Q 矩阵已经够复杂，不再需要 pos-vel 交叉项增加额外复杂度。

---

**Q: Joseph form 协方差更新有什么好处？代价是什么？**

标准形式 `P = (I-KH)P` 理论正确，但浮点运算累积后 P 会变非对称甚至非正定，导致 Cholesky 分解失败。

Joseph form：`P = (I-KH)P(I-KH)ᵀ + KRKᵀ`。分析：`A·P·Aᵀ` 天然对称（对任意 A），`KRKᵀ` 半正定（R 正定）。两项之和保证 P 对称且正定。

代价：多一次矩阵转置乘法，对于 9×9 矩阵计算开销可忽略。这是所有三个 EKF 都选 Joseph form 的原因。

---

**Q: 单目初始化怎么估计深度？假设是什么？**

扫描深度范围（每 1cm 一步），对每个候选深度 d，将 bbox 四个角点沿射线投影到 3D 空间，计算矩形框对边的 3D 长度差：
```
score = |width_top - width_bottom| / avg_width
      + |height_left - height_right| / avg_height
```
选 score 最小的深度（对边最接近相等）。

假设：被追踪的物体是矩形的（包装商品），矩形对边 3D 长度相等。当且仅当深度猜对时，投影到 3D 的 bbox 才满足这个约束。

局限：圆形/不规则物品（生鲜）会失效；正方体物品在任意深度 score 都很低，结果不稳定。因此单目初始化的 P_dim 和 P_pos_y 都设高（等待立体更新纠正）。

---

**Q: Barcode-only 观测为什么要特殊处理？**

barcode scanner 检测到的是条形码本身的 bbox，中心点可靠，但 bbox 尺寸是条形码大小，不是商品大小。直接用会把商品 3D 尺寸更新成错误值，影响后续所有帧的投影预测。

双重保险处理：
1. `R_dim = 1e6`：通过 Kalman gain 推导路径，让尺寸相关的 gain 趋近于零
2. `K.row(6/7/8) = 0`：显式强制，防止数值误差泄漏
3. Gating 用 2-DOF（只 gate 中心点），不用 bbox 尺寸判断是否属于当前 track

---

### 反思与改进

**Q: 如果重新设计 OmniTrack，会改哪三点？**

**1. 概率状态机（理论）**：当前 3-zone 是硬边界。EKF P 矩阵的 (2,2) 元素给出了 z 轴的不确定性 σ_z，可以用软 zone 概率替代硬边界：`P(OUTSIDE) = Φ((z - rimUpperCm) / σ_z)`。避免 z=4.0cm 和 z=3.9cm 产生截然不同的行为。

**2. EKF 参数在线自适应（工程）**：用 Innovation Sequence Test 监控 EKF 健康度。当 `E[yᵀS⁻¹y]` 显著偏离测量维数时，说明 Q 或 R 低估了实际噪声，自动调整。处理不同门店、不同光照条件下噪声不同的问题，而不是靠人工调参。

**3. 精化 REMOVE 时间戳（业务影响最大）**：当前 state machine 报告的 REMOVE 时间戳 = 进入 OUTSIDE 的时刻，而真正的离开时刻是 `AT_RIM→OUTSIDE` 的穿越瞬间（`StateTransitionEvent.timestamp`）。两者差几十毫秒。精化时间戳能缩小 RII 的 query window，减少手遮挡帧混入 query embedding，直接提升 re-identification Top-1 准确率。

---

## 八、Q&A — RII

### 训练设计

**Q: 为什么用 retrieval 而不是 classification？**

两个核心原因：

第一，商品空间是**开放集合**。超市有几万 SKU，每天有新品上架。Classification 每次加新商品都需要重新训练部署，在边缘设备上不可行。Retrieval 只需要在扫码时动态建索引，zero-shot 支持新商品。

第二，购物车上下文天然是 retrieval。REMOVE 时已知候选集（当前购物车的商品，通常 < 20 件），是 closed-set retrieval，不需要大规模 ANN 索引，候选空间极小。

---

**Q: LoRA r=16 怎么选的？alpha=32 的作用是什么？**

r 控制低秩矩阵的秩，即可训练参数量。r=4 表达能力不足；r=64 接近全量微调。r=16 是 NLP 任务广泛验证的中间值，在评估集上对比了 r=8/16/32，r=16 精度和参数量平衡最好。

alpha=32 的作用：LoRA 更新公式是 `W_new = W_base + (alpha/r) × B × A`，alpha/r = 2.0 相当于给 adapter 输出做 2 倍放大，等价于把 adapter 的有效 LR 乘以 2。B 矩阵初始化为全零，需要较大有效 LR 从零快速收敛。这也是 LR=5e-4 比全量微调高 50× 的原因：base model 冻结，大 LR 只影响 adapter，不破坏预训练表征。

---

**Q: 为什么 target_modules 包含 out_proj？很多实现只加 q_proj 和 v_proj。**

`q_proj/v_proj` 控制 attention pattern（关注哪里、如何加权），`out_proj` 是 multi-head attention 的输出投影，控制各个 head 的结果如何聚合回 hidden space。

购物车图像和 CLIP 预训练数据（网络图像）的视觉统计特性差异很大（低光、灰度、固定视角）。只适配 attention pattern 而不适配输出聚合，会有信息瓶颈：attention 分布变了但 hidden space 的表达方式没变。加入 `out_proj` 让模型能重新组合各 head 的特征，实验上 Top-1 有明显提升。

---

**Q: 训练用了 GPT-5 文字描述，推理时为什么完全不用 text encoder？**

文字描述在训练时作为 supervision anchor：冻结 text encoder → text embedding → InfoNCE loss 拉近商品图像 embedding 与对应描述 embedding。文字提供了比纯图像对比学习更丰富的语义信号（"Horizon 品牌，有机全脂牛奶，1 加仑，白色标签"）。

推理时：只导出 `model.visual`，text encoder 不打包进 TRT engine。Jetson 边缘设备上 text encoder（~63M 参数）运行成本高；更重要的是，REMOVE 时没有文字信息，本来就是 image-to-image 的问题。文字只是训练时的"老师"，训练完后就不再需要。

---

**Q: K-means k=4 为什么选这个值？**

直觉依据：一件包装商品的主要视觉外观变体通常对应 4 个方向（正面/背面/侧面/顶部），k=4 对应这个先验。

验证：在评估集上跑 k=2/4/8/16 的 Top-1 accuracy，k=4 是边际收益开始下降的拐点。同时考虑内存约束：k=4 是 8KB/item，k=16 是 32KB/item，Jetson 内存有限。cv::kmeans 输出 compactness，k=4 时已经较低，增加 k 提升不大。

理论上 spherical K-means（在单位球面上聚类，适合 L2-normalized embedding）比标准 K-means 更准确，但实践差异很小，标准 K-means 使用 OpenCV 内置实现，维护成本低。

---

### 工程部署

**Q: SharedTrtModel 共享一个 TRT engine 给 item recognition 和 location recognition，有什么风险？**

**好处**：两个任务都用 CLIP visual encoder，共享 engine 节省 Jetson GPU 显存（有限资源），减少引擎初始化时间。

**风险**：
1. 资源竞争：两个任务并发请求时，共享 engine 的 batch queue 可能产生排队延迟
2. 耦合性：item recognition 需要升级模型时，location recognition 也要跟着重新部署
3. 输出解析耦合：两个任务的 postprocessFunc 通过 sessionId 区分，出 bug 时难以隔离

**当前可接受**：两个任务在时序上基本不重叠。未来如需解耦，分成两个 engine 是正确方向，代价是多占一份显存。

---

**Q: 生产模式下聚类后删除原始 embeddings，如果需要重新聚类怎么办？**

这是一个不可逆的工程 tradeoff。删除后改 k 值无法对已有商品重新聚类，只能对新扫入的商品用新 k 值。

实际影响有限：k 值来自 proto config，一次购物 session（20-30 分钟）内不会改变；k 值属于模型配置，更新频率极低。

如果要支持动态 k：`debugModeEnabled=true` 时保留原始 embeddings，可以随时重新聚类，代价是每件商品内存从 8KB 增加到可能 100KB+。这是 debug 和 production 两种模式的根本差异。

---

### 评估与反思

**Q: Offline 评估和线上场景有什么 gap？怎么缓解？**

主要三个 gap：

1. **数据分布**：lab 数据光照/角度相对固定，线上用户随机拿放，遮挡更多、图像质量更差
2. **时间差**：lab 的 index/query session 可能同天采集；线上 index 是几秒前扫码建的，商品可能被移动过
3. **Basket size**：lab 控制了 basket size，线上是真实分布（通常 5-15 件）

缓解措施：
- `simulate_multi_basket_size_evaluation.py` 模拟不同 basket size，给出带置信区间的 accuracy vs size 曲线，对齐真实分布
- 大规模生产数据评估（MAX_ITEMS=2000）使用线上采集数据，比 lab 更接近真实
- Frame-level 和 item-level 两个粒度评估，item-level mean 聚合更接近线上逻辑

---

**Q: 如果重新设计 RII，会改什么？**

**1. Spherical K-means（理论）**：CLIP embedding 是 L2-normalized 单位向量，存在于高维球面。标准 K-means 用 Euclidean 均值，centroid 不在球面上，理论偏差。Spherical K-means 在每次更新 centroid 后做 L2 normalize，更适合 cosine similarity 检索。实现成本低，预期小幅精度提升。

**2. 在线增量更新 centroid（工程）**：当前 barcode 扫码时一次性聚类，之后不再更新。但用户拿取过程中持续获得新 embedding（不同角度、不同光照）。可以用 exponential moving average 持续更新 centroid，让索引随时间更完整。

**3. 软标签 vs 硬 K-means（理论）**：K-means 是硬分配，每个 embedding 只归属一个 cluster。圆柱形商品（瓶装饮料）从任意角度看变化连续，GMM 或 soft assignment 能更好建模这种连续性。代价是推理时不再是 max cosine，需要 density estimation，计算量增加。

---

## 九、系统与搜广推的关联（必备）

| OmniTrack / RII | 搜广推对应概念 |
|---|---|
| CLIP ViT-B/16 visual retrieval | 双塔模型（同构双塔，query/item 同一 visual encoder） |
| 动态建索引（每次购物实时建） | 实时召回补充路径（解决新商品冷启动） |
| K-means centroid 作为 item 表示 | Item embedding 离线建索引 |
| Precision-first threshold 调优 | 广告召回宁可少召不召低质（False Positive 代价高） |
| Basket size ablation（recall set size 影响） | Recall set size 对 ranking 精度的影响分析 |
| LoRA 域适配（冻结 text encoder 作为 anchor） | 预训练双塔模型业务微调（冻结一个 tower） |
| EKF 边缘侧部署约束反向影响模型设计 | 移动端推理约束（量化、蒸馏、延迟预算）反向影响模型选择 |
| OmniTrack 事件驱动 → RII 触发 | 用户行为事件驱动召回（实时行为序列触发个性化） |

---

## 十、答题策略

| 问题类型 | 套路 |
|---|---|
| 系统介绍（3 min） | 问题定义 → 整体架构 → 两个关键设计选择 → 结果 |
| 设计决策（为什么 X） | 约束（场景限制）→ X 如何满足 → 代价 → 如何缓解代价 |
| 数字选择（为什么 k=4、r=16） | 先说直觉/先验 → 说明验证方式 → 说明可配置 |
| 数学推导（Joseph form、Singer Q） | 先说结论和直觉 → 给代码行号 → 再展开数学 |
| 边界情况 | 场景描述 → 代码中具体 guard → 残留风险 → 监控方式 |
| 改进类 | 理论严谨性 + 工程可落地 + 业务影响排序，punchline 选业务影响最大的 |
| Offline/Online gap | 具体 gap 列举 → 现有缓解措施 → 残留 gap + 未来方向 |
