# 搜索推荐 Ranking 岗位基础知识

> 适用：L5 MLE，Ranking 方向
> 覆盖：召回→粗排→精排→重排全链路，重点在精排

---

## 一、Ranking 在推荐系统中的位置

### 1.1 整体漏斗架构

```
用户请求
   │
   ▼ 召回 (Retrieval)
候选集: 百万 → 数千
   │  ANN向量检索 / 倒排索引 / 规则过滤
   │
   ▼ 粗排 (Pre-ranking)
候选集: 数千 → 数百
   │  轻量模型，延迟敏感，向量内积打分
   │
   ▼ 精排 (Ranking)          ← 核心岗位职责
候选集: 数百 → 数十
   │  重模型，特征工程丰富，复杂交叉
   │
   ▼ 重排 (Re-ranking)
最终列表: 数十
   │  多样性、打压重复、业务规则、position bias修正
   │
   ▼ 展示给用户
```

### 1.2 各阶段核心关注点

| 阶段 | 核心目标 | 延迟要求 | 模型复杂度 |
|------|---------|---------|-----------|
| 召回 | 高召回率，不漏好item | <50ms | 极简（双塔，向量） |
| 粗排 | 粗粒度过滤，保精排质量 | <30ms | 轻量（浅层MLP） |
| 精排 | 准确预估用户行为概率 | <100ms | 复杂（DNN+特征交叉） |
| 重排 | 列表整体最优 | <50ms | 规则+MMR/DPP |

---

## 二、精排模型演进

### 2.1 经典模型时间线

```
LR (2010)
  └── 手工特征+特征交叉，可解释性强，大规模稀疏特征

GBDT + LR (2014, Facebook)
  └── GBDT做特征变换，LR做最终预测
  └── 解决非线性，但离线特征，不支持实时更新

FM / FFM (2010/2016)
  └── 自动学习二阶特征交叉，参数共享
  └── FM: O(kd)，FFM: O(kd²/2)

Wide & Deep (2016, Google)
  └── Wide: 记忆（LR+手工交叉）
  └── Deep: 泛化（DNN）
  └── 联合训练

DeepFM (2017)
  └── 用FM替换Wide部分，无需手工特征工程
  └── FM + DNN 共享embedding

DIN (2018, Alibaba)
  └── 用target item对用户历史行为做attention
  └── 解决用户兴趣多样性问题

DIEN (2019, Alibaba)
  └── DIN + GRU对行为序列建模时序关系
  └── Interest Extractor + Interest Evolving

BST (2019, Alibaba)
  └── 用Transformer对行为序列建模

SIM (2020, Alibaba)
  └── 超长序列（百万）：先检索相关子序列，再attention

DCN v2 (2021, Google)
  └── 显式高阶特征交叉，Cross Network²

DSSM/双塔 (召回向)
  └── 用户塔+物品塔，内积召回
```

### 2.2 DeepFM 架构（精排标准基线）

```
输入: [user_id, item_id, user_age, item_cate, ...]
  ↓ Embedding层
  [e1, e2, e3, e4, ...]  每个field → d维向量

FM部分:                    DNN部分:
  一阶: Σ wi·xi             concat所有embedding
  二阶: Σ<vi,vj>xi·xj       → FC → FC → FC
       ↑共享embedding        ↑共享embedding

concat → sigmoid → CTR预测
```

**FM二阶交叉公式**（O(kd) 实现）：
```
Σ_{i<j} <vi,vj> xi·xj = 1/2 * [||Σ vi·xi||² - Σ ||vi||²·xi²]
```

### 2.3 DIN 核心机制

```python
# 用 target item 对历史行为序列做 attention
def din_attention(target_item_emb, behavior_seq_emb):
    # target_item_emb: (B, d)
    # behavior_seq_emb: (B, T, d)

    # 扩展target维度与序列对齐
    target_expanded = target_item_emb.unsqueeze(1).expand_as(behavior_seq_emb)

    # 拼接：[behavior, target, behavior-target, behavior*target]
    concat = torch.cat([
        behavior_seq_emb,
        target_expanded,
        behavior_seq_emb - target_expanded,
        behavior_seq_emb * target_expanded
    ], dim=-1)  # (B, T, 4d)

    # 注意力分数
    attn_score = MLP(concat).squeeze(-1)  # (B, T)
    attn_score = softmax(attn_score)

    # 加权求和
    user_interest = (attn_score.unsqueeze(-1) * behavior_seq_emb).sum(dim=1)
    return user_interest
```

**设计动机**：用户对一件运动鞋的兴趣，应该主要由历史中点击过的运动相关item决定，而不是所有历史行为的平均。

---

## 三、特征工程

### 3.1 特征类型体系

```
特征
├── 用户特征 (User Features)
│   ├── 静态: age, gender, city, device
│   ├── 统计: 7d/30d CTR, 购买频率, 品类偏好
│   └── 序列: 行为历史 (点击/购买/收藏序列)
│
├── 物品特征 (Item Features)
│   ├── 基础: category, brand, price, seller
│   ├── 统计: 全局CTR, 曝光量, 销量, 评分
│   └── 内容: title embedding, image embedding
│
├── 上下文特征 (Context Features)
│   ├── 时间: 小时/星期/是否节假日
│   ├── 场景: 搜索词, 推荐位, 页面类型
│   └── 设备: 网络类型, app版本
│
└── 交叉特征 (Cross Features)
    ├── user×item: 用户品类偏好 × item品类 (个性化)
    ├── user×context: 用户时段行为偏好
    └── item×context: item在该时段的历史表现
```

### 3.2 特征处理

**离散特征（ID类）**：
```python
# Embedding lookup，处理高基数稀疏特征
user_emb = Embedding(num_users, 64)[user_id]
item_emb = Embedding(num_items, 64)[item_id]
# 低频截断：出现次数<threshold的ID映射到同一个OOV向量
```

**连续特征**：
```python
# 1. 标准化（正态分布假设）
x = (x - mean) / std

# 2. 分桶（处理长尾/异常值）
x = pd.qcut(x, q=10, labels=False)  # 等频分桶

# 3. log变换（幂律分布）
x = log(x + 1)
```

**序列特征**：
- 固定长度截断（取最近N条）或 padding
- 时间衰减加权：`w_t = exp(-λ·Δt)`
- 超长序列：SIM 先检索，再 attention

### 3.3 实时特征 vs 离线特征

| 特征类型 | 示例 | 更新频率 | 存储 |
|---------|------|---------|------|
| 离线统计 | 用户30天CTR | 天级T+1 | HDFS→特征平台 |
| 近实时 | 用户1小时行为 | 分钟级 | Kafka→Redis |
| 实时 | 当前session行为 | 秒级 | 内存/本地缓存 |
| 在线实时 | 当前请求上下文 | 请求级 | 直接传入 |

**实时特征的挑战**：训练时用离线特征，推理时用实时特征，存在 training-serving skew。

---

## 四、损失函数

### 4.1 Pointwise（逐点）

```python
# 二分类 BCE，最常用
loss = -[y·log(p) + (1-y)·log(1-p)]

# 优点：简单，直接预测概率，可用于多任务
# 缺点：不显式优化排序关系，需要大量负样本
```

### 4.2 Pairwise（逐对）

```python
# BPR Loss：正样本得分 > 负样本得分
loss = -log(sigmoid(score_pos - score_neg))

# 优点：直接优化相对顺序
# 缺点：O(n²)对数量，训练慢
```

### 4.3 Listwise（列表级）

```python
# ListNet：预测分布与GT分布的KL散度
# LambdaRank：直接优化NDCG
# 工业界较少用，复杂度高
```

### 4.4 工业界实践

- **CTR 任务**：BCE（Pointwise），正负样本比例 1:N（N=10~100）
- **多目标任务**：多个 BCE 加权求和
- **序列排序**：Pairwise BPR 或 Softmax（分母为全部候选）

---

## 五、负样本采样

### 5.1 为什么负样本采样很重要

正样本：用户点击/购买的item（天然标注）
负样本：用户未点击的item（不等于"不感兴趣"，可能是未曝光）

**不加处理的全量负样本会引入大量噪声**。

### 5.2 常见策略

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| 曝光未点击 | 曝光但未点击视为负样本 | 高质量，已排除未见item | 有position bias |
| 随机采样 | 从全量item池随机采 | 简单，覆盖范围广 | 低曝光item被过采样 |
| 热度采样 | 按item曝光量采样 | 缓解长尾 | 热门item被过度压制 |
| Hard negative mining | 召回模型高分但未点击 | 训练精排模型更有效 | 计算成本高 |
| In-batch负样本 | 同batch其他用户的正样本 | 高效，天然hot distribution | 可能采到假负样本 |

### 5.3 负样本偏差修正

In-batch负样本天然倾向于热门item（因为热门item被更多用户点击，在batch中出现更频繁）：

```python
# 采样修正：对热门item的得分做降权
corrected_score = score - log(item_frequency)
```

---

## 六、评估指标

### 6.1 离线指标

**AUC（Area Under ROC Curve）**：
```
衡量模型区分正负样本的能力
AUC = P(score_pos > score_neg)
工业界常用 GAUC (Group AUC)：按用户分组计算AUC再平均
原因：全局AUC会被活跃用户主导，GAUC更公平
```

**GAUC**：
```python
gauc = Σ_u (|I_u| / |I|) * AUC_u
# 每个用户的AUC按其样本量加权平均
```

**NDCG（Normalized Discounted Cumulative Gain）**：
```
DCG@K = Σ_{i=1}^{K} (2^rel_i - 1) / log2(i+1)
NDCG@K = DCG@K / IDCG@K
# 位置越靠前的item权重越大
# 工业界关注 NDCG@5, NDCG@10
```

**MRR（Mean Reciprocal Rank）**：
```
MRR = (1/|Q|) * Σ_q (1 / rank_q)
# 适合只关心第一个相关结果位置的场景（搜索）
```

### 6.2 在线指标

| 指标 | 定义 | 提升难度 |
|------|------|---------|
| CTR | 点击/曝光 | 中 |
| CVR | 购买/点击 | 高（稀疏） |
| GMV | 成交金额 | 终极目标 |
| 用户时长/DAU | 长期留存 | 难以直接优化 |
| 多样性 | 列表item分布熵 | 需要显式优化 |

### 6.3 离线 vs 在线 Gap

离线 AUC 涨不一定在线 CTR 涨，原因：
1. 离线样本不代表真实分布（position bias、曝光偏差）
2. 离线指标无法衡量多样性、新颖性
3. 模型变化影响用户行为分布（feedback loop）

---

## 七、多任务学习

### 7.1 为什么需要多任务

单纯优化 CTR 会导致标题党（高点击低购买）。实际需要同时优化：
- CTR（点击率）
- CVR（转化率）
- 停留时长
- 收藏/分享

### 7.2 ESMM（Entire Space Multi-Task Model）

```
曝光空间                    点击空间
    │                           │
    ├── CTR Task: p(click|imp)   │
    │         │                 │
    │         └── 点击 ──────→  CVR Task: p(buy|click)
    │                           │
    └── CTCVR = CTR × CVR ← 联合训练
```

**核心思想**：CVR 直接在曝光空间训练（解决点击稀疏问题），用 CTR×CVR=CTCVR 作为监督信号，embedding 在两个任务间共享。

### 7.3 MMoE（Multi-gate Mixture of Experts）

```python
# 多个Expert网络，每个任务有独立的Gate
experts = [Expert_1, Expert_2, ..., Expert_K]  # K个专家

for task in tasks:
    gate = softmax(Linear(input))  # (K,) 权重
    task_input = Σ_k gate[k] * experts[k](input)
    task_output = Tower_task(task_input)
```

**优点**：不同任务可以选择性地使用不同的专家，减少任务间干扰（seesaw problem）。

### 7.4 最终排序分数融合

```python
# 线性加权（最常用，可调）
score = w1 * p_ctr + w2 * p_cvr * price + w3 * p_like

# 乘法融合（保证各任务都不太差）
score = p_ctr^α * p_cvr^β * p_quality^γ
```

---

## 八、Position Bias

### 8.1 什么是 Position Bias

用户更倾向于点击列表靠前的item，即使内容相同。导致：
- 训练数据中靠前位置的click被过采样
- 模型学到"位置靠前 → 高分"而非"item好 → 高分"

### 8.2 修正方法

**Inverse Propensity Weighting (IPW)**：
```python
# 估计每个位置的曝光倾向 θ_k
# 给样本加权，位置越靠前权重越小
loss = BCE(y, p) / θ_position
```

**Position-aware 模型**：
```python
# 训练时把 position 作为特征输入
# 推理时把 position 置为0（或平均位置）
# 让模型学会"去除"位置影响
```

**两塔 Position Model（腾讯/字节方案）**：
```
训练: score = f(user, item, position)
推理: score = f(user, item, position=null)  # 去掉位置输入
```

---

## 九、样本构建中的常见问题

### 9.1 CVR 预估的 SSB 问题（Sample Selection Bias）

CVR 只能在点击样本上训练，但推理时是在曝光空间预测。

```
训练空间: {clicked items}  ← 有偏，只有点击才有转化标签
推理空间: {all exposed items}  ← 包含未点击的item
```

解决：ESMM（在曝光空间联合训练 CTR×CVR）

### 9.2 延迟反馈问题（Delayed Feedback）

用户购买行为有延迟（可能3天后才付款）。如果实时训练用当前数据，会把"未来才会转化的样本"标记为负样本。

解决：
1. **等待窗口**：等24小时再标注，牺牲时效性
2. **假负样本重标记**：先标为负，实际转化后重新标为正，加入训练
3. **延迟感知模型**：把"是否延迟标注"建模为额外任务

---

## 十、工程实现关键点

### 10.1 特征存储与拉取

```
离线特征:   HDFS → 特征平台(Hive/Spark) → 批量导入Redis/KV
近实时特征: Kafka → Flink实时聚合 → Redis
实时特征:   请求时在线计算 or 从内存cache读取

打分时:
  1. 接收请求（user_id + candidate_items）
  2. 批量从Redis拉取user特征（1次IO）
  3. 批量从Redis拉取item特征（1次IO，batch lookup）
  4. 拼装特征，送入模型
  5. 返回排序分数
```

### 10.2 模型部署

```
训练: GPU集群，TensorFlow/PyTorch
部署: TF Serving / TorchServe / TensorRT
优化:
  - INT8量化（精度损失<0.1% AUC）
  - 模型蒸馏（大模型→小模型）
  - Embedding压缩（PQ量化）
  - 算子融合（BatchNorm+ReLU合并）
```

### 10.3 Embedding Table 大小问题

大型推荐系统 Embedding Table 可达 TB 级别（亿级用户×64维×4字节）：
- **Parameter Server 架构**：Embedding 存在 PS，模型参数存在 Worker
- **Embedding 压缩**：哈希 trick、频率剪枝（低频ID不存储）、混合精度（FP16）

---

## 十一、常见面试题

### Q1: CTR 模型中 embedding 维度怎么选？

> 没有统一答案，经验法则：
> - 基数小的特征（如性别）：4~8维
> - 基数中等（如品类，千级）：16~32维
> - 基数大（如user_id, item_id，亿级）：64~256维
> - 原则：`dim ≈ min(600, round(cardinality^0.25 * 6))`（Google经验）
>
> 实践中用 embedding_dim 作为超参数，在小数据集上 grid search，或者用 NAS 自动搜索。

---

### Q2: 为什么 GAUC 比 AUC 更适合推荐系统？

> 全局 AUC 的问题：活跃用户产生大量样本，他们的 AUC 主导全局 AUC。如果模型对活跃用户准确但对低活跃用户很差，全局 AUC 仍然很高。
>
> GAUC 按用户分组计算 AUC 后加权平均，保证每个用户的贡献相对均等，更能反映模型在不同用户群体上的整体表现。
>
> 数学上：GAUC = Σ_u (|I_u|/|I|) × AUC_u，其中 |I_u| 是用户 u 的样本数，|I| 是总样本数。

---

### Q3: 精排模型特征数量很多时，如何防止过拟合？

> 1. **Dropout**：在 DNN 层加 dropout（0.1~0.3），推荐系统中常用
> 2. **L2正则**：对 embedding 加 L2 penalty，防止 embedding 过大
> 3. **Embedding 共享**：user_id 和 user 统计特征共享部分 embedding 空间
> 4. **特征选择**：用 IV（Information Value）或 feature importance 筛掉低信息特征
> 5. **数据增强**：随机 dropout 部分特征（模拟线上特征缺失）
> 6. **Early stopping**：在验证集 AUC 不再提升时停止

---

### Q4: 如何处理冷启动问题（新用户/新item）？

> **新用户冷启动**：
> - 用人口统计学特征（age, gender, region）初始化
> - 用注册时行为（前N次点击）快速构建兴趣
> - 召回阶段加大探索权重（bandit策略）
>
> **新item冷启动**：
> - 用 item 的内容特征（text/image embedding）作为 ID embedding 的初始值
> - 用相似 item 的 embedding 作为新 item 的 embedding（迁移）
> - 在粗排/精排阶段给新 item 加流量扶持（提分策略）
> - 快速收集真实CTR后更新模型（增量训练）

---

### Q5: CTR 模型上线后，线上 CTR 比预期低，怎么排查？

> 系统性排查框架：
>
> 1. **数据分布 shift**：检查线上特征分布是否与训练数据分布一致（training-serving skew）
> 2. **特征缺失/异常**：检查线上是否有特征拉取失败，用默认值替换导致偏差
> 3. **Position bias**：新模型是否改变了item位置分布，影响实际CTR
> 4. **人群差异**：线上流量是否是AB测试的随机样本，是否存在实验组偏差
> 5. **模型版本**：确认部署的模型版本和评估版本一致
> 6. **指标口径**：确认线上CTR计算口径（去重？去刷量？）和离线一致

---

### Q6: 如何设计一个多目标排序的 score 融合公式？

> 实践中常用**乘法融合 + 业务调权**：
>
> ```
> score = CTR^α × CVR^β × price_factor × quality_factor
> ```
>
> 设计原则：
> 1. 各任务预估概率先校准（isotonic regression / Platt scaling），确保数值范围可比
> 2. 乘法优于加法：乘法保证某个目标极差时整体分数也低，防止单一目标极端优化
> 3. 权重 α, β 通过业务目标（GMV/时长/满意度）在线实验调优，不能只看单指标
> 4. 价格等业务因素单独引入，不混入模型学习，保持可控性

---

### Q7: Transformer 用在推荐系统里有什么挑战？

> 1. **计算开销**：Self-attention 是 O(n²)，用户行为序列很长时开销大。解决：限制序列长度（取最近50条），或用 LinFormer/Performer 近似
> 2. **位置编码**：推荐序列的时间间隔不均匀，绝对位置编码不适合，需用相对时间编码或时间戳编码
> 3. **稀疏性**：ID 类特征极其稀疏，Transformer 的 embedding 需要大量数据才能训练好
> 4. **在线 serving 延迟**：Transformer 比 MLP 重很多，需要量化/蒸馏才能满足 <100ms 延迟
> 5. **过拟合**：用户行为序列中噪声多（误点击），Transformer 容易过拟合噪声行为

---

### Q8: 推荐系统中的 Exploration vs Exploitation 如何平衡？

> **Exploitation**：利用已知信息，推当前认为最好的item（短期收益高）
>
> **Exploration**：探索未知item，获取新信息（长期收益高）
>
> 常用方法：
> - **ε-greedy**：以ε概率随机推荐（简单粗暴）
> - **UCB（Upper Confidence Bound）**：score = μ + c×√(log(t)/n)，对曝光少的item给置信区间奖励
> - **Thompson Sampling**：对CTR的Beta分布采样，自然平衡探索与利用
> - **Bandit + 个性化**：为每个用户维护独立的 bandit 状态（LinUCB）
>
> 工业界通常在召回层加流量扶持（新item保量），精排层保留少量位置给探索，重排层做多样性保证。

---

## 十二、关键数字速查

| 参数/指标 | 典型值 | 备注 |
|---------|--------|------|
| 召回候选数 | 1000~5000 | 取决于item库大小 |
| 精排候选数 | 100~500 | |
| 最终展示数 | 10~50 | |
| Embedding维度 | 64~256 | user/item ID |
| DNN层数 | 3~5层 | |
| DNN每层节点 | 256~1024 | |
| 训练batch size | 1024~4096 | |
| 学习率 | 1e-4~1e-3 | Adam |
| 正负样本比 | 1:10~1:100 | 视任务而定 |
| 行为序列长度 | 50~200 | DIN；SIM可到百万 |
| AUC好的基线 | 0.72~0.75 | 场景差异大 |
| GAUC提升 | +0.001~0.003 | 有显著业务意义 |
| CTR绝对提升 | +0.1%~0.5% | 有显著业务意义 |
| 精排延迟要求 | <100ms | P99 |
| 特征数量 | 几十~几百 | |
| 模型参数量 | 亿~百亿 | Embedding为主 |
