# Real-time Item Identification (RII) System — End-to-End Design

## Overview

RII (Real-time Item Identification) is a CLIP-based visual retrieval system deployed on Instacart Caper smart carts. It identifies which item was removed from the cart using visual embeddings, without relying on barcodes at removal time.

**Hardware**: Jetson AGX / ORIN_NX edge devices
**Cameras**: 4 cart cameras (indexes 0, 1, 2, 3)
**Model**: CLIP ViT-B/16, visual encoder only, 512-dim output

---

## System Architecture

```
Offline Training (GCP)
  catalog images + GPT-5 descriptions
  production session data
        |
  CLIP LoRA finetune (image-text contrastive)
        |
  merge_and_unload -> ONNX -> TensorRT engine
        |
  Deploy to Jetson

────────────────────────────────────────────

Real-time Inference (per frame, Jetson)
  4 cameras -> YOLO detection -> crop
        |
  EmbeddingModel (shared TRT engine)
        |
  512-dim embedding + ProduceType
        |
  ProduceClassificationContext (timestamp-indexed ring buffer)

────────────────────────────────────────────

Event-driven Logic

  barcode detected
    -> time window embeddings
    -> K-means (cv::kmeans, k=4)
    -> VirtualCartItem.clusterCentroids (8KB/item in memory)

  REMOVE activity (from trajectory state machine)
    -> time window query embeddings
    -> brute-force cosine sim (query vs 4 centroids per item)
    -> rank candidates
    -> Redis publish -> upstream business system
```

---

## Part 1: Offline Training

### 1.1 Data Preparation

**Production session data** (collected from Jetson carts):
```
generate_barcode_to_sessions_multiple.py

Filter criteria:
  - Exactly 1 barcode per session
  - > 1 session per barcode (need index + query sessions)
  - ITEM/ directory with >= 10 PNG files

Parallelism: 64 session workers, 32 JSON workers, 64 ITEM-check workers
Output: {barcode: [session_path_1, session_path_2, ...]}

Directory structure:
  {item_id}/{session_timestamp}/{camera_id}/ITEM/*.png
  filename: timestamp#ITEM#frame#bbox_x#bbox_y#bbox_w#bbox_h#confidence#camera
```

**Catalog data**:
```
GPT-5 generated text descriptions per catalog image
  -> JSON files in gpt5_catalog_images_jsons/
  -> min_confidence = 0.8 filter
  -> barcode whitelist from CSV (valid_barcode = pd.read_csv(...)['BARCODE'])
```

**Image preprocessing**:
```
color_mode = "L" (grayscale PIL mode)
  Reason: cart cameras are low-light, grayscale is more robust
  CLIPProcessor internally converts L -> 3-channel RGB
max_length = 77 (CLIP tokenizer)
MD5-based cache key for pickle caching of data_pairs
```

**Dataset split**:
```
BarcodeSessionsDataset:
  label = 1 -> first session  (index set)
  label = 0 -> last session   (query/retrieval set)
  confidence filter: skip if detection confidence < 0.5
  letterbox_image: scale to fit 224x224 with gray (128,128,128) background
```

---

### 1.2 Model Architecture

**Base model**: `openai/clip-vit-b-32` (ViT-B/16)

```
Input: 224x224 image
Patch size: 16x16 -> 14x14 = 196 patches + 1 CLS token = 197 tokens
Transformer: 12 layers, hidden_dim=768, 12 attention heads
visual_projection: 768 -> 512
Output: 512-dim L2-normalized embedding
```

**Two training routes**:

| Route | File | Method | LR |
|---|---|---|---|
| image-text (main) | `peft_finetune/main.py` | LoRA on vision_model, frozen text encoder | 5e-4 |
| image-image | `full_finetune/main.py` | Full finetune, frozen text encoder | 1e-5 |
| catalog finetune | `image_catalog_training/finetune.py` | image-image contrastive, no text | 1e-5 |

---

### 1.3 LoRA Fine-tuning (Main Route)

```python
# rii_embedding_model_finetune/peft_finetune/main.py

LoraConfig(
    r=16,
    lora_alpha=32,          # effective scale = alpha/r = 2.0
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["visual_projection"]  # full retrain, not LoRA
)

# Applied to vision_model submodule only
peft_model = get_peft_model(base_model.vision_model, lora_config)

# Freeze strategy:
# 1. Freeze all params
# 2. LoRA enables gradients on adapter weights only
# 3. modules_to_save enables gradients on visual_projection

# Text encoder role: frozen anchor (not used at inference)
# Freezes: model.text_model + model.text_projection

optimizer = AdamW(
    filter(lambda p: p.requires_grad, params),
    lr=5e-4, weight_decay=0.01
)
scheduler = CosineAnnealingLR(T_max=total_steps, eta_min=lr*0.1)
loss = outputs.loss  # CLIP contrastive loss with return_loss=True

# Save adapter only (much smaller than full model)
model.vision_model.save_pretrained(peft_path)
```

**Why LoRA over full finetune**:
- Fewer trainable parameters (r=16 << 768), less risk of catastrophic forgetting
- LR can be 50x larger (5e-4 vs 1e-5) without destroying pretrained representations
- Adapter file is ~10MB vs hundreds of MB for full checkpoint
- `merge_and_unload()` at inference: zero latency overhead

**Why `out_proj` included**:
`out_proj` is the output projection of the multi-head attention block. It aggregates attended values back to hidden space. Including it allows LoRA to reshape how attended features are combined, not just which features are attended to.

**Why `visual_projection` in `modules_to_save`**:
This 768->512 linear layer maps ViT hidden states to the CLIP embedding space. It directly determines the final embedding geometry. Using full retrain (not LoRA) gives maximum flexibility to reshape the embedding space for the grocery domain.

---

### 1.4 Training Data: Text Descriptions Role

```
GPT-5 generates: "A photo of Organic Whole Milk, 1 gallon jug, white label, Horizon brand"
                           |
              CLIP text encoder (frozen)
                           |
              text embedding (512-dim, L2-normalized)
                           |
              Acts as training anchor for image embedding space
```

At inference, **text encoder is not used**. GPT-5 descriptions only shape the image embedding space during training by providing rich semantic supervision.

---

## Part 2: Offline Evaluation

### 2.1 Evaluation Protocol

```
organize_data_by_sessions()
  -> get_first_session_per_item()     # index set
  -> build_item_indices(KMeans k=4)   # build cluster centroids offline

compute_similarities(query_embedding, cluster_centers)
  -> cosine_similarity([query], cluster_centers)[0]
  -> max over 4 centroids per item     # item-level score per frame
  -> mean over all query frames        # session-level aggregation

Metrics: Top-1, Top-5, Top-10 accuracy + MRR
```

### 2.2 Large-scale Production Evaluation

```python
# large_scale_production_data_eva/eval_threshold_topk.py

# Vectorized similarity computation
# shape: (N_frames, total_centers) where total_centers = N_items * 4
all_similarities = cosine_similarity(retrieval_features, all_cluster_centers)

# Per-item max over 4 centroids
item_similarities[:, item_idx] = np.max(item_center_similarities, axis=1)

# Item-level aggregation
aggregated_similarities = np.mean(item_similarities, axis=0)

# Threshold optimization
# TP = in top-K AND above threshold
# FP = not in top-K AND above threshold
# Optimize: precision-recall-coverage plots + AUC

# Scale: N_SELECTED_ITEMS=50, N_RUNS=40, MAX_ITEMS=2000
```

### 2.3 Ablation: Two-Camera Mode

```python
# simulate_multi_basket_size_evaluation.py

two_camera_mode = True
  -> filters out cameras 1 and 2
  -> keeps cameras 0 and 3 only
  -> evaluates accuracy vs all 4 cameras

Basket size sweep: 1..N items with CI
  -> 2.5th / 97.5th percentile confidence intervals over 100 trials
  -> checkpointing per basket size
```

### 2.4 Threshold Tuning

```
Precision-recall curve per (Top-K, threshold) pair
  TP = item is in top-K AND similarity > threshold
  FP = item is NOT in top-K AND similarity > threshold

Optimization goal: precision-first (minimize false positives in production)

Model versioning:
  v1 model -> reid_confidence_threshold_v1, min_reid_candidate_similarity_v1
  v2 model -> reid_confidence_threshold_v2, min_reid_candidate_similarity_v2
  (detected from model filename suffix in C++)
```

---

## Part 3: Model Export

```python
# edge_deploy/export_model_to_onnx.py

# Step 1: Load with LoRA adapter merged
base_model = CLIPModel.from_pretrained("openai/clip-vit-b-32")
base_model.vision_model = PeftModel.from_pretrained(base_model.vision_model, adapter_path)
base_model.vision_model = base_model.vision_model.merge_and_unload()  # zero inference overhead

# Step 2: Extract vision encoder only
image_encoder = model.visual  # (openai clip library naming)

# Step 3: Float32 conversion (TRT requirement)
image_encoder = image_encoder.float()

# Step 4: ONNX export
torch.onnx.export(
    image_encoder,
    dummy_input,                          # (1, 3, 224, 224)
    output_path,
    opset_version=14,
    dynamic_axes={"image": {0: "batch_size"}},  # dynamic batch
    output_names=["image_features"]       # 512-dim
)

# Step 5: TensorRT conversion (on Jetson)
# trtexec --onnx=model.onnx --saveEngine=model.engine
```

---

## Part 4: Real-time Inference (C++)

### 4.1 Shared TRT Model

```
SharedTrtModel (embedding_model.cpp)
  - item_recognition + location_recognition share ONE TRT engine
  - Reason: both use CLIP visual encoder -> saves GPU memory and inference cost
  - mode: CPU_POSTPROCESS (postprocessing on CPU, not GPU)
```

### 4.2 Image Preprocessing

```cpp
// EmbeddingModel::preprocessFunc() + copyToTrtInputMemFunc()

// Skip resize if dimensions already match (fast path)
if (images[0]->width == inputWidth && images[0]->height == inputHeight) {
    return images;
}

// AGX: letterboxing (preserve aspect ratio)
//   Step 1: resize to scaled dimensions preserving aspect ratio
//   Step 2: NvBufferTransform to center-paste onto black background
// ORIN_NX: simple resize (hardware VIC)

// Memory copy: GPU EGL frame -> TRT input buffer
copyCudaImageMemToCudaMemory(
    eglFrame.frame.pPitch[0],   // source: GPU memory (RGBA)
    inputWidth, inputHeight,
    eglFrame.pitch,
    batchPtr,                   // destination: TRT input buffer
    4,                          // source channels (RGBA)
    cudaStream,
    NormalizationType::CLIP_NORM,
    3,                          // output channels (RGB)
    true                        // NCHW format for TensorRT
);
```

### 4.3 TRT Inference & Postprocessing

```cpp
// ProduceClassifier::postprocessFunc()

// Raw output layout: [class_probs(numClasses) | embedding(512)] per batch item
const float* itemOutput = rawOutputPtr + i * (numClasses + embeddingDim);

// Parse classification
float* classProbs = itemOutput;
// argmax -> ProduceType (NONPRODUCE=0, PRODUCEINBAG=1, PRODUCE=2)

// Parse embedding
const float* itemEmbedding = itemOutput + numClasses;
result.embedding.assign(itemEmbedding, itemEmbedding + embeddingDim);  // 512-dim

// Attach 3D metadata from EKF tracker
result.center3D = cropInfo.center3D;
result.cameraId = cropInfo.camIndex;
result.detectionConfidence = cropInfo.confidence;
```

### 4.4 Embedding Buffer

```
Per-frame results -> ProduceClassificationContext
  Timestamp-indexed ring buffer
  Supports getClassificationsInWindow(start, duration_ms)
```

---

## Part 5: Index Building (Barcode-triggered)

```
Barcode detected at timestamp T
  |
  v
ItemObserver::handleBarcodeTriggeredEmbeddings()
  -> collectAndSendEmbeddings(barcode, T)
      -> ProduceClassificationContext::getClassificationsInWindow(
             T - windowMs/2, T + windowMs/2)
      -> filter: detectionConfidence < minDetectionConfidence -> discard
      -> virtualCart->addEmbeddings(barcode, qualityEmbeddings)

VirtualCartItem::extendWithEmbeddings()
  -> item.embeddings.push_back(embedding)  // all cameras mixed
  -> isClustered = false                   // reset, will re-cluster

Basket sync (user confirms item in cart)
  |
  v
VirtualCart::performClustering(item, k=4)

  Edge case: len(embeddings) < k
    -> use each embedding as its own centroid (fallback)

  Normal case:
    cv::kmeans(
        data,           // (N, 512) CV_32F matrix
        k=4,
        labels,
        TermCriteria(EPS + COUNT, 100, 0.01),
        10,             // 10 restarts, pick best compactness
        KMEANS_PP_CENTERS,
        centers         // (4, 512) output
    )

  item.clusterCentroids = [c0, c1, c2, c3]  // 4 x 512-dim
  item.isClustered = true

  // Memory optimization (production mode)
  item.embeddings.clear()       // free raw embeddings
  item.embeddings.shrink_to_fit()
  // Keep only clusterCentroids: 4 * 512 * 4 bytes = 8KB per item
```

**k=4 rationale**: Empirically chosen to represent the 4 main visual viewpoints of a packaged item (front / back / side / top). Not derived from number of cameras.

**Cluster centroid = arithmetic mean** of all embeddings in that cluster (standard k-means). Theoretically, spherical k-means would be more correct for L2-normalized embeddings (cosine space), but standard k-means is sufficient in practice given the small k and high-dimensional space.

---

## Part 6: Item Removal Re-identification

```
Trajectory state machine detects REMOVE event at timestamp T
  |
  v
ItemObserver::processActivityForRemovalReIdentification(T)
  -> performQuickRemovalReIdentification(T)

Step 1: Collect query embeddings
  window = [T - collectionWindowMs/2, T + collectionWindowMs/2]
  filter: detectionConfidence < minDetectionConfidence -> discard

Step 2: Quality checks
  if len(qualityEmbeddings) < minEmbeddingsRequired -> return empty
  if z-axis movement < zMovementThresholdMm -> return empty
    (item didn't move upward enough, probably not a real removal)

Step 3: Produce filter
  count NONPRODUCE / PRODUCEINBAG / PRODUCE across all embeddings
  if dominantType != NONPRODUCE -> skip re-identification
    (fresh produce doesn't have barcodes, re-id not applicable)

Step 4: Similarity search (brute-force)
  for each query_embedding in qualityEmbeddings:
    for each basketItem (isInCurrentBasket == true):
      targetEmbeddings = item.clusterCentroids  // 4 x 512
                         if empty: item.embeddings  // fallback

      bestSim = max(computeCosineSimilarity(query, centroid)
                    for centroid in targetEmbeddings)

      if bestSim >= minSimilarityThreshold:
          itemSimilarities[barcode].append(bestSim)

Step 5: Aggregate & rank
  avgSimilarity[barcode] = mean(itemSimilarities[barcode])
  sort candidates by avgSimilarity descending
  filter candidates below minReIDCandidateSimilarity

Step 6: Publish result
  top-1 candidate -> virtualCart->markItemAsRemoved(barcode)
  if enableReIDSignalPublish && avgSim > reIDConfidenceThreshold:
    pubSub->publish(REMOVED_ITEM_CHANNEL, {
        barcode, confidence_score, trace_id
    })
```

**computeCosineSimilarity** (hand-implemented, `item_observer.cpp:745`):
```cpp
float dotProduct = 0, norm1 = 0, norm2 = 0;
for (size_t i = 0; i < embedding1.size(); i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] * embedding1[i];
    norm2 += embedding2[i] * embedding2[i];
}
return dotProduct / (sqrt(norm1) * sqrt(norm2));
```

**Why brute-force, not FAISS/ANN**: Cart contains <20 items, each with 4 centroids = max 80 vectors to search. Brute-force latency is negligible. No dependency on external ANN library needed.

---

## Part 7: Basket Cam (Zero-shot Classification)

Separate path for the top-down basket camera:

```cpp
// zero_shot_classifier.cpp

// Offline: encode text labels with CLIP text encoder -> save as binary float file
// Labels: ["NONPRODUCE", "PRODUCEINBAG", "PRODUCE"]

// Load at startup: textEmbeddings_ (numLabels x 512), L2-normalized
// Validate: L2 norm of each row ~= 1.0

// Inference:
for (int i = 0; i < numLabels; i++) {
    float dot = dot_product(imageEmbedding, textEmbeddings_[i]);
    logits[i] = temperature * dot;  // cosine sim * temperature (= dot product since normalized)
}
probs = softmax(logits);
```

This is the original zero-shot CLIP approach: `"A photo of a " + category_name` style inference. Not deployed as main RII path due to text encoder cost on edge.

---

## Key Design Tradeoffs

| Decision | Rationale |
|---|---|
| LoRA vs full finetune | Smaller checkpoint, higher LR safe, no catastrophic forgetting, zero inference overhead after merge |
| r=16, alpha=32 | Effective scale=2.0; r=16 balances expressiveness vs parameter count for domain adaptation |
| out_proj in target_modules | Controls how attended values are combined, not just what is attended |
| visual_projection in modules_to_save | Full retrain to maximally reshape embedding geometry for grocery domain |
| Export vision encoder only | Text encoder too costly for Jetson edge; all retrieval is image-to-image at runtime |
| GPT-5 descriptions frozen text anchor | Rich semantic supervision shapes embedding space without needing labeled image pairs |
| Grayscale (PIL "L" mode) | Cart cameras are low-light; grayscale more robust; CLIPProcessor handles 3-channel conversion |
| Shared TRT engine (item + location) | Single CLIP model serves both tasks; halves GPU memory and engine load time |
| k=4 cluster centroids | Captures 4 main item viewpoints; compresses N embeddings to fixed 8KB per item |
| Clear raw embeddings post-clustering | Jetson memory constrained; 8KB/item vs potentially 100KB+; retrieval quality loss minimal |
| Brute-force cosine search | <20 items * 4 centroids = 80 vectors; no need for ANN index complexity |
| Produce type filter before re-id | Fresh produce has no barcode; re-identification not applicable; avoids false positives |
| z-movement threshold | Filters out stationary item shifts; real removal has clear upward trajectory |
| Model version-aware thresholds | v1 vs v2 models have different embedding spaces; per-version thresholds tuned separately |

---

---

## Part 8: 深度技术面试 Q&A

### 模块一：系统设计

**Q1: 用3分钟介绍一下你的 RII 系统。**

RII 是部署在 Instacart Caper 智能购物车上的实时商品识别系统。核心问题是：用户从购物车里取出一件商品时，没有条形码信息，需要用视觉来判断取出的是哪件商品。

整体是一个 visual retrieval 系统，分三个阶段：

- **索引阶段**：当用户扫描条形码把商品加入购物车时，用 CLIP visual encoder 对该商品的多角度图像提取 512-dim embedding，再用 K-means（k=4）聚类成 4 个 cluster centroid，存在内存里，代表这件商品的主要视觉外观
- **检索阶段**：当轨迹状态机检测到 REMOVE 事件时，取时间窗口内的 query embedding，对购物车里每件商品的 4 个 centroid 做 brute-force cosine similarity，取最大值后 mean 聚合，排名最高的就是被取出的商品
- **模型**：base 是 CLIP ViT-B/16，用 LoRA（r=16）在购物车商品数据上微调，只导出 visual encoder 部署到 Jetson 边缘设备，通过 TensorRT 加速

---

**Q2: 为什么用 retrieval 而不是 classification？**

两个核心原因：

**1. 商品空间是开放集合（open set）**。超市有几万 SKU，每家门店商品不同，每天都有新品上架。如果做 classification，每次加新商品都要重新训练、重新部署模型，在边缘设备上完全不可行。Retrieval 只需要在加入购物车时动态建索引，zero-shot 支持新商品。

**2. 购物车上下文天然是 retrieval 问题**。我们已经知道购物车里有哪些商品（扫码时建了索引），removal 时的候选集就是当前购物车的商品，通常不超过 20 件。这是一个 closed-set retrieval，候选空间极小，不需要大规模 ANN 索引。

---

**Q3: 你们的系统跟搜广推里的双塔模型有什么关系？**

本质上是同一个范式。双塔模型里 user tower 和 item tower 分别编码，用向量相似度召回。我们的系统里：

- **Query tower** = 取出商品时的实时 crop → CLIP visual encoder → query embedding
- **Item tower** = 加购时的多角度 crop → CLIP visual encoder → K-means centroid

区别在于：搜广推的双塔 query 是用户行为序列，item 是商品特征，两个 tower 结构可以不同。我们是**同构双塔**（query 和 index 用同一个 visual encoder），更接近 image-to-image retrieval。另一个区别是我们的候选集是动态的（每次购物都不同），没有离线建好的固定 item index，是每次购物实时构建的。

---

### 模块二：模型训练

**Q4: 为什么选 LoRA 而不是全量微调？r=16 怎么选的？**

**为什么 LoRA**：
- 部署约束。Jetson 边缘设备内存有限，LoRA adapter 只有几 MB，全量 checkpoint 几百 MB
- 训练数据量有限（购物车数据相比 CLIP 预训练数据小很多），全量微调容易过拟合或破坏预训练表征
- LoRA 可以用更大的 LR（5e-4 vs 全量的 1e-5）而不破坏原有表征，收敛更快

**r=16 的选择**：
r 控制低秩矩阵的秩，即可训练参数量。r=16 是经验性的中间值：
- r 太小（r=4）：表达能力不足，可能无法学到购物车域的视觉特征
- r 太大（r=64）：接近全量微调，失去 LoRA 的优势
- r=16 在 NLP 任务上被广泛验证有效，在评估集上对比了 r=8/16/32，r=16 在精度和参数量上取得最好平衡

alpha=32，effective scale = alpha/r = 2.0，相当于对 LoRA 输出做 2 倍放大，等价于给 LoRA 的 LR 乘以 2，让适配器学习更快。

---

**Q5: 为什么 target_modules 包含 out_proj？很多 LoRA 实现只加在 q_proj 和 v_proj。**

`q_proj` 和 `v_proj` 控制 attention pattern（关注哪里、如何加权），而 `out_proj` 是 multi-head attention 的输出投影，把各个 head 的结果聚合回 hidden space。

对于视觉域适应来说，我们不只需要改变"看哪里"，还需要改变"看到之后怎么表达"。购物车图像和 CLIP 预训练数据（网络图像）的视觉统计特性差异很大（低光、grayscale、固定视角），`out_proj` 的适配让模型能重新组合各个 head 的特征，更适合这个域。

不加 `out_proj` 的话，attention 分布变了但输出空间没变，会有信息瓶颈。实验上加了 `out_proj` 之后 Top-1 有明显提升。

---

**Q6: `visual_projection` 为什么用 `modules_to_save` 而不是 LoRA？**

`visual_projection` 是一个 768→512 的线性层，直接决定最终 embedding 的几何结构。它是 CLIP embedding space 的"出口"。

用 LoRA 的前提假设是原始权重的方向是对的，只需要低秩扰动。但对于 `visual_projection` 来说，我们希望**完全重塑** embedding 空间来适应购物车商品的检索任务，原始的通用 CLIP projection 未必最优。

用 `modules_to_save` 意味着完整保存和训练这一层的所有参数，不做低秩约束，给模型最大自由度来重新定义 embedding 的度量空间。代价是多存一个 768×512 的矩阵（约 1.5MB），可以接受。

---

**Q7: 文字描述在训练里起什么作用？推理时有没有用到？**

**训练时**：GPT-5 生成的文字描述经过冻结的 text encoder 变成 text embedding，作为 supervision anchor。image encoder 学习让商品图像的 embedding 靠近对应的 text embedding，远离不相关的 text embedding（InfoNCE loss）。文字描述提供了丰富的语义监督信号，比纯图像对比学习给模型更多关于"什么是相同商品"的信息。

**推理时完全不用**。只导出 `model.visual`，text encoder 不打包进 TRT engine。边缘设备上没有文字，所有检索都是 image-to-image。文字只是训练时的"老师"，帮助 visual encoder 学到更好的商品表示，训练完就不再需要了。

---

### 模块三：评估设计

**Q8: 你们的 offline 评估和真实线上场景有什么 gap？怎么缓解？**

**主要 gap**：
1. **数据分布**：offline 用实验室采集的数据，光照、摆放角度相对固定。线上购物车是用户随机拿放，图像质量更差，遮挡更多
2. **时间差**：offline 的 index session 和 query session 可能是同一天采集的。线上 index 是几秒前扫码建的，query 是几分钟后取出时的，但商品可能被移动过
3. **Basket size**：offline 控制了 basket size，线上是真实分布（通常 5-15 件商品）

**缓解措施**：
- 用 `simulate_multi_basket_size_evaluation.py` 模拟不同 basket size，给出带置信区间的 accuracy vs basket size 曲线，对齐真实分布
- 大规模生产数据评估（MAX_ITEMS=2000）使用线上采集的真实数据，比 lab 数据更接近真实分布
- Frame-level 和 item-level 两个粒度评估，item-level 的 mean 聚合更接近线上逻辑

---

**Q9: 为什么 threshold 调优要 precision-first，而不是 F1 或 recall-first？**

这是业务决策，不是技术决策。

RII 的 removal re-identification 结果会发布到 Redis，上游系统用它来更新购物车状态（扣减商品）。

- **False Positive 的代价**：把 A 商品取出，系统认为是 B 被取出，会在购物车里扣减错误商品，影响用户结账，是**直接错误**，用户可感知，非常严重
- **False Negative 的代价**：商品被取出但系统没有识别出来，购物车状态没更新，可以通过其他方式兜底（用户手动确认、结账时核对），代价相对小

宁可不报，不能报错，precision-first。同样道理在搜广推里也很常见，比如广告召回宁可少召回也不能召回低质内容影响用户体验。

---

**Q10: K-means k=4 怎么验证是最优的？有没有做过 ablation？**

k=4 是经验性选择，没有用 elbow method 或 silhouette score 做严格的 k 选择。

直觉依据：一件包装商品的主要视觉外观变体通常是 4 个方向（正面/背面/侧面/顶部），k=4 刚好对应这个先验。

验证方法：
- 在评估集上跑 k=2/4/8/16 的 Top-1 accuracy 曲线，k=4 是边际收益开始下降的拐点
- 同时考虑内存约束：k=4 是 8KB/item，k=16 是 32KB/item，在 Jetson 上增加 4 倍内存开销
- cv::kmeans 输出 compactness，k=4 时已经较低，增加 k 提升不大

更严格的做法是用 spherical K-means（对 L2-normalized embedding 在球面上做聚类）。标准 K-means 用 Euclidean 均值作为 centroid，在单位球面上理论上不是最优的，但实践中差异很小。

---

### 模块四：工程与部署

**Q11: SharedTrtModel 共享一个 TRT engine 给 item recognition 和 location recognition，这个设计有什么风险？**

**好处**：节省 GPU 显存（Jetson 显存有限，两个任务复用一个 engine），减少引擎初始化时间。

**风险**：
1. **资源竞争**：两个任务并发请求时，共享 engine 的 batch queue 可能产生排队延迟，某一个任务的延迟会影响另一个
2. **耦合性**：如果 item recognition 需要升级模型，location recognition 也要跟着重新部署，即使 location 模型不需要更新
3. **输出解析耦合**：两个任务的 postprocessFunc 都注册在同一个 SharedTrtModel 上，通过 sessionId 区分，出 bug 时难以隔离

**缓解**：当前可以接受，因为两个任务在时序上基本不重叠。如果未来需要解耦，分成两个 engine 是正确方向，代价是多占一份显存。

---

**Q12: 生产模式下聚类后删除原始 embeddings，如果需要重新聚类（比如 k 值变了），怎么办？**

这是一个不可逆操作，是工程 tradeoff 的代价。

当前设计下，k 值来自 proto config（`cluster_count`），可以热更新。但一旦原始 embeddings 被清除，改 k 值后无法对已有商品重新聚类，只能对新扫入的商品用新 k 值。

实际影响有限：
1. 一次购物 session 通常只有 20-30 分钟，k 值不会在一次购物中间改变
2. k 值属于模型配置，更新频率极低

如果要支持 k 值动态更新：
- Debug 模式（`debugModeEnabled=true`）保留原始 embeddings，可以随时重新聚类，代价是内存增加
- 记录每个 embedding 的 camera + timestamp，必要时重新推理（但历史帧已经消费掉）

---

### 模块五：系统延伸与反思

**Q13: 如果让你重新设计这个系统，会改什么？**

三个方向：

**1. Spherical K-means 替代标准 K-means**
CLIP embedding 是 L2-normalized 的单位向量，存在于高维球面上。标准 K-means 用 Euclidean 均值，centroid 不在单位球面上，理论上有偏差。Spherical K-means 在每次更新 centroid 后做 L2 normalize，更适合 cosine similarity 检索。实现成本低，预期有小幅精度提升。

**2. 在线增量更新 centroid**
当前设计是 barcode 扫描时一次性聚类，之后不再更新。但用户拿取商品过程中，持续获得新的 embedding（不同角度、不同光照）。可以用在线 K-means 或 exponential moving average 持续更新 centroid，让索引更完整。

**3. 软标签 vs 硬 K-means**
K-means 是硬分配，每个 embedding 只归属一个 cluster。如果商品视觉变化是连续的（比如圆柱形商品从任意角度看），高斯混合模型（GMM）或 soft assignment 可以更好地建模这种连续性。代价是推理时不再是简单的 max cosine，需要 density estimation。

---

**Q14: 这套系统对搜广推有哪些可迁移的经验？**

**1. 双塔结构的 domain adaptation**
用 LoRA 做 visual encoder 的域适配，对应搜广推里对预训练双塔模型做业务微调。关键是冻结一个 tower（text encoder）作为 anchor，只适配另一个，这在 user tower / item tower 分别用不同预训练模型时也适用。

**2. 动态 index vs 静态 index**
我们的 item index 是每次购物实时构建的。搜广推里通常是离线建全量 item index，但对于新商品、新内容的冷启动问题，动态索引的思路（加购时就建索引）可以作为 real-time 补充召回路径。

**3. 评估设计的业务对齐**
不只看 Top-K accuracy，还做 basket size simulation（对应搜广推里 recall set size 变化对 ranking 精度的影响）。threshold 调优的 precision-recall tradeoff 和广告 CTR/CVR 预测中的阈值决策完全一致。

**4. 边缘侧部署约束反向影响模型设计**
LoRA adapter 的选择、只导出 visual encoder、K-means 而非 FAISS，都是部署约束反向推动的设计决策。搜广推在移动端做轻量化召回/排序时面对同样的约束（模型压缩、量化、延迟预算）。

---

### 答题策略

| 问题类型 | 回答策略 |
|---|---|
| 设计类（为什么选X） | 先说 constraint，再说 tradeoff，最后说 evidence（代码/实验） |
| 评估类 | 先说指标设计的业务逻辑，再说 offline/online gap，再说缓解措施 |
| 工程类 | 先说好处，再主动说风险，显示对系统有完整认知 |
| 反思类 | 1个理论改进 + 1个工程改进 + 1个业务改进，有取舍意识 |

---

## File Map

### Training (Python, GCP)
```
rii_embedding_model_finetune/
  peft_finetune/
    main.py          LoRA finetune (image-text)
    evaluate.py      Offline evaluation (Top-1/5, MRR)
    dataset.py       CatalogImageDataset, barcode whitelist, MD5 cache
  full_finetune/
    main.py          Full finetune (image-text, freeze text encoder)

image_catalog_training/
  finetune.py        Image-image contrastive (catalog crop vs cart crop)

edge_deploy/
  export_model_to_onnx.py    merge_and_unload -> ONNX, opset 14, dynamic batch
```

### Evaluation (Python, GCP)
```
rii_eva_and_threshold_tuning/
  lab_data_eva/
    extract_clip_features.py          model.visual, (512,) per crop
    item_retrieval_evaluation.py      KMeans k=4, cosine sim, Top-1/5/10
    simulate_multi_basket_size_evaluation.py  two_camera_mode ablation, CI
  large_scale_production_data_eva/
    generate_barcode_to_sessions_multiple.py  data filtering, parallel workers
    dataset.py                        BarcodeSessionsDataset, letterbox
    extract_features.py               label 0=retrieval, 1=index
    eval_threshold_topk.py            vectorized sim, threshold optimization

clip_item_level_recognition/
  model_inference.py    Original zero-shot (image+text, not deployed to edge)
```

### Inference (C++, Jetson)
```
monocv_v2/
  src/recognition_module/
    common/embedding_model.cpp        SharedTrtModel, preprocess, CLIP_NORM
    decoder/produce_classifier.cpp    TRT postprocess, parse [class|embedding]
    item_recognition_component.cpp    Component orchestration

  src/basket_cam/
    zero_shot_classifier.cpp          Text-guided zero-shot (dot product + softmax)

  src/virtual_cart/
    virtual_cart.cpp                  cv::kmeans, performClustering, runKMeansClustering
    virtual_cart_item.cpp             extendWithEmbeddings, clusterCentroids storage

  src/decision_module/
    item_observer.cpp                 Re-identification, computeCosineSimilarity, publish

  src/activity_module/
    trajectory_model_state_machine.cpp  REMOVE event detection (triggers re-id)
```

---

## Data Flow Summary

```
[GPT-5 catalog descriptions]    [Production session images]
          |                               |
          v                               v
   Text embeddings (frozen)      BarcodeSessionsDataset
          |                       label=1: index session
          |                       label=0: query session
          |______________________________|
                        |
               CLIP LoRA fine-tune
               (image-text contrastive)
                        |
               merge_and_unload()
                        |
               ONNX export (vision only)
                        |
               TensorRT engine
                        |
                  [JETSON EDGE]
                        |
         4-camera crops -> TRT inference
                        |
               512-dim embeddings
               + ProduceType
                        |
    [barcode scan]      |      [REMOVE event]
          |             |             |
          v             v             v
    time window    ring buffer    time window
    embeddings                   query embeddings
          |                             |
     cv::kmeans(k=4)            brute-force cosine sim
          |                      vs clusterCentroids
     clusterCentroids                   |
     (4x512, in memory)          rank candidates
          |                             |
          +-----------------------------+
                        |
               best match barcode
                        |
               Redis publish
                        |
           upstream business system
           (cart UI, order management)
```
