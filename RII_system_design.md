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


  ┌──────────────────┬────────────────────────────────────┬────────────────────────────────────────────┬──────┐
  │      Route       │                File                │                Supervision                 │  LR  │
  ├──────────────────┼────────────────────────────────────┼────────────────────────────────────────────┼──────┤
  │ LoRA finetune    │ peft_finetune/main.py              │ image-text (frozen text encoder as anchor) │ 5e-4 │
  ├──────────────────┼────────────────────────────────────┼────────────────────────────────────────────┼──────┤
  │ Full finetune    │ full_finetune/main.py              │ image-text (frozen text encoder as anchor) │ 1e-5 │
  ├──────────────────┼────────────────────────────────────┼────────────────────────────────────────────┼──────┤
  │ Catalog finetune │ image_catalog_training/finetune.py │ image-image only (no text at all)          │ 1e-5 │
  └──────────────────┴────────────────────────────────────┴────────────────────────────────────────────┴──────┘

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
