# CILP: Multimodal Learning & Fusion Assessment

## 1. Project Overview
This project explores multimodal learning techniques for fusing **RGB images** and **LiDAR depth maps** to classify geometric shapes (Cubes vs. Spheres). The implementation focuses on two main objectives:
1.  **Architecture Comparison:** Analyzing different multimodal fusion strategies (Late Fusion vs. Intermediate Fusion variants) and downsampling techniques (MaxPool vs. Strided Convolution).
2.  **Contrastive Image-LiDAR Pretraining (CILP):** Implementing a multi-stage pipeline to align representations across modalities using contrastive learning, followed by a cross-modal projector and a final classifier.

The project demonstrates how contrastive pretraining can effectively align distinct modalities (Vision + Depth) to achieve high accuracy (>95%) on downstream tasks, even with limited synthetic data.

## 2. Setup Instructions

### Google Colab (Recommended)
1.  Upload the `.ipynb` notebooks and the `src/` folder to your Google Drive.
2.  Upload the dataset `assessment.zip` to your Drive (e.g., under `CILP_Project/data/`).
3.  Open the notebooks and mount your Drive when prompted:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
4.  Install dependencies at the start of the session:
    ```bash
    !pip install wandb torch torchvision matplotlib tqdm
    ```


## 3. Experiment Tracking (W&B)
All training runs, metrics, and visualizations are logged to Weights & Biases.

* **Project Name:** `cilp-extended-assessment`
* **W&B Project Link:** https://wandb.ai/rishikeshbharti007-university-of-potsdam/cilp-extended-assessment
## 4. Summary of Results

### Task 3: Fusion Architecture Comparison
We compared Late Fusion against three variants of Intermediate Fusion. Due to the synthetic nature of the dataset, all models achieved perfect accuracy, but **Intermediate Hadamard** proved most efficient.

| Architecture | F1 Score | Validation Loss | Parameters | GPU Mem (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **Late Fusion** | 1.0000 | 0.0001 | 372,743 | 281.1 |
| **Int (Concat)** | 1.0000 | 0.0001 | 372,743 | 281.1 |
| **Int (Addition)** | 1.0000 | 0.0022 | **364,551** | 281.1 |
| **Int (Hadamard)**| **1.0000** | **0.0000** | **364,551** | 281.1 |

### Task 4: Ablation Study (MaxPool vs. Strided Conv)
We analyzed the impact of replacing fixed `MaxPool2d` layers with learnable `Strided Convolution` layers.

| Architecture | Val Loss | Parameters | Time/Epoch | Result |
| :--- | :--- | :--- | :--- | :--- |
| **MaxPool2d (Baseline)** | 0.0001 | 372,743 | ~1.1s | Stable convergence |
| **Strided Conv** | 0.0001 | 473,243 | ~1.2s | Higher parameter count |

### Task 5: CILP Assessment Performance
The final CILP model uses a **Cosine Annealing Scheduler** and **MaxPooling Encoders** to achieve optimal alignment.


| Stage | Metric | Target | Achieved |
| :--- | :--- | :--- | :--- |
| **1. Contrastive Pretrain** | Val Loss | < 3.5 | **~1.50** |
| **2. Projector** | MSE | < 2.5 | **0.00** |
| **3. Final Classifier** | Accuracy | > 95% | **93.0%** |

## 5. Instructions to Reproduce

### Reproducibility
* **Seed:** Global seed is set to `51` in all notebooks to ensure deterministic results.
* **Data Split:** Training on random 10% subset, Validation on random 2% subset (shuffled).

## 6. Known Issues & Limitations

* **Class Imbalance:** The raw dataset is heavily imbalanced towards "Cubes."
    * *Fix:* A custom shuffling logic was implemented in `MultimodalDataset` to ensure balanced batches during subset creation. Without this, models collapse to 0.0 F1 score.
* **Contrastive Instability:** Using a constant learning rate (e.g., `1e-3`) often causes Stage 1 loss to plateau at ~3.41 (random guessing).
    * *Fix:* A `CosineAnnealingLR` scheduler is required to decay the learning rate, allowing the model to converge to a loss < 3.0.
* **Data Simplicity:** The dataset consists of simple geometric shapes on a black background. As a result, properly tuned models can easily hit 100% accuracy, making it difficult to distinguish fine-grained architectural advantages without restricting data size significantly.
