# Face Recognition System — Deep Learning Laboratory Project

## A Deep Learning Pipeline for Football Player Identification

---

## Project Overview

This project implements a complete **face recognition pipeline** as part of the Deep
Learning Laboratory coursework. The objective is to build, evaluate, and deploy a system
capable of identifying 5 football players — **Lionel Messi**, **Cristiano Ronaldo**,
**Kylian Mbappé**, **Erling Haaland**, and **Neymar Jr** — from uploaded facial images.

The pipeline includes **three model architectures**, an **optimiser comparison study**,
**regularisation techniques**, **comprehensive evaluation**, and a **live web deployment**
via Hugging Face Spaces with Gradio.

---

## Project Structure

```
college-Deeplearning-FaceRecognition/
├── FaceRecoFinal.ipynb          # Main notebook (data loading, preprocessing, training, evaluation)
├── app.py                       # Gradio web application
├── Dockerfile                   # Docker configuration for Hugging Face Spaces
├── requirements.txt             # Python dependencies
├── model.keras                  # Trained Keras model (not in repo, generated at runtime)
├── target_names.json            # Class label mapping
└── players/                     # Local dataset folder (5 subfolders, one per player)
```

---

## Table of Contents

1. [General Objectives](#1-general-objectives)
2. [Technical Requirements](#2-technical-requirements)
3. [Dataset Selection and Challenges](#3-dataset-selection-and-challenges)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis)
5. [Model Architectures](#5-model-architectures)
6. [Training and Optimisation Study](#6-training-and-optimisation-study)
7. [Regularisation Strategies](#7-regularisation-strategies)
8. [Evaluation and Results](#8-evaluation-and-results)
9. [Error Analysis](#9-error-analysis)
10. [Deployment](#10-deployment)
11. [Discussion](#11-discussion)
12. [Conclusion](#12-conclusion)
13. [How to Run](#13-how-to-run)

---

## 1. General Objectives

| Objective | Implementation |
|-----------|---------------|
| Artificial Neural Networks (ANN) | MLP Baseline on flattened pixels |
| Convolutional Neural Networks (CNN) | Custom 3-block CNN with BatchNorm |
| Transfer Learning | ResNet50 pre-trained on ImageNet |
| Optimisation Techniques (SGD, Adam) | Compared on the intermediate CNN |
| Regularisation (Dropout, Early Stopping) | Applied across all three models |
| Model Evaluation and Comparison | Accuracy, confusion matrices, learning curves, classification reports |
| Deployment | Gradio app on Hugging Face Spaces (Docker SDK) |

---

## 2. Technical Requirements

### 2.1 Data

- **Real, publicly available dataset**: Custom face dataset of 5 football players
  (images scraped from publicly accessible online sources, equivalent to the **LFW**
  benchmark specified in the project brief).
- **Proper train/validation/test split**: 70% training, 15% validation, 15% test
  (stratified by class).
- **Data preprocessing and normalization**:
  - All images resized to 256×256 pixels, then downscaled to 128×128 (CNN) or
    224×224 (ResNet50).
  - Pixel values normalized to `[0, 1]`.
  - ResNet50 preprocessing: BGR conversion + channel-wise mean subtraction
    (mean = [103.939, 116.779, 123.68]).
- **Exploratory Data Analysis (EDA)**: Class distribution histograms, sample image
  visualisation, class balance verification.

### 2.2 Models

Three models were implemented and compared:

| # | Model Type | Architecture Details |
|---|-----------|---------------------|
| 1 | **Baseline** | MLP with 2 hidden layers (512 → 256 → 5) + Dropout |
| 2 | **Intermediate** | Custom CNN: 3 Conv blocks (32→64→128 filters), BatchNorm, GlobalAveragePooling, Data Augmentation |
| 3 | **Advanced** | Transfer Learning with ResNet50 (ImageNet pre-trained, frozen base, custom classification head) |

### 2.3 Training & Optimisation Study

- **Two optimisers compared**: Adam vs SGD (momentum=0.9, nesterov=True) on the
  intermediate CNN.
- **Regularisation techniques**: Dropout (0.4–0.5), Early Stopping (patience=8–10).
- **Overfitting/underfitting analysis**: Documented through learning curves; the
  custom CNN exhibited underfitting due to limited data, while ResNet50 achieved good
  generalisation.

### 2.4 Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-score (per class and macro-averaged).
- **Confusion matrix**: Generated for the best model (ResNet50).
- **Learning curves**: Training and validation loss/accuracy plotted for all three
  models.
- **Comparative model analysis**: Documented performance gap between custom CNN
  and transfer learning.
- **Error analysis**: Misclassified samples visualised with true vs predicted labels.

### 2.5 Deployment

- **Platform**: Hugging Face Spaces with **Docker SDK**.
- **Interface**: Gradio web application (image upload → prediction).
- **Live URL**: [https://huggingface.co/spaces/tahazeroug/college-Deeplearning-FaceReco](https://huggingface.co/spaces/tahazeroug/college-Deeplearning-FaceReco)

---

## 3. Dataset Selection and Challenges

### 3.1 Selected Axis and Topic

- **Axis 1 — Computer Vision**
- **Topic 1 — Face Recognition System** (LFW or equivalent)

### 3.2 Attempted Datasets and Failures

A significant portion of the project involved overcoming dataset-related obstacles:

| # | Dataset | Approach | Issue | Outcome |
|---|---------|----------|-------|---------|
| 1 | **LFW (full)** via `sklearn.datasets.fetch_lfw_people` | Full 62-class LFW | Extreme class imbalance; 62×47 pixel images caused CNN spatial dimension collapse; models predicted only the majority class | ❌ Failed |
| 2 | **LFW (7-person subset)** via `sklearn` (`min_faces_per_person=70`) | Balanced 7-class subset | Still heavily imbalanced: George W Bush ≈ 530 images, Hugo Chavez ≈ 71 images. Models collapsed to predicting Colin Powell (~41%). Class weights partially helped but model still memorised majority class | ❌ Failed |
| 3 | **CelebA** via `tensorflow_datasets` | Automatic download | Google Drive authentication error — `ValueError: Failed to obtain confirmation link for GDrive URL` | ❌ Failed |
| 4 | **LFW** via `tensorflow_datasets` (`tfds.load('lfw')`) | Original LFW via TFDS | Dataset is a **pair verification task**, not identity recognition — `KeyError: 'person'` | ❌ Failed |
| 5 | **Olivetti Faces** via `sklearn` | 40-class balanced dataset (filtered to 7) | Only 70 images for 7 classes (10 per class). ResNet50 achieved 14.29% test accuracy (chance level = 1/7). Insufficient data for deep learning | ❌ Failed |
| 6 | **Online image scraping** via `simple_image_download` | Google/Bing image search | Extremely slow download speeds; timeout errors; inconsistent results | ❌ Failed |
| 7 | **Custom football players dataset** (manual collection) | 5 players, manually downloaded high-quality facial images | Dataset successfully created! | ✅ **Success** |

### 3.3 Final Dataset

| Player | Number of Images |
|--------|-----------------|
| Lionel Messi | 70 |
| Cristiano Ronaldo | 41 |
| Kylian Mbappé | 33 |
| Erling Haaland | 69 |
| Neymar Jr | 49 |
| **Total** | **262** |

- **Image format**: RGB, 256×256 pixels after preprocessing.
- **Class balance**: Some imbalance present, mitigated through **class weighting**
  (using `sklearn.utils.class_weight`).
- **Split**: 183 training / 39 validation / 40 test (70% / 15% / 15%, stratified).

---

## 4. Exploratory Data Analysis (EDA)

EDA was conducted directly in the notebook (`FaceRecoFinal.ipynb`):

1. **Sample image display**: Random samples from each class were visualised to
   verify image quality, correct labelling, and visible facial features.
2. **Class distribution analysis**: Bar charts showed the number of images per
   player, confirming manageable imbalance.
3. **Data shape verification**: All images were resized to uniform 256×256×3,
   ensuring compatibility with neural network input layers.
4. **Normalisation check**: Pixel value ranges verified before and after
   preprocessing to confirm proper scaling.

---

## 5. Model Architectures

### 5.1 Baseline Model — MLP (Multi-Layer Perceptron)

```
Input: 128×128×3 = 49152 flattened pixels
───────────────────────────────────────
Dense(512, activation='relu')
Dropout(0.5)
Dense(256, activation='relu')
Dropout(0.5)
Dense(5, activation='softmax')
───────────────────────────────────────
Total params: ~25 million
```

This model serves as the **simplest baseline** — it has no spatial awareness and treats
each pixel independently. It demonstrates the fundamental difficulty of face
recognition without convolutional feature extraction.

### 5.2 Intermediate Model — Custom CNN

```
Input: 128×128×3
───────────────────────────────────────
Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast)
───────────────────────────────────────
Conv2D(32, 3×3) → BatchNorm → MaxPooling2D(2×2)
Conv2D(64, 3×3) → BatchNorm → MaxPooling2D(2×2)
Conv2D(128, 3×3) → BatchNorm → MaxPooling2D(2×2)
───────────────────────────────────────
GlobalAveragePooling2D  ← replaces Flatten (reduces parameters dramatically)
───────────────────────────────────────
Dropout(0.5)
Dense(128, activation='relu')
Dropout(0.5)
Dense(5, activation='softmax')
───────────────────────────────────────
Total params: ~4.3 million
```

**Key design decisions**:
- **GlobalAveragePooling2D** instead of `Flatten` reduced the first dense layer
  parameters from ~4.1 million to ~16,512 — critical for a small dataset.
- **Batch Normalization** after each convolution stabilises training.
- **Heavy data augmentation** compensates for the limited training samples.

### 5.3 Advanced Model — Transfer Learning with ResNet50

```
Input: 224×224×3
───────────────────────────────────────
Preprocessing: BGR conversion + mean subtraction
───────────────────────────────────────
ResNet50 (include_top=False, weights='imagenet', frozen)
───────────────────────────────────────
GlobalAveragePooling2D
Dropout(0.5)
Dense(128, activation='relu')
Dropout(0.5)
Dense(5, activation='softmax')
───────────────────────────────────────
Total params: ~24 million (only ~527k trainable)
```

**Why ResNet50?** ResNet50 was pre-trained on ImageNet (1.2 million images, 1000
classes). Its convolutional filters already encode general visual features — edges,
textures, shapes — that transfer directly to face recognition. By freezing the base and
training only the classification head, we benefit from this prior knowledge without
overfitting on our small dataset.

---

## 6. Training and Optimisation Study

### 6.1 Optimiser Comparison: Adam vs SGD

The intermediate CNN was trained twice — once with **Adam** (default learning rate
0.001) and once with **SGD** (learning rate 0.001, momentum 0.9, nesterov=True,
gradient clipping `clipnorm=1.0`).

| Optimiser | Final Validation Accuracy | Observations |
|-----------|--------------------------|--------------|
| **Adam** | ~25% (chance level) | Converged slowly; unable to escape random guessing due to insufficient data for training a CNN from scratch |
| **SGD + momentum** | ~25% (chance level) | Similar performance. Without gradient clipping, initial loss exploded to NaN (gradient overflow). After clipping, stabilised but still did not learn |

**Key insight**: Neither optimiser could overcome the fundamental data limitation
when training a CNN from scratch on ~183 training images across 5 classes. This result
**validates the necessity of transfer learning** for small datasets.

### 6.2 Advanced Model Training (ResNet50)

| Optimiser | Adam |
|-----------|------|
| Learning rate | 0.001 (with ReduceLROnPlateau) |
| Batch size | 8 |
| Epochs | 50 (Early Stopping after ~20 epochs) |
| Training accuracy | ~98% |
| **Validation accuracy** | **~82–86%** |

The ResNet50 converged rapidly, achieving meaningful validation accuracy within 10
epochs. The **ReduceLROnPlateau** callback reduced the learning rate when validation
loss plateaued, allowing finer weight updates in later epochs.

---

## 7. Regularisation Strategies

| Strategy | Where Applied | Rationale |
|----------|--------------|-----------|
| **Dropout (0.4–0.5)** | All three models (after dense layers) | Prevents co-adaptation of neurons; acts as implicit model averaging |
| **Early Stopping** | All models (patience=8–10, monitor=`val_loss`) | Halts training when validation performance degrades; prevents overfitting |
| **Data Augmentation** | Intermediate CNN (RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast) | Artificially expands the training set; forces the model to learn rotation/lighting-invariant features |
| **Batch Normalisation** | After every Conv2D layer in the CNN | Reduces internal covariate shift; acts as a mild regulariser |
| **Class Weighting** | All models (via `class_weight` parameter in `.fit()`) | Compensates for class imbalance (e.g., Mbappé has fewer images than Messi) |
| **Gradient Clipping** (`clipnorm=1.0`) | SGD training | Prevents exploding gradients with class weights |

---

## 8. Evaluation and Results

### 8.1 Model Performance Comparison

| Model | Training Accuracy | Validation Accuracy | Test Accuracy | Notes |
|-------|------------------|-------------------|---------------|-------|
| MLP Baseline | ~65–70% | ~55–60% | Moderate | Works better than CNN due to simpler hypothesis space |
| Custom CNN (Adam) | ~30% | ~25% | ~25% | Chance level; could not learn from limited data |
| Custom CNN (SGD) | ~25% | ~25% | ~25% | Identical behaviour; NaN loss without clipping |
| **ResNet50 (Transfer)** | **~98%** | **~82–86%** | **~80–85%** | **Best model — selected for deployment** |

### 8.2 Confusion Matrix (ResNet50 — Final Model)

The confusion matrix shows strong diagonal dominance, indicating that the ResNet50
correctly distinguishes between the 5 football players. Per-class metrics:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Lionel Messi | ~0.80 | ~0.85 | ~0.82 |
| Cristiano Ronaldo | ~0.80 | ~0.80 | ~0.80 |
| Kylian Mbappé | ~0.75 | ~0.75 | ~0.75 |
| Erling Haaland | ~0.85 | ~0.85 | ~0.85 |
| Neymar Jr | ~0.85 | ~0.80 | ~0.82 |

### 8.3 Learning Curves

Learning curves (training vs validation loss/accuracy over epochs) were generated
for all three models:

- **MLP**: Gradual convergence, moderate gap between training and validation
  (mild overfitting).
- **Custom CNN (Adam)**: Training accuracy fluctuated around 30%, validation
  accuracy flat at ~25%. Both loss curves remained high — classic
  **underfitting**.
- **ResNet50**: Training accuracy approached 98% within 20 epochs. Validation
  accuracy rose to ~82–86% then plateaued. Early Stopping triggered after
  ~20 epochs. The small gap between training and validation indicates **good
  generalisation**.

---

## 9. Error Analysis

### 9.1 Misclassification Patterns

Misclassified test samples were visually inspected. Common patterns include:

- **Mbappé ↔ Haaland confusion**: Both are young, light-skinned players; some
  images had similar poses or lighting.
- **Messi ↔ Ronaldo confusion**: Occasional misclassification when images were
  cropped tightly to the face (removing contextual hair/clothing cues).
- **Low-quality images**: Blurry or poorly lit images were more likely to be
  misclassified.

### 9.2 Limitations

1. **Dataset size**: 262 images is small by deep learning standards. Each class
   has only 33–70 images, limiting the model's ability to learn fine-grained
   facial features.
2. **Image diversity**: Most images are from similar contexts (match photos,
   press conferences). The model may struggle with casual/unposed photos.
3. **No face detection**: The current pipeline relies on pre-cropped face images.
   Real-world deployment would benefit from a face detection pre-processing step
   (e.g., MTCNN, Haar Cascades).

---

## 10. Deployment

### 10.1 Platform: Hugging Face Spaces (Docker SDK)

The best model (ResNet50) was deployed as an interactive web application:

| Component | Technology |
|-----------|-----------|
| **Frontend** | Gradio (image upload + label output) |
| **Backend** | Python 3.11, TensorFlow/Keras |
| **Containerisation** | Dockerfile (custom Python 3.11-slim image) |
| **Hosting** | Hugging Face Spaces (free CPU tier) |

### 10.2 Deployment Journey

The deployment went through multiple iterations:

| Attempt | Approach | Issue | Outcome |
|---------|----------|-------|---------|
| 1 | TFLite + `tflite-runtime` on Gradio SDK | `tflite-runtime` has no wheel for Python 3.13 (Hugging Face default) | ❌ Build failed |
| 2 | TFLite + Docker SDK | `tflite-runtime` package unavailable | ❌ Build failed |
| 3 | Full TensorFlow + Python 3.11-slim Docker image | Successfully built and deployed | ✅ **Success** |

**Final Dockerfile**:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**Final `requirements.txt`**:

```
tensorflow==2.19.0
gradio==6.0.0
pillow
numpy
```

### 10.3 Live Application

The deployed application presents a simple interface:
1. User uploads a photo of a football player.
2. The image is preprocessed (resized to 224×224, BGR conversion, mean subtraction).
3. The ResNet50 model predicts the identity.
4. The result is displayed as a ranked probability distribution across all 5 classes.

**Live URL**: [https://huggingface.co/spaces/tahazeroug/college-Deeplearning-FaceReco](https://huggingface.co/spaces/tahazeroug/college-Deeplearning-FaceReco)

---

## 11. Discussion

### 11.1 Why the Custom CNN Failed

Training a CNN from scratch on ~183 images (across 5 classes) is fundamentally
challenging. CNNs learn hierarchical features — edges in early layers, textures in
middle layers, and object parts in deeper layers. Without sufficient data, the
network never develops meaningful filters and converges to random guessing (~20–25%
accuracy for 5 classes).

This is **not a code error** but a well-documented phenomenon in deep learning:
CNNs typically require **thousands of labeled examples** to learn discriminative
features from scratch. Our experiment **empirically confirms** this limitation
and provides a strong justification for transfer learning.

### 11.2 Why Transfer Learning Succeeded

ResNet50 was pre-trained on ImageNet (1.2M images, 1000 classes). Its convolutional
filters already recognise general-purpose visual patterns — edges, corners, facial
shapes, skin textures — that are directly usable for face recognition. By freezing
these pre-trained weights and training only a small classification head (~527k
parameters), the model:

1. **Avoids overfitting**: Only a fraction of total weights are updated.
2. **Benefits from prior knowledge**: ImageNet features generalise to face
   recognition.
3. **Requires minimal data**: The ~183 training images are sufficient to learn
   classification boundaries in the pre-trained feature space.

### 11.3 Lessons Learned

1. **Dataset quality > dataset quantity**: Small, balanced, high-quality datasets
   outperform large, imbalanced ones for face recognition.
2. **Class weighting is essential** but cannot fully compensate for extreme
   imbalance (as seen with the LFW 7-person subset).
3. **Image resolution matters**: 50×37 LFW images were too small for meaningful
   CNN feature extraction; 256×256 images worked well.
4. **Transfer learning is the practical default** for small-data computer vision
   tasks.
5. **Deployment platforms have hidden constraints**: Python version compatibility
   with `tflite-runtime` caused deployment failures; switching to full TensorFlow
   on Python 3.11 resolved the issue.

---

## 12. Conclusion

This project successfully delivered a **complete deep learning pipeline** for face
recognition:

- **Three models** implemented and compared (MLP, Custom CNN, Transfer Learning).
- **Optimiser comparison** (Adam vs SGD) conducted with documented results.
- **Regularisation strategies** applied (Dropout, Early Stopping, Data
  Augmentation, BatchNorm, Class Weighting, Gradient Clipping).
- **Comprehensive evaluation** performed (accuracy, confusion matrices, learning
  curves, classification reports, error analysis).
- **Best model deployed** on Hugging Face Spaces via Docker SDK and Gradio.

The **ResNet50 transfer learning model** achieved **~82–86% validation accuracy**
on a custom dataset of 262 face images across 5 football players — demonstrating
that transfer learning is the most effective approach for small-scale face
recognition tasks.

---

## 13. How to Run

### 13.1 Training (Google Colab)

1. Open `FaceRecoFinal.ipynb` in Google Colab with GPU runtime enabled:
   - Runtime → Change runtime type → **T4 GPU**.
2. Mount Google Drive and upload the `players/` folder containing subfolders for
   each football player (Messi, Ronaldo, Mbappé, Haaland, Neymar).
3. Run all cells sequentially:
   - Cells 1–5: Imports, data loading, train/val/test split, preprocessing.
   - Cells 6–8: MLP baseline definition and training.
   - Cells 9–12: Custom CNN definition, Adam training, SGD training, optimiser
     comparison plots.
   - Cells 13–15: ResNet50 transfer learning, evaluation, model saving.
4. Download `model.keras` and `target_names.json` for deployment.

### 13.2 Deployment (Hugging Face Spaces)

1. Create a new Space on Hugging Face with **Docker SDK**:
   - [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose SDK: **Docker**
   - Hardware: **CPU Basic** (free)
2. Clone the Space repository and add files:
   ```bash
   git clone https://huggingface.co/spaces/your-username/your-space-name
   cp app.py requirements.txt Dockerfile model.keras your-space-name/
   cd your-space-name
   git add .
   git commit -m "Deploy face recognition model"
   git push
   ```
3. The Space will automatically build and deploy. Access the live app at:
   ```
   https://huggingface.co/spaces/your-username/your-space-name
   ```

### 13.3 Local Deployment (Optional)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860 in your browser
```

---

## Academic Integrity

- All models were implemented independently using TensorFlow/Keras.
- The ResNet50 pre-trained weights (ImageNet) are publicly available and properly
  cited: *He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for
  Image Recognition. CVPR 2016.*
- The football player dataset was collected from publicly accessible online sources
  for academic purposes.

---

## Authors

**Taha ZEROUG**, **Ishak ALILI**, **Ayoub SABKHAOUI** — Deep Learning Laboratory Project

---

## License

This project is submitted as academic coursework. All third-party components
(TensorFlow, ResNet50 weights, Gradio) retain their original licenses.

---