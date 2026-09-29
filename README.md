# AgriGrade AI: Multi-Spectral & Class-Conditioned Quality Grading System

**Project Title:** AgriGrade AI — Context-Aware Produce Quality Grading via Dual-Stream Feature Fusion and Random Forest Inference

**Domain:** AgriTech & Smart Farming

**Problem Statement ID:** AGR-S05

## 1. Executive Summary

Manual quality grading of agricultural produce is inherently subjective, prone to human error, and inconsistent across supply chain nodes. Standard computer vision approaches rely on static color averages or basic CNN classifications, which fail under variable lighting or when handling visually non-standard produce (e.g., brown fruits such as Kiwis or Chikoos).

**AgriGrade AI** resolves these limitations by introducing a **Class-Conditioned Hybrid Pipeline**:

1. **Produce Classification & Segmentation:** Automatically identifies the produce type and isolates the target region from background noise using YOLOv8-Segmentation.

2. **Dynamic Spectral & Texture Feature Extraction:** Adapts mathematical feature formulas (Visible Spectral Indices, GLCM Texture Metrics, and Color Space Analysis) based on the specific produce class.

3. **Calibrated Physical Metric Calculation:** Computes true size and volume using on-screen standard reference calibration.

4. **Random Forest Quality Grading:** Combines physical, spectral, and textural metrics into a Random Forest classifier that outputs standard market quality grades (Grade A, Grade B, Grade C, Reject) alongside a calibrated confidence score and XAI (Explainable AI) feature importance metrics.

## 2. System Architecture

```
                                  [ Smartphone Camera Input ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  YOLOv8 Instance Segmentation    │
                              └──────────────────────────────────┘
                                        │               │
                                        │ Foreground    │ Produce Class
                                        │ Mask          │ (e.g., "Kiwi", "Apple")
                                        ▼               ▼
                              ┌──────────────────────────────────┐
                              │  Dynamic Feature Router          │
                              └──────────────────────────────────┘
                                               │
       ┌───────────────────────────────────────┼───────────────────────────────────────┐
       ▼                                       ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
│ Red/Green Fruits (Apple/Tomato)│   │ Yellow/Green (Banana/Mango)   │   │ Brown/Textured (Kiwi/Chikoo)  │
├───────────────────────────────┤   ├───────────────────────────────┤   ├───────────────────────────────┤
│ • NDTI (Red-Green Index)      │   │ • YI (Yellowness Index)       │   │ • ExB (Excess Blue / Mold)    │
│ • VARI (Vegetation Index)     │   │ • VARI (Chlorophyll Decay)    │   │ • L* Lightness Variance (Rot) │
│ • L*a*b* Color Distribution   │   │ • Spotting Contour Ratio      │   │ • GLCM & LBP Texture Analysis │
└───────────────────────────────┘   └───────────────────────────────┘   └───────────────────────────────┘
       │                                       │                                       │
       └───────────────────────────────────────┼───────────────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Geometric Metric Calibration     │
                              │ • ArUco / Coin Reference Scaling │
                              │ • Equivalent Diameter & Area     │
                              └──────────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Feature Vector Concatenation     │
                              └──────────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Random Forest Classifier         │
                              └──────────────────────────────────┘
                                        │               │
                                        ▼               ▼
                              ┌──────────────────┐   ┌──────────────────┐
                              │ Final Quality    │   │ Confidence Score │
                              │ Grade (A / B / C)│   │ & SHAP XAI Map   │
                              └──────────────────┘   └──────────────────┘

```

## 3. Dynamic Feature Extraction Engine

### A. Visible-Spectrum Spectral Index Math

Instead of simple RGB channel averages, AgriGrade AI uses non-linear band ratios to isolate chlorophyll density, carotenoid shifts, and fungal decay while eliminating global illumination glare.

1. **VARI (Visible Automatically Resistant Index):**
   

   $$
   \text{VARI} = \frac{G - R}{G + R - B + \epsilon}
   $$

   
   *Measures chlorophyll density and photosynthetic activity.*

2. **NDTI (Normalized Difference Red-Green Index):**
   

   $$
   \text{NDTI} = \frac{R - G}{R + G + \epsilon}
   $$

   
   *Quantifies anthocyanin and lycopene accumulation during fruit ripening.*

3. **YI (Yellowness Index):**
   

   $$
   \text{YI} = \frac{R + G - 2B}{R + G + B + \epsilon}
   $$

   
   *Tracks carotenoid progression in bananas, mangoes, and papayas.*

4. **ExB (Excess Blue Index - Mold & Fungal Detector):**
   

   $$
   \text{ExB} = 2B - R - G
   $$

   
   *Isolates grey/white fungal growth and surface mold on dark or brown produce skin.*

### B. Class-Conditioned Feature Routing Strategy

Different fruits require distinct feature extraction strategies to eliminate false positive defect detections:

| **Produce Category** | **Examples** | **Primary Spectral Metric** | **Secondary Texture / Luminance Metric** | **Defect Signature** | 
| **Red / Smooth** | Apple, Tomato, Red Grape | NDTI & VARI | $a^*$ channel ($L^*a^*b^*$) variance | Dark spots drop NDTI below produce baseline. | 
| **Yellow / Green** | Banana, Mango, Citrus | YI & VARI | Black spot contour area ratio | High YI indicates ripeness; low VARI indicates age. | 
| **Brown / Textured** | Kiwi, Chikoo, Brown Pear | ExB (Excess Blue) | GLCM Homogeneity & $L^*$ Variance | Soft rot shows high $L^*$ variance; mold spikes ExB. | 

### C. Physical Size & Volume Calibration

To achieve camera-distance-invariant size measurement:

1. An **on-screen reference marker** (e.g., standard coin or ArUco tag) is placed in the frame.

2. Calibration scale calculation:
   

   $$
   \text{Scale Ratio (mm/pixel)} = \frac{\text{Known Reference Diameter (mm)}}{\text{Reference Bounding Diameter (pixels)}}
   $$

3. True produce surface area calculation:
   

   $$
   \text{True Area } (\text{cm}^2) = \text{Pixel Area} \times (\text{Scale Ratio})^2 \times 10^{-2}
   $$

## 4. Random Forest Quality Classifier & Confidence Calculation

### A. Prediction & Confidence Math

The extracted feature vector $X = [\text{Spectral Metrics}, \text{Texture Metrics}, \text{Size Area}, \text{Color Statistics}]$ is passed into a trained **Random Forest Ensemble** consisting of $N = 100$ decision trees.

* **Class Prediction:**
  

  $$
  \hat{y} = \arg\max_{c \in C} \left( \frac{1}{N} \sum_{i=1}^{N} P_i(y = c \mid X) \right)
  $$

* **Confidence Score (%):**
  

  $$
  \text{Confidence} = \max_{c \in C} \left( \frac{1}{N} \sum_{i=1}^{N} P_i(y = c \mid X) \right) \times 100
  $$

Where $C = \{\text{Grade A}, \text{Grade B}, \text{Grade C}, \text{Reject}\}$.

### B. Machine Learning Pipeline Implementation (Python)

```
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_produce_features(image_rgb, mask, crop_type, pixel_to_mm_ratio):
    """
    Extracts dynamic features based on crop_type.
    """
    img = image_rgb.astype(np.float32) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    eps = 1e-6

    # Foreground pixels
    fg_R, fg_G, fg_B = R[mask > 0], G[mask > 0], B[mask > 0]
    
    # Base Physical Features
    total_pixels = np.sum(mask > 0)
    true_area_cm2 = (total_pixels * (pixel_to_mm_ratio ** 2)) / 100.0

    features = {"true_area_cm2": true_area_cm2}

    # Dynamic Routing
    if crop_type in ["apple", "tomato"]:
        ndti = (fg_R - fg_G) / (fg_R + fg_G + eps)
        vari = (fg_G - fg_R) / (fg_G + fg_R - fg_B + eps)
        features["spectral_primary"] = float(np.mean(ndti))
        features["spectral_secondary"] = float(np.mean(vari))
        features["defect_index"] = float(np.std(ndti))

    elif crop_type in ["banana", "mango"]:
        yi = (fg_R + fg_G - (2 * fg_B)) / (fg_R + fg_G + fg_B + eps)
        vari = (fg_G - fg_R) / (fg_G + fg_R - fg_B + eps)
        features["spectral_primary"] = float(np.mean(yi))
        features["spectral_secondary"] = float(np.mean(vari))
        features["defect_index"] = float(np.sum(yi < 0.1) / len(yi))

    elif crop_type in ["kiwi", "chikoo"]:
        exb = (2 * fg_B) - fg_R - fg_G
        lab_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        L_channel = lab_img[:,:,0][mask > 0]
        features["spectral_primary"] = float(np.mean(exb))        # Mold
        features["spectral_secondary"] = float(np.mean(L_channel)) # Lightness
        features["defect_index"] = float(np.std(L_channel))       # Soft rot variance

    return features

# Random Forest Classifier Sample
def classify_produce(feature_vector, trained_rf_model):
    probabilities = trained_rf_model.predict_proba([feature_vector])[0]
    classes = trained_rf_model.classes_
    
    best_class_idx = np.argmax(probabilities)
    predicted_grade = classes[best_class_idx]
    confidence = probabilities[best_class_idx] * 100

    return predicted_grade, confidence

```

## Repository Layout

This spec is implemented across three parallel branches — `feature-extraction`,
`model` and `frontend` — cut from `main`, which owns the shared contracts in
`src/agrigrade/core/`. No branch imports another branch's internals.

```
src/agrigrade/core/          shared contracts (produce, family, grade vocabularies)
src/agrigrade/features/      spectral, texture, geometry extraction   -> feature-extraction
src/agrigrade/model/         Random Forest, confidence, XAI           -> model
src/agrigrade/segmentation/  YOLOv8-seg adapter                       -> main
src/agrigrade/api/           FastAPI surface                          -> main
frontend/                    React + TS + Vite client                 -> frontend
docs/ARCHITECTURE.md         branch map, boundaries, data flow
CONTRIBUTING.md              ownership table and PR rules
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full map and
[CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow.

## 5. Key Innovations & Hackathon Differentiation

1. **Illumination-Invariant Spectral Math:** Operates reliably under varying room lights, eliminating the need for expensive darkroom boxes or external ring lights.

2. **Context-Aware Class Routing:** Eliminates common computer vision bugs where brown produce (e.g., Kiwis) are incorrectly flagged as rotten apples.

3. **On-Device Edge Deployment:** Runs feature extraction via OpenCV and inference via TFLite/ONNX Runtime on mobile devices without relying on high-latency cloud servers.

4. **Transparent Explainable AI (XAI):** Provides farmers and buyers with feature breakdown plots explaining *why* a particular batch received a specific grade.