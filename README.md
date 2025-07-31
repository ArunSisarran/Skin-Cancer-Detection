# SkinGuard AI 🔬

> **⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. It is not intended to diagnose, treat, cure, or prevent any disease. Always consult with a qualified dermatologist or healthcare provider for any skin concerns.

An AI-powered web application for skin cancer detection using deep learning. SkinGuard AI assists in the early identification of melanoma versus non-melanoma skin lesions through advanced computer vision techniques.


## 🚀 Features

- **Binary Classification**: Melanoma vs. Non-melanoma detection with high sensitivity
- **Real-time Analysis**: Fast predictions with confidence scores
- **Visual Explanations**: Grad-CAM heatmaps showing decision regions
- **User-Friendly Interface**: Intuitive drag-and-drop image upload
- **Medical Compliance**: Comprehensive disclaimers and safety messaging
- **REST API**: FastAPI backend for integration with other systems
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture

```
┌──────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│   React Frontend │────│  FastAPI Backend  │────│  PyTorch Model     │
│                  │    │                   │    │                    │
│ • Image Upload   │    │ • Image Processing│    │ • EfficientNet-B0  │
│ • Results Display│    │ • Prediction API  │    │ • Transfer Learning│
│ • Grad-CAM Viz   │    │ • Grad-CAM Gen    │    │ • Binary Classifier│
└──────────────────┘    └───────────────────┘    └────────────────────┘
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Sensitivity (Melanoma)** | 87.3% |
| **Specificity (Non-melanoma)** | 84.1% |
| **Overall Accuracy** | 85.2% |
| **F1-Score** | 0.83 |
| **AUC-ROC** | 0.91 |

*Evaluated on HAM10000 test set with 2,000 images*

## 🛠️ Tech Stack

**Backend:** 
- PyTorch & Torchvision for deep learning
- FastAPI for REST API
- PIL for image processing
- Scikit-learn for metrics

**Frontend:**
- React 18 with modern hooks
- Lucide React for icons
- CSS3 with responsive design
- Drag-and-drop file upload

**Model:**
- EfficientNet-B0 backbone
- Transfer learning from ImageNet
- Custom binary classifier head
- Data augmentation pipeline
