# Multimodal ALK Prediction Framework
## Official implementation for predicting ALK rearrangements in lung adenocarcinoma (LUAD) by bridging the macro-to-micro gap. This framework integrates Pathology Foundation Models (PFMs) and 3D Radiology Backbones using a WSI-led cross-attention mechanism.

1. 🌟 Key Highlights
   - 100% Sensitivity: Correctly identified all ALK-positive cases in a real-world imbalanced cohort (n=254).Publication-Ready Analytics.
   - Macro-to-Micro Integration: Selectively queries CT features using PFM-derived pathology signatures as the primary guide.
   - Automated 600 DPI visualizations for SHAP attribution and CKA alignment analysis.
2. 🛠️ Quick Start
   - 1. Installation
    pip install -r requirements.txt
   - 2. Configuration
    Anonymize your environment by updating local paths in configs/configure.yaml.
   - 3. Usage
    To Train:
    python src/train.py --config configs/configure.yaml --strategy cross_attention
   - 4. To Analyze Interpretability:
    python src/interpretability.py --config configs/configure.yaml
3. 🙏 Acknowledgements
   - 1. Radiology
    MedicalNet: 3D ResNet-18 pretrained weights for medical image analysis.
   - 2. Pathology
    Virchow2: State-of-the-art pathology foundation model.
    Prov-GigaPath: Whole-slide pathology foundation model.
    UNI2: Competitive pathology foundation model.
    CONCH v1.5: Vision-language foundation model for pathology.
    CTransPath: Swin-Transformer based pathology encoder.
    ResNet-50: Standard baseline histology encoder.
   - 3. Frameworks
    CLAM: Constrained-Attention Multiple-Instance Learning for WSI classification.
