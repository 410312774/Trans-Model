# Trans-Model: Longitudinal MRI + Pathomics for LF Prediction after SRT

Attention-based fusion for lesion-level local failure (LF) prediction after stereotactic radiotherapy (SRT) for brain metastases.

## What it does
- Inputs (PCA to 32-dim each):
  - Radiomics (pre/post multiparametric MRI)
  - Deep features ViT (pre/post MRI)
  - Pathomics (primary-tumor WSI)
- Fusion: Self-Attention (intra-modal) + Cross-Attention (inter-modal) + LN + FFN + Dropout 0.3
- Output: P(LF) via FC + sigmoid, trained with class-weighted BCE (AdamW 1e-4)

### Implemented on the Pixelmed AI platform
