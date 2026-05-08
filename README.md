# Isolating Synthetic Fingerprints: A Frequency-Domain Approach to Robust Deepfake Detection using 2D FFT and Lightweight CNNs

This repository contains the official implementation for the paper **"Isolating Synthetic Fingerprints."** As generative artificial intelligence continues to produce increasingly realistic synthetic media, the efficacy of traditional, spatial-domain deepfake detectors has significantly diminished. Modern deepfakes are highly resilient to standard detection when subjected to compression or blurring. This project presents a novel, lightweight deepfake detection engine that shifts the detection paradigm to the frequency domain to isolate synthetic fingerprints.

## System Architecture

The proposed pipeline aims to distinguish pristine media from synthetically generated deepfakes by analyzing artifacts in the frequency domain. The system comprises three primary modules:

### 1. Micro-Blur Preprocessing Pipeline
A critical vulnerability of spatial-only deepfake detectors is their susceptibility to image degradation. To counter this, a micro-blur preprocessing step is applied prior to spectral analysis. By applying a controlled Gaussian kernel ($\sigma=1.5$), high-frequency spatial noise is intentionally suppressed, forcing the subsequent spectral extraction to isolate the deep, structural synthetic fingerprints rather than superficial pixel-level noise.

The Gaussian kernel is defined as:
$$G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 2. Frequency-Domain Transformation via 2D FFT
Generative models often leave distinct, grid-like artifacts in the high-frequency spectrum during the upsampling process. To expose these artifacts, the preprocessed image $f(x,y)$ of size $M \times N$ is transformed using the 2D Discrete Fourier Transform (DFT):

$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$

The resulting magnitude spectrum is shifted to center the zero-frequency component, providing a structured spectral map.

### 3. Feature Extraction and Classification
The preprocessed spectral magnitude maps are fed into a ResNet-18 architecture. ResNet-18 was strategically selected to maintain computational efficiency while the residual connections prevent vanishing gradients. To train the classifier, we optimize the network using the Binary Cross-Entropy (BCE) loss function:

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}[y_{i}\log(\hat{y}_{i})+(1-y_{i})\log(1-\hat{y}_{i})]$$

## Experimental Results

The system was implemented using PyTorch and evaluated on two highly benchmarked datasets: FaceForensics++ (FF++) and Celeb-DF v2. 

| Dataset | Acc. | Prec. | Rec. | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **FF++ (Raw)** | 99.12% | 98.90% | 99.21% | 99.05% |
| **Celeb-DF v2** | 98.45% | 97.82% | 98.10% | 97.96% |

**Inference Speed:** The reliance on ResNet-18 drastically reduced computational overhead compared to Vision Transformer (ViT) baselines, achieving **45 frames per second (FPS)** on a standard NVIDIA RTX 3060 GPU.

## Ablation Study
An ablation study on the Celeb-DF v2 dataset confirmed that removing the micro-blur pipeline dropped the overall accuracy from 98.45% to 94.12%. Without the suppression of high-frequency spatial noise, the ResNet-18 backbone prematurely overfit to superficial, pixel-level artifacts, losing its ability to generalize.

## Implementation Details
* **Framework:** PyTorch
* **Backbone Weights:** ImageNet Initialized
* **Optimizer:** Adam (Learning Rate: $1\times 10^{-4}$)
* **Batch Size:** 32
* **Epochs:** 50

## Citation
If you find this research helpful, please consider citing:
```bibtex
@article{das2026deepfake,
  title={Isolating Synthetic Fingerprints: A Frequency-Domain Approach to Robust Deepfake Detection using 2D FFT and Lightweight CNNs},
  author={Das, Anushka},
  year={2026}
}
