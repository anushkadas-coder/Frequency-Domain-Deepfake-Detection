# Frequency-Domain Deepfake Detection

A lightweight, robust deepfake detection engine that isolates synthetic fingerprints in the frequency domain using 2D Fast Fourier Transforms (FFT) and a ResNet-18 backbone.

As generative artificial intelligence produces increasingly realistic synthetic media, traditional spatial-domain detectors have become susceptible to image degradation like compression and blurring. This project shifts the detection paradigm to the frequency domain to expose the inherent, grid-like synthetic fingerprints left behind by generative upsampling processes.

## System Architecture

The pipeline comprises three primary modules:

### 1. Micro-Blur Preprocessing
To counter the vulnerability of spatial detectors to image degradation, the pipeline applies a controlled Gaussian kernel ($\sigma=1.5$). This intentionally suppresses superficial high-frequency spatial noise, forcing the network to learn robust spectral representations. 

The Gaussian kernel is defined as:
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 2. Frequency-Domain Transformation (2D FFT)
Generative models often leave distinct artifacts in the high-frequency spectrum during synthesis. The preprocessed image $f(x,y)$ of size $M\times N$ is transformed using a 2D Discrete Fourier Transform (DFT) to expose these anomalies:

$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$

The magnitude spectrum is shifted to center the zero-frequency component, yielding a structured spectral map where synthetic grid artifacts become highly visible.

### 3. Feature Extraction & Classification
The spectral magnitude maps are fed into a lightweight ResNet-18 architecture, effectively capturing hierarchical frequency patterns without the immense computational complexity of massive Vision Transformers (ViTs). The network is optimized using Binary Cross-Entropy (BCE) loss:

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}[y_{i}\log(\hat{y}_{i})+(1-y_{i})\log(1-\hat{y}_{i})]$$

## Performance & Benchmarks

The model was evaluated on two highly benchmarked datasets: FaceForensics++ (FF++) and Celeb-DF v2.

| Dataset | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **FF++ (Raw)** | 99.12% | 98.90% | 99.21% | 99.05% |
| **Celeb-DF v2** | 98.45% | 97.82% | 98.10% | 97.96% |

*Source: Classification Performance on Benchmark Datasets.*

**Inference Speed:** By utilizing ResNet-18 instead of heavier models, the system achieves real-time inference speeds of **45 Frames Per Second (FPS)** on a standard NVIDIA RTX 3060 GPU, making it highly viable for edge-device media forensics.

## Ablation Study Results
An ablation study on the Celeb-DF v2 dataset confirmed the necessity of the micro-blur pipeline. Removing this preprocessing step dropped overall accuracy from 98.45% to 94.12%, as the network prematurely overfit to pixel-level noise rather than the structural frequency anomalies.

## Implementation Details
* **Framework:** PyTorch
* **Optimizer:** Adam (Learning Rate: $1\times 10^{-4}$)
* **Batch Size:** 32
* **Epochs:** 50
