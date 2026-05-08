# Frequency-Domain Deepfake Detection

[cite_start]A lightweight, robust deepfake detection engine that isolates synthetic fingerprints in the frequency domain using 2D Fast Fourier Transforms (FFT) and a ResNet-18 backbone[cite: 1, 2, 8].

[cite_start]As generative artificial intelligence produces increasingly realistic synthetic media, traditional spatial-domain detectors have become susceptible to image degradation like compression and blurring[cite: 4, 5, 17]. [cite_start]This project shifts the detection paradigm to the frequency domain to expose the inherent, grid-like synthetic fingerprints left behind by generative upsampling processes[cite: 19, 21].

## System Architecture

[cite_start]The pipeline comprises three primary modules[cite: 31]:

### 1. Micro-Blur Preprocessing
[cite_start]To counter the vulnerability of spatial detectors to image degradation, the pipeline applies a controlled Gaussian kernel ($\sigma=1.5$)[cite: 33, 69]. [cite_start]This intentionally suppresses superficial high-frequency spatial noise, forcing the network to learn robust spectral representations[cite: 35, 81]. 

[cite_start]The Gaussian kernel is defined as[cite: 35]:
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 2. Frequency-Domain Transformation (2D FFT)
[cite_start]Generative models often leave distinct artifacts in the high-frequency spectrum during synthesis[cite: 50]. [cite_start]The preprocessed image $f(x,y)$ of size $M\times N$ is transformed using a 2D Discrete Fourier Transform (DFT) to expose these anomalies[cite: 51]:

$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$

[cite_start]The magnitude spectrum is shifted to center the zero-frequency component, yielding a structured spectral map where synthetic grid artifacts become highly visible[cite: 55].

### 3. Feature Extraction & Classification
[cite_start]The spectral magnitude maps are fed into a lightweight ResNet-18 architecture, effectively capturing hierarchical frequency patterns without the immense computational complexity of massive Vision Transformers (ViTs)[cite: 16, 57, 58, 59]. [cite_start]The network is optimized using Binary Cross-Entropy (BCE) loss[cite: 61]:

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}[y_{i}\log(\hat{y}_{i})+(1-y_{i})\log(1-\hat{y}_{i})]$$

## Performance & Benchmarks

[cite_start]The model was evaluated on two highly benchmarked datasets: FaceForensics++ (FF++) and Celeb-DF v2[cite: 66].

| Dataset | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **FF++ (Raw)** | 99.12% | 98.90% | 99.21% | 99.05% |
| **Celeb-DF v2** | 98.45% | 97.82% | 98.10% | 97.96% |

[cite_start]*Source: Classification Performance on Benchmark Datasets[cite: 73, 74].*

[cite_start]**Inference Speed:** By utilizing ResNet-18 instead of heavier models, the system achieves real-time inference speeds of **45 Frames Per Second (FPS)** on a standard NVIDIA RTX 3060 GPU, making it highly viable for edge-device media forensics[cite: 10, 75, 91].

## Ablation Study Results
[cite_start]An ablation study on the Celeb-DF v2 dataset confirmed the necessity of the micro-blur pipeline[cite: 77]. [cite_start]Removing this preprocessing step dropped overall accuracy from 98.45% to 94.12%, as the network prematurely overfit to pixel-level noise rather than the structural frequency anomalies[cite: 79, 80].

## Implementation Details
* [cite_start]**Framework:** PyTorch [cite: 69]
* [cite_start]**Optimizer:** Adam (Learning Rate: $1\times 10^{-4}$) [cite: 69]
* [cite_start]**Batch Size:** 32 [cite: 69]
* [cite_start]**Epochs:** 50 [cite: 69]
