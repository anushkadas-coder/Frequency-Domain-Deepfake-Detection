# Isolating Synthetic Fingerprints: A Frequency-Domain Approach to Robust Deepfake Detection

[cite_start]A lightweight deepfake detection engine that isolates synthetic fingerprints in the frequency domain using a 2D Fast Fourier Transform (FFT) and a ResNet-18 backbone[cite: 2, 6, 8].

[cite_start]As generative artificial intelligence produces increasingly realistic synthetic media, traditional, spatial-domain deepfake detectors have significantly diminished in efficacy[cite: 4]. [cite_start]Modern deepfakes are highly resilient to standard detection when subjected to compression or blurring[cite: 5]. [cite_start]This project shifts the detection paradigm from the spatial to the frequency domain to expose the inherent, grid-like synthetic fingerprints left behind by generative upsampling processes[cite: 19, 21].

## System Architecture

[cite_start]The system comprises three primary modules[cite: 31]:

### 1. Micro-Blur Preprocessing Pipeline
[cite_start]To counter the susceptibility of spatial-only detectors to image degradation, a micro-blur preprocessing step is applied prior to spectral analysis[cite: 33, 34]. [cite_start]By applying a controlled Gaussian kernel ($\sigma=1.5$), high-frequency spatial noise is intentionally suppressed[cite: 35, 69]. [cite_start]This step forces the subsequent spectral extraction to isolate the deep, structural synthetic fingerprints rather than superficial pixel-level noise[cite: 40].

### 2. Frequency-Domain Transformation via 2D FFT
[cite_start]Generative models often leave distinct, grid-like artifacts in the high-frequency spectrum during the upsampling process[cite: 50]. [cite_start]To expose these artifacts, the preprocessed image is transformed using the 2D Discrete Fourier Transform (DFT)[cite: 51]:

[cite_start]$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}$$ [cite: 52]

[cite_start]The resulting magnitude spectrum is shifted to center the zero-frequency component, providing a structured spectral map[cite: 55].

### 3. Feature Extraction and Classification
[cite_start]The preprocessed spectral magnitude maps are fed into a ResNet-18 architecture[cite: 57]. [cite_start]The residual connections prevent vanishing gradients while effectively capturing the hierarchical patterns present in the frequency spectra[cite: 59]. [cite_start]The network is optimized using the Binary Cross-Entropy (BCE) loss function[cite: 61]:

[cite_start]$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}[y_{i}log(\hat{y}_{i})+(1-y_{i})log(1-\hat{y}_{i})]$$ [cite: 63]

## Experimental Results

[cite_start]Experiments were conducted on two highly benchmarked datasets: FaceForensics++ (FF++) and Celeb-DF v2[cite: 66]. 

| Dataset | Acc. | Prec. | Rec. | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **FF++ (Raw)** | 99.12% | 98.90% | 99.21% | 99.05% |
| **Celeb-DF v2** | 98.45% | 97.82% | 98.10% | 97.96% |

[cite_start]*Source: Classification Performance on Benchmark Datasets[cite: 73, 74].*

[cite_start]**Inference Speed:** The reliance on ResNet-18 drastically reduced computational overhead compared to Vision Transformer (ViT) baselines, achieving 45 frames per second (FPS) on a standard NVIDIA RTX 3060 GPU[cite: 75].

## Ablation Study
[cite_start]An ablation study conducted on the Celeb-DF v2 dataset validated the efficacy of the micro-blur module[cite: 77]. [cite_start]Removing the micro-blur pipeline resulted in a performance degradation, dropping the overall accuracy from 98.45% to 94.12%[cite: 79]. [cite_start]Without the suppression of high-frequency spatial noise, the ResNet-18 backbone prematurely overfit to superficial, pixel-level artifacts[cite: 80].

## Implementation Details
* [cite_start]**Framework:** PyTorch [cite: 69]
* [cite_start]**Optimizer:** Adam (learning rate $1\times10^{-4}$) [cite: 69]
* [cite_start]**Batch Size:** 32 [cite: 69]
* [cite_start]**Epochs:** 50 [cite: 69]

## Citation
If you utilize this research, please consider citing:
```bibtex
@article{das2026deepfake,
  title={Isolating Synthetic Fingerprints: A Frequency-Domain Approach to Robust Deepfake Detection using 2D FFT and Lightweight CNNs},
  author={Das, Anushka},
  year={2026}
}
