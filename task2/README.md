# 🛰️🌌 Sentinel-2 image matching. LoFTR. Detector-Free Local Feature Matching with Transformers

<div>
    <img src="https://images.unsplash.com/photo-1526666923127-b2970f64b422?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1172" height="500" alt="space" />
</div>

### This project provides a comprehensive guide and implementation for matching Sentinel-2 satellite images using LoFTR (Local Feature TRansformer), a state-of-the-art method for detector-free local feature matching with transformers.

### Used dataset: [Deforestation in Ukraine from Sentinel2 data](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine/data)


### This implementation is demonstrated on the given dataset, showing LoFTR's effectiveness in finding correspondences between satellite images taken at different times.

### Project Structure:
```bash

task2/
├── dataset/
│   ├── *_TCI.jp2
├── dataset_preparation.ipynb
├── Demo.ipynb
├── algorithm_creation.py
├── model_inference.py
├── requirements.txt
└── README.md
```

### Installation requirements:
1. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/Scripts/activate  
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   
### How to open the dataset creation notebook:
1. Make sure you have Jupyter Notebook installed.:
    ```bash
    pip install jupyter
    ```
2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook dataset_preparation.ipynb
    ```
   
### How to open the Demo Notebook:
1. Make sure you have Jupyter Notebook installed.:
    ```bash
    pip install jupyter
    ```
2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook Demo.ipynb
    ```

### Model Overview:
Detailed info about the algorithm: [Link](https://zju3dv.github.io/loftr/)

LoFTR (Local Feature TRansformer) is a novel approach to local feature matching. Unlike traditional methods (like SIFT or ORB) which follow a detect-then-describe paradigm, LoFTR is detector-free.

It works by first establishing dense, coarse-level correspondences using a Transformer architecture and then refining these matches at a fine level. This method allows LoFTR to find matches in low-texture areas, handle repetitive patterns, and be more robust to significant viewpoint and illumination changes—all of which are common challenges in satellite imagery.

### Pipeline overview:
<div>
<img src="https://zju3dv.github.io/loftr/images/loftr-arch.png"/>
</div>

LoFTR has four components: 
1. A local feature CNN extracts the coarse-level feature maps $\tilde{F}^A$ and $\tilde{F}^B$, together with the fine-level feature maps $\hat{F}^A$ and $\hat{F}^B$ from the image pair $I^A$ and $I^B$. 
2. The coarse feature maps are flattened to 1-D vectors and added with the positional encoding. The added features are then processed by the Local Feature TRansformer (LoFTR) module, which has $N_c$ self-attention and cross-attention layers. 
3. A differentiable matching layer is used to match the transformed features, which ends up with a confidence matrix $P_c$. The matches in $P_c$ are selected according to the confidence threshold and mutual-nearest-neighbor criteria, yielding the coarse-level match prediction $\mathcal{M}_c$. 
4. For every selected coarse prediction $(\tilde{i}, \tilde{j}) \in \mathcal{M}_c$, a local window with size $w \times w$ is cropped from the fine-level feature map. Coarse matches will be refined within this local window to a sub-pixel level as the final match prediction $\mathcal{M}_f$.


### Model inference process:
```bash
    python model_inference.py --dataset_path ./dataset --confidence 0.8 --image_size 1098 1098 --indexes_of_images 15 5 --model_pretrained outdoor
```
Output: plots with images, plot with matches, and saved figure with matches in the working directory with name `result.png`.



