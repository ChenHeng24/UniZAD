## UniZAD
## Abstract
Zero-shot industrial anomaly detection aims to recognize and localize unknown defects without target-domain training samples. However, existing models still struggle to jointly achieve stable global semantics, fine-grained local structures, and precise boundary localization. To address this bottleneck, we propose UniZAD, a unified framework for zero-shot anomaly detection. UniZAD enhances anomaly representation via heterogeneous feature alignment and complementarity, adaptive hierarchical fusion, and multi-scale localization refinement. Specifically, we design a Direction-Aware Semantic-Structure Interaction Module (DSIM) to model complementarity and spatial alignment between ViT global semantics and DINOv2 local structures. To dynamically weight multi-level features, we design an Adaptive Hierarchical Complementary Fusion Module (AHCFM). It employs attention mechanisms to improve response consistency and spatial integrity for defects at various scales. In addition, a Progressive Multi-Scale Supervised Refinement Module (PMSRM) propagates low-resolution anomaly priors to high-resolution stages. It integrates edge gradient information to continuously correct anomaly boundaries, thereby achieving stepwise refinement from coarse localization to fine segmentation. Experimental results demonstrate that UniZAD outperforms existing state-of-the-art methods in both image-level anomaly recognition and pixel-level anomaly segmentation. Ablation studies and visualizations further validate each module, demonstrating that UniZAD yields stable, discriminative representations for industrial zero-shot anomaly detection.
### Overall Architecture
The overall framework of UniZAD is shown below:

![UniZAD Architecture](image.png)

## Requirements

Code has been tested to work on:

- Python 3.8
- PyTorch 1.6, 1.8
- CUDA 10.0, 10.1
- using additional packages as listed in `requirements.txt`

## Datasets

You can download the datasets from their official sources, and use utilities in datasets/generate_dataset_json/ to generate a compatible meta.json. 
Alternatively from the AdaCLIP repository which has provided a compatible format of the datasets.
Place all datasets under DATASETS_ROOT, which is defined in ./__init__.py.    
## Training
To train new checkpoints and test on the medical and industrial datasets using the default setting, simply run:

bash reproduce.sh new_model 0
where new_model and 0 specify the name for the checkpoint and the available cuda device ID.
## Organize Your Data
Your dataset must either include a meta.json file at the root directory, or be organized so that one can be automatically generated.

The meta.json should follow this format:

A dictionary with "train" and "test" at the highest level
Each section contains class names mapped to a list of samples
Each sample includes:
img_path: path to the image relative to the root dir
mask_path: path to the mask relative to the root dir (empty for normal samples)
cls_name: class name
specie_name: subclass or condition (e.g., "good", "fault1")
anomaly: anomaly label; 0 (normal) or 1 (anomalous)
If your dataset does not include the required meta.json, you can generate it automatically by organizing your data as shown below and running datasets/generate_dataset_json/custom_dataset.py:
## Run Testing
Then you should place your dataset in the DATASETS_ROOT, specified in datasets/generate_dataset_json/__init__.py and run the inference:

python test.py --dataset YOUR_DATASET --model_name default --epoch 5
