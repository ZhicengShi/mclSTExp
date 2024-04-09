# Multimodal contrastive learning for spatial gene expression prediction using histology images

In this study, we propose mclSTExp: a multimodal deep learning approach utilizing Transformer and contrastive learning architecture. Inspired by the field of natural language processing, we regard the spots detected by ST technology as ''words'' and the sequences of these spots as ''sentences'' containing multiple ''words''. We employ a self-attention mechanism to extract features from these ''words'' and combine them with learnable position encoding to seamlessly integrate the positional information of these ''words''. Subsequently, we employ contrastive learning methods to fuse the combined features with image features. we employed two human breast cancer datasets and one human cutaneous squamous cell carcinoma (cSCC) dataset. Our experimental results demonstrate that mclSTExp accurately predicts gene expression in histopathological images at different spatial resolutions. This is achieved by leveraging the features of each spot, its spatial information, and histological image features. Additionally, mclSTExp exhibits the capability to interpret cancer-specific overexpressed genes and identify specific spatial domains annotated by pathologists.

![(Variational)](workflow.png)


## System environment
Required package:
- PyTorch >= 2.1.0
- scanpy >= 1.8
- python >=3.9

## Datasets

-  human HER2-positive breast tumor ST data https://github.com/almaan/her2st/.
-  human cutaneous squamous cell carcinoma 10x Visium data (GSE144240).
-  10x Genomics Visium data and Swarbrick’s Laboratory Visium data https://doi.org/10.48610/4fb74a9.

## mclSTExp pipeline

- Run `hvg_her2st.py` generation of highly variable genes.
- Run `train.py`
- Run `evel.py`
- Run `tutorial.ipynb`
