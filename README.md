# Multimodal contrastive learning for spatial gene expression prediction using histology images

In this study, we propose mclSTExp: a multimodal deep learning approach utilizing Transformer and contrastive learning architecture. Inspired by the field of natural language processing, we regard the spots detected by ST technology as ''words'' and the sequences of these spots as ''sentences'' containing multiple ''words''. We employ a self-attention mechanism to extract features from these ''words'' and combine them with learnable position encoding to seamlessly integrate the positional information of these ''words''. Subsequently, we employ a contrastive learning framework to fuse the combined features with image features. we employed two human breast cancer datasets and one human cutaneous squamous cell carcinoma (cSCC) dataset. Our experimental results demonstrate that mclSTExp accurately predicts gene expression in H\&E images at different spatial resolutions. This is achieved by leveraging the features of each spot, its spatial information, and H\&E image features. Additionally, mclSTExp demonstrates the ability to interpret specific cancer-overexpressed genes, immunologically relevant genes, preserve the original gene expression patterns, and identify specific spatial domains annotated by pathologists.

![(Variational)](workflow.png)

## System environment
Required package:
- PyTorch >= 2.1.0
- scanpy >= 1.8
- python >=3.9

## Datasets
Four publicly available ST datasets were used in this study.
-  DLPFC dataset consists of 12 sections of the dorsolateral prefrontal cortex (DLPFC) sampled from three individuals. The number of spots for each section ranges from 3498 to 4789. The original authors have manually annotated the areas of the DLPFC layers and white matter. The datasets are available in the spatialLIBD package from http://spatial.libd.org/spatialLIBD.
-  MouseBrain dataset includes a coronal brain section sample from an adult mouse, with 2903 sampled spots. The datasets are available in https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_web_summary.html.
-  Human Breast Cancer1 (BC1) dataset includes a fresh frozen invasive ductal carcinoma breast tissue section sample, with 3813 sampled spots. The datasets are available in https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-0-0.
-  Human Breast Cancer2 (BC2) dataset includes a formalin-fixed invasive breast carcinoma tissue section sample, with 2518 sampled spots. The datasets are available in https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0

## HisToSGE pipeline

- Run `train.py`
- Run `test.py`
