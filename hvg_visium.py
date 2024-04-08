import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import scanpy as sc
from scanpy import read_visium, read_10x_mtx
import argparse
import pickle
import pandas as pd
import warnings
import json
from pathlib import Path
from typing import Union, Optional
from anndata import AnnData
from matplotlib.image import imread
import scprep as scp

# only need to run once to save hvg_matrix.npy
# filter expression matrices to only include HVGs shared across all datasets


def hvg_selection_and_pooling(exp_paths, samp_names, n_top_genes=1000):
    # input n expression matrices paths, output n expression matrices with only the union of the HVGs

    # read adata and find hvgs
    hvg_bools = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        print(adata.shape)
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # save hvgs
        hvg = adata.var['highly_variable']

        hvg_bools.append(hvg)

    # find union of hvgs
    hvg_union = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())

    # filter expression matrices
    filtered_exp_mtxs = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        adata = adata[hvg_union]
        filtered_exp_mtxs.append(adata)

    return filtered_exp_mtxs


def read_visium_alex(path: Union[str, Path],
                     *,
                     count_dir: str = 'raw_feature_bc_matrix',
                     library_id: str = None,
                     load_images: Optional[bool] = True,
                     source_image_path: Optional[Union[str, Path]] = None
                     ) -> AnnData:
    path = Path(path)
    adata = read_10x_mtx(path / count_dir)
    print(adata)
    adata.uns['spatial'] = dict()

    adata.uns["spatial"][library_id] = dict()
    if load_images:
        files = dict(
            tissue_positions_file=path / 'spatial/tissue_positions_list.csv',
            scalefactors_json_file=path / 'spatial/scalefactors_json.json',
            hires_image=path / 'spatial/tissue_hires_image.png',
            lowres_image=path / 'spatial/tissue_lowres_image.png',
        )
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    warnings.warn(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")
        adata.uns['spatial'][library_id]['images'] = dict()
        for res in ['hires', 'lowres']:
            try:
                adata.uns['spatial'][library_id]['images'][res] = imread(
                    str(files[f'{res}_image'])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]['scalefactors'] = json.loads(
            files['scalefactors_json_file'].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {}

        # read coordinates
        positions = pd.read_csv(files['tissue_positions_file'], header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm['spatial'] = adata.obs[
            ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        ].to_numpy()
        adata.obs.drop(
            columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata


def intersect_section_genes(adata_list):
    shared = set.intersection(*[set(adata.var_names) for adata in adata_list])
    return list(shared)


def adata_hvg_selection_and_pooling(adata_list, n_top_genes=1000):
    shared = intersect_section_genes(adata_list)
    #
    hvg_bools = []

    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to the spots in barcodes.tsv
        if adata.uns['library_id'] in samps1:
            index = \
                pd.read_csv(f"D:\dataset\Alex_NatGen/{adata.uns['library_id']}/filtered_count_matrix/barcodes.tsv.gz",
                            sep="\t",
                            header=None)[0].tolist()
        elif adata.uns['library_id'] in samps2:
            index = \
                pd.read_csv(
                    rf"D:\dataset\10xGenomics/{adata.uns['library_id']}/filtered_feature_bc_matrix/barcodes.tsv.gz",
                    sep="\t", header=None)[0].tolist()
        adata = adata[index].copy()
        # Subset to shared genes
        adata = adata[:, shared]
        print(adata.shape)
        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)

    hvg_union = hvg_bools[0]
    hvg_intersection = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
        print(sum(hvg_intersection), sum(hvg_bools[i]))
        hvg_intersection = hvg_intersection & hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())
    print("Number of HVGs (intersection): ", hvg_intersection.sum())

    with open('hvgs_intersection6.pickle', 'wb') as handle:
        pickle.dump(hvg_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hvgs_union.pickle6', 'wb') as handle:
        pickle.dump(hvg_union, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Add all the markers
    with open(r'D:\spatial_transcriptomics\DeepHis2Exp\src\scripts\Benchmarking_Figure1\1000hvg_common.pkl', 'rb') as f:
        gene_list = pickle.load(f).to_list()

    # hvg_union[gene_list] = True

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to the spots in barcodes.tsv
        if adata.uns['library_id'] in samps1:
            index = \
            pd.read_csv(f"D:\dataset\Alex_NatGen/{adata.uns['library_id']}/filtered_count_matrix/barcodes.tsv.gz",
                        sep="\t", header=None)[0].tolist()
        elif adata.uns['library_id'] in samps2:
            index = \
            pd.read_csv(rf"D:\dataset\10xGenomics/{adata.uns['library_id']}/filtered_feature_bc_matrix/barcodes.tsv.gz",
                        sep="\t", header=None)[0].tolist()
        adata = adata[index].copy()
        # Subset to shared genes
        # adata = adata[:, shared]
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())
    return filtered_exp_mtxs


samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
samps2 = ["block1", "block2", "FFPE"]

samps = samps1 + samps2
paths1 = [f"D:\dataset\Alex_NatGen/{samp}"
          for samp in samps1]
adata_list1 = [read_visium_alex(path) for path in paths1]
adata_list2 = [read_visium(rf"D:\dataset\10xGenomics/{samp}") for samp in samps2]
adata_list = adata_list1 + adata_list2
for i, adata in enumerate(adata_list):
    adata.uns['library_id'] = samps[i]

filtered_mtx = adata_hvg_selection_and_pooling(adata_list)


preprocessed_mtx = []
for i, mtx in enumerate(filtered_mtx):

    log_transformed_expression = scp.transform.log(scp.normalize.library_size_normalize(mtx))
    preprocessed_mtx.append(log_transformed_expression)
    pathset = f"./data/preprocessed_expression_matrices/Alex_10x_hvg/{samps[i]}"
    if not os.path.exists(pathset):
        os.makedirs(pathset)
    np.save(f"./data/preprocessed_expression_matrices/Alex_10x_hvg/{samps[i]}/preprocessed_matrix.npy",
            log_transformed_expression)
    print(f"her_data_preprocessed_mtx[{i}]:", log_transformed_expression.shape)
