import anndata
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
from model import mclSTExp_Attention
from dataset import HERDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from utils import get_R
from train import generate_args


def build_loaders_inference():
    datasets = []
    for i in range(32):
        dataset = HERDataset(train=False, fold=i)
        print(dataset.id2name[0])
        datasets.append(dataset)

    dataset = torch.utils.data.ConcatDataset(datasets)

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Finished building loaders")
    return test_loader


def get_embeddings(model):
    test_loader = build_loaders_inference()
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to('cuda')
    print("Finished loading model")

    test_image_embeddings = []
    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)

            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_feature = spot_feature + centers_x + centers_y
            # coordinates = batch["position"].float().cuda()
            # scale = max(coordinates[:, 0].max() - coordinates[:, 0].min(),
            #             coordinates[:, 1].max() - coordinates[:, 1].min())
            # coordinates[:, 0] = (coordinates[:, 0] - coordinates[:, 0].min()) / scale
            # coordinates[:, 1] = (coordinates[:, 1] - coordinates[:, 1].min()) / scale
            # pe = model.pe_enc(coordinates)
            # pe = model.act(pe)
            # pe = model.layer_norm(pe)
            # spot_feature = spot_feature + pe
            spot_features = spot_feature.unsqueeze(dim=0)
            spot_embedding = model.spot_encoder(spot_features)
            spot_embedding = model.spot_projection(spot_embedding).squeeze(dim=0)
            spot_embeddings.append(spot_embedding)
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings)


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()


def save_embeddings(model_path, save_path, datasize):

    args = generate_args()
    model = mclSTExp_Attention(encoder_name=args.encoder_name,
                               spot_dim=args.dim,
                               temperature=args.temperature,
                               image_dim=args.image_embedding_dim,
                               projection_dim=args.projection_dim,
                               heads_num=args.heads_num,
                               heads_dim=args.heads_dim,
                               head_layers=args.heads_layers,
                               dropout=args.dropout)

    img_embeddings_all, spot_embeddings_all = get_embeddings(model_path, model)
    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    print("img_embeddings_all.shape", img_embeddings_all.shape)
    print("spot_embeddings_all.shape", spot_embeddings_all.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        print("image_embeddings.shape", image_embeddings.shape)
        print("spot_embeddings.shape", spot_embeddings.shape)
        np.save(save_path + "img_embeddings_" + str(i + 1) + ".npy", image_embeddings.T)
        np.save(save_path + "spot_embeddings_" + str(i + 1) + ".npy", spot_embeddings.T)


SAVE_EMBEDDINGS = False

names = os.listdir(r"D:\dataset\Her2st\data/ST-cnts")
names.sort()
names = [i[:2] for i in names][1:33]

datasize = [np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy").shape[1] for
            name in names]

if SAVE_EMBEDDINGS:
    for fold in range(32):
        save_embeddings(model_path=f"./model_result/her2st/{names[fold]}/best_{fold}.pt",
                        save_path=f"./embedding_result/her2st_result/embeddings_{fold}/",
                        datasize=datasize, dim=785, fold=fold)


spot_expressions = [np.load(f"./data/preprocessed_expression_matrices/her2st/{name}/preprocessed_matrix.npy")
                    for name in names]
hvg_pcc_list = []
heg_pcc_list = []
mse_list = []
mae_list = []

for fold in range(32):

    save_path = f"./embedding_result/her2st_result/embeddings_{fold}/"
    spot_embeddings = [np.load(save_path + f"spot_embeddings_{i + 1}.npy") for i in range(32)]
    image_embeddings = np.load(save_path + f"img_embeddings_{fold + 1}.npy")


    image_query = image_embeddings
    expression_gt = spot_expressions[fold]
    spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold + 1:]
    spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold + 1:]

    spot_key = np.concatenate(spot_embeddings, axis=1)
    expression_key = np.concatenate(spot_expressions_rest, axis=1)

    method = "weighted"
    save_path = f"./her2st_pred_att/{names[fold]}/"
    os.makedirs(save_path, exist_ok=True)
    if image_query.shape[1] != 256:
        image_query = image_query.T
        print("image query shape: ", image_query.shape)
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        print("expression_gt shape: ", expression_gt.shape)
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
        print("spot_key shape: ", spot_key.shape)
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T
        print("expression_key shape: ", expression_key.shape)

    indices = find_matches(spot_key, image_query, top_k=200)
    matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
    matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
    for i in range(indices.shape[0]):
        a = np.linalg.norm(spot_key[indices[i, :], :] - image_query[i, :], axis=1, ord=1)
        # from sklearn.metrics.pairwise import cosine_similarity
        #
        # a = 1 - cosine_similarity(spot_key[indices[i, :], :], image_query[i, :].reshape(1, -1))
        reciprocal_of_square_a = np.reciprocal(a ** 2)
        weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
        weights = weights.flatten()
        matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
        matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0,
                                                        weights=weights)

    # np.save(save_path + "matched_spot_expression_pred_mclSTExp.npy", matched_spot_expression_pred.T)
    true = expression_gt
    pred = matched_spot_expression_pred

    gene_list_path = "D:\dataset\Her2st\data/her_hvg_cut_1000.npy"
    gene_list = list(np.load(gene_list_path, allow_pickle=True))
    adata_ture = anndata.AnnData(true)
    adata_pred = anndata.AnnData(pred)

    adata_pred.var_names = gene_list
    adata_ture.var_names = gene_list

    gene_mean_expression = np.mean(adata_ture.X, axis=0)
    top_50_genes_indices = np.argsort(gene_mean_expression)[::-1][:50]
    top_50_genes_names = adata_ture.var_names[top_50_genes_indices]
    top_50_genes_expression = adata_ture[:, top_50_genes_names]
    top_50_genes_pred = adata_pred[:, top_50_genes_names]

    heg_pcc, heg_p = get_R(top_50_genes_pred, top_50_genes_expression)
    hvg_pcc, hvg_p = get_R(adata_pred, adata_ture)
    hvg_pcc = hvg_pcc[~np.isnan(hvg_pcc)]

    heg_pcc_list.append(np.mean(heg_pcc))
    hvg_pcc_list.append(np.mean(hvg_pcc))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true, pred)
    mse_list.append(mse)
    print("Mean Squared Error (MSE): ", mse)
    mae = mean_absolute_error(true, pred)
    mae_list.append(mae)
    print("Mean Absolute Error (MAE): ", mae)

print(f"avg heg pcc: {np.mean(heg_pcc_list):.4f}")
print(f"avg hvg pcc: {np.mean(hvg_pcc_list):.4f}")
print(f"Mean Squared Error (MSE): {np.mean(mse_list):.4f}")
print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.4f}")
