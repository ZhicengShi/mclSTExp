import argparse
import torch
import os
from dataset import SKIN, HERDataset, TenxDataset
from model import mclSTExp_Attention
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--max_epochs', type=int, default=90, help='')
    parser.add_argument('--temperature', type=float, default=1., help='temperature')
    parser.add_argument('--fold', type=int, default=0, help='fold')
    parser.add_argument('--dim', type=int, default=785, help='spot_embedding dimension (# HVGs)')  # 171, 785, 685
    parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
    parser.add_argument('--heads_num', type=int, default=8, help='attention heads num')
    parser.add_argument('--heads_dim', type=int, default=64, help='attention heads dim')
    parser.add_argument('--heads_layers', type=int, default=2, help='attention heads layer num')
    parser.add_argument('--dropout', type=float, default=0., help='dropout')
    parser.add_argument('--dataset', type=str, default='her2st', help='dataset')  # her2st cscc 10x
    parser.add_argument('--encoder_name', type=str, default='densenet121', help='image encoder')
    args = parser.parse_args()
    return args


def train(model, train_dataLoader, optimizer, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def load_data(args):
    if args.dataset == 'her2st':
        print(f'load dataset: {args.dataset}')
        train_dataset = HERDataset(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = HERDataset(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == 'cscc':
        print(f'load dataset: {args.dataset}')
        train_dataset = SKIN(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = SKIN(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == '10x':
        print(f'load dataset: {args.dataset}')
        examples1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
        examples2 = ["block1", "block2", "FFPE"]
        examples = examples1 + examples2
        datasets = [
                       TenxDataset(image_path=f"D:\dataset\Alex_NatGen/{example}/image.tif",
                                   spatial_pos_path=f"D:\dataset\Alex_NatGen/{example}/spatial/tissue_positions_list.csv",
                                   reduced_mtx_path=f"./data/preprocessed_expression_matrices/Alex_10x_hvg/{example}/preprocessed_matrix.npy",
                                   barcode_path=f"D:\dataset\Alex_NatGen/{example}/filtered_count_matrix/barcodes.tsv.gz")
                       for example in examples1
                   ] + [
                       TenxDataset(image_path=rf"D:\dataset\10xGenomics/{example}/image.tif",
                                   spatial_pos_path=rf"D:\dataset\10xGenomics/{example}/spatial/tissue_positions_list.csv",
                                   reduced_mtx_path=f"./data/preprocessed_expression_matrices/Alex_10x_hvg/{example}/preprocessed_matrix.npy",

                                   barcode_path=rf"D:\dataset\10xGenomics/{example}/filtered_feature_bc_matrix/barcodes.tsv.gz")
                       for example in examples2
                   ]

        datasets.pop(args.fold)
        print("Test name: ", examples[args.fold], "Test fold: ", args.fold)

        train_dataset = torch.utils.data.ConcatDataset(datasets)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        return train_loader, examples


def save_model(args, model, test_dataset=None, examples=[]):
    if args.dataset != '10x':
        os.makedirs(f"./model_result/{args.dataset}/{test_dataset.id2name[0]}", exist_ok=True)
        torch.save(model.state_dict(),
                   f"./model_result/{args.dataset}/{test_dataset.id2name[0]}/best_{args.fold}.pt")
    else:
        os.makedirs(f"./model_result/{args.dataset}/{examples[args.fold]}", exist_ok=True)
        torch.save(model.state_dict(),
                   f"./model_result/{args.dataset}/{examples[args.fold]}/best_{args.fold}.pt")


def main():
    args = generate_args()
    for i in range(32):  # 30,32   27
        args.fold = i
        print("当前fold:", args.fold)
        if args.dataset == '10x':
            train_dataLoader, examples = load_data(args)
        else:
            train_dataLoader, test_dataset = load_data(args)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = mclSTExp_Attention(encoder_name=args.encoder_name,
                                   spot_dim=args.dim,
                                   temperature=args.temperature,
                                   image_dim=args.image_embedding_dim,
                                   projection_dim=args.projection_dim,
                                   heads_num=args.heads_num,
                                   heads_dim=args.heads_dim,
                                   head_layers=args.heads_layers,
                                   dropout=args.dropout)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        for epoch in range(args.max_epochs):
            model.train()
            train(model, train_dataLoader, optimizer, epoch)

        if args.dataset == '10x':
            save_model(args, model, examples=examples)
        else:
            save_model(args, model, test_dataset=test_dataset)
        print("Saved Model")


if __name__ == '__main__':
    main()
