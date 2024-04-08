import timm
import torch
from torch import nn, einsum
from einops import rearrange
from torchvision.models import DenseNet121_Weights, ResNet18_Weights
import torchvision.models as models
import torch.nn.functional as F


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 64*8 = 512
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_VIT(nn.Module):
    def __init__(
            self, model_name="vit_base_patch32_224", pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncdoer_res18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncdoer_res101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x


class mclSTExp_MLP(nn.Module):
    def __init__(self, temperature, image_embedding, spot_embedding, projection_dim, dropout=0.):
        super().__init__()
        self.x_embed = nn.Embedding(65536, spot_embedding)
        self.y_embed = nn.Embedding(65536, spot_embedding)
        self.image_ecode = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_ecode(batch["image"])
        spot_features = batch["expression"]
        image_embeddings = self.image_projection(image_features)
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_features + centers_x + centers_y

        spot_embeddings = self.spot_projection(spot_features)
        cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


class mclSTExp_Attention(nn.Module):
    def __init__(self,encoder_name, temperature, image_dim, spot_dim, projection_dim, heads_num, heads_dim, head_layers, dropout=0.):
        super().__init__()
        self.x_embed = nn.Embedding(65536, spot_dim)
        self.y_embed = nn.Embedding(65536, spot_dim)
        if encoder_name == "resnet50":
            self.image_encoder = ImageEncoder_Resnet()
        if encoder_name == "densenet121":
            self.image_encoder = ImageEncoder()
        if encoder_name == "vit":
            self.image_encoder = ImageEncoder_VIT()
        if encoder_name == "res18":
            self.image_encoder = ImageEncdoer_res18()
        if encoder_name == "res101":
            self.image_encoder = ImageEncdoer_res101()
        self.spot_encoder = nn.Sequential(
            *[attn_block(spot_dim, heads=heads_num, dim_head=heads_dim, mlp_dim=spot_dim, dropout=0.) for _ in
              range(head_layers)])

        self.image_projection = ProjectionHead(embedding_dim=image_dim, projection_dim=projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=spot_dim, projection_dim=projection_dim)

        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_feature = batch["expression"]
        image_embeddings = self.image_projection(image_features)

        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_feature + centers_x + centers_y
        spot_features = spot_features.unsqueeze(dim=0)

        spot_embeddings = self.spot_encoder(spot_features)
        spot_embeddings = self.spot_projection(spot_embeddings)
        spot_embeddings = spot_embeddings.squeeze(dim=0)

        cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean()


