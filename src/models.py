from copy import deepcopy

import torch
from torch import nn
from transformers import AutoModel


class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.3):
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class DualViewModel(nn.Module):
    def __init__(self, num_classes, model_name="owkin/phikon-v2", dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for block in self.encoder.encoder.layer[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.encoder.layernorm.parameters():
            parameter.requires_grad = True

        feature_dim = self.encoder.config.hidden_size
        self.target_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim), nn.GELU())
        self.context_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim), nn.GELU())
        self.gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 512), nn.GELU(), nn.Linear(512, dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.classifier = MLP(dim, num_classes)

    def encode(self, image):
        return self.encoder(pixel_values=image).last_hidden_state[:, 0]

    def forward(self, raw, masked):
        context = self.context_projection(self.encode(raw))
        target = self.target_projection(self.encode(masked))
        gate = self.gate(torch.cat([target, context], dim=1))
        mixed = gate * target + (1 - gate) * context
        fused = self.fusion(torch.cat([target, context, mixed], dim=1))
        return {"logits": self.classifier(fused), "fused": fused}


class NeighborGraphModel(nn.Module):
    def __init__(self, node_dim, num_classes, num_scales=3, hidden_dim=256):
        super().__init__()
        self.input_projection = nn.Linear(node_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, 0.2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3)
        self.target_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.readout = nn.MultiheadAttention(hidden_dim, 8, 0.2, batch_first=True)
        self.scale_gate = nn.Linear(hidden_dim, 1)
        self.classifier = MLP(hidden_dim, num_classes, hidden_dim)
        self.composition_head = nn.Linear(hidden_dim, num_classes)
        self.num_scales = num_scales

    def forward(self, graphs):
        scale_features = []
        for graph in graphs:
            nodes = self.input_projection(graph)
            nodes = torch.cat([self.target_query.expand(len(nodes), -1, -1), nodes[:, 1:]], dim=1)
            nodes = self.encoder(nodes)
            pooled, _ = self.readout(nodes[:, :1], nodes[:, 1:], nodes[:, 1:])
            scale_features.append(pooled[:, 0])
        stacked = torch.stack(scale_features, dim=1)
        weights = torch.softmax(self.scale_gate(stacked), dim=1)
        fused = (weights * stacked).sum(1)
        return {
            "logits": self.classifier(fused),
            "composition": torch.softmax(self.composition_head(fused), dim=1),
            "fused": fused,
        }


class HierarchicalFusion(nn.Module):
    def __init__(self, image_model, graph_model, image_dim=768, graph_dim=256):
        super().__init__()
        self.image_model = image_model.eval()
        self.graph_model = graph_model.eval()
        for parameter in self.image_model.parameters():
            parameter.requires_grad = False
        for parameter in self.graph_model.parameters():
            parameter.requires_grad = False

        self.graph_delta = MLP(graph_dim, image_dim)
        nn.init.zeros_(self.graph_delta[-1].weight)
        nn.init.zeros_(self.graph_delta[-1].bias)
        self.gate = nn.Sequential(nn.LayerNorm(image_dim * 2), nn.Linear(image_dim * 2, 512), nn.GELU(), nn.Linear(512, image_dim), nn.Sigmoid())
        self.classifier = deepcopy(image_model.classifier)

    def train(self, mode=True):
        super().train(mode)
        self.image_model.eval()
        self.graph_model.eval()
        return self

    def forward(self, raw, masked, graphs):
        with torch.no_grad():
            image = self.image_model(raw, masked)["fused"]
            graph = self.graph_model(graphs)["fused"]
        delta = self.graph_delta(graph)
        fused = image + self.gate(torch.cat([image, delta], dim=1)) * delta
        return {"logits": self.classifier(fused), "fused": fused}
