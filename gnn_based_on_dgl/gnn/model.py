import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import dgl.function as fn

class HeteroRGCNLayer(nn.Module):
    """
    Một lớp GNN cho đồ thị không đồng nhất theo kiểu R-GCN,
    với một ma trận trọng số riêng cho từng loại quan hệ (etype).

    Lớp này thực hiện message passing theo từng quan hệ, sau đó gộp lại bằng tổng.

    Args:
        in_size (int): Kích thước đầu vào của feature vector cho mỗi node.
        out_size (int): Kích thước đầu ra sau khi truyền qua Linear.
        etypes (list[str]): Danh sách các tên edge types (tức các quan hệ trong đồ thị).
    """

    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # Khởi tạo ma trận W_r riêng cho mỗi loại quan hệ (etype)
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size)
            for name in etypes
        })

    def forward(self, G, feat_dict):
        """
        Truyền thông tin (message passing) qua các quan hệ để cập nhật đặc trưng node.

        Args:
            G (dgl.DGLHeteroGraph): Đồ thị không đồng nhất đã định nghĩa canonical edge types.
            feat_dict (dict[str, Tensor]): Dict ánh xạ từ node type → feature tensor.

        Returns:
            dict[str, Tensor]: Dict ánh xạ từ node type → feature mới sau message passing.
        """
        funcs = {}

        # Duyệt qua từng canonical edge type: (src_type, etype, dst_type)
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                # W_r * h: biến đổi feature node gốc theo ma trận tương ứng với etype
                Wh = self.weight[etype](feat_dict[srctype])

                # Gán vào node src để sử dụng trong truyền thông tin
                G.nodes[srctype].data[f'Wh_{etype}'] = Wh

                # Xác định hàm truyền và gom theo quan hệ này:
                # copy_u: copy 'Wh_*' từ node gốc làm message
                # mean: tính trung bình các message đến node đích
                funcs[etype] = (
                    fn.copy_u(f'Wh_{etype}', 'm'),
                    fn.mean('m', 'h')
                )

        # Gọi message passing đồng thời cho tất cả loại quan hệ, rồi cộng các output lại
        G.multi_update_all(funcs, 'sum')

        # Trả về đặc trưng mới của các node sau khi update
        return {
            ntype: G.nodes[ntype].data['h']
            for ntype in G.ntypes
            if 'h' in G.nodes[ntype].data
        }

class HeteroRGCN(nn.Module):
    """
    Mô hình Heterogeneous Relational Graph Convolutional Network (Hetero-RGCN) áp dụng
    cho đồ thị không đồng nhất, hỗ trợ embedding học được cho các node không có đặc trưng.

    Args:
        ntype_dict (dict[str, int]): Số lượng node của mỗi loại node type.
        etypes (list[str]): Danh sách tên các quan hệ (edge types) trong đồ thị.
        in_size (int): Kích thước embedding đầu vào cho các node không có feature.
        hidden_size (int): Kích thước layer ẩn trong GNN.
        out_size (int): Kích thước đầu ra (logits, class scores).
        n_layers (int): Số lượng tầng GNN (bao gồm hidden + 1 input).
        embedding_size (int): Kích thước input của tầng GNN đầu tiên (thường bằng `in_size`).
    """

    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, embedding_size):
        super(HeteroRGCN, self).__init__()

        # Sử dụng nhúng nút có thể đào tạo làm đầu vào không có tính năng
        embed_dict = {
            ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
            for ntype, num_nodes in ntype_dict.items()
            if ntype != 'target'
        }

        # Khởi tạo embedding bằng Xavier Uniform
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)

        # Lưu lại embedding dưới dạng ParameterDict
        self.embed = nn.ParameterDict(embed_dict)

        # Tạo danh sách các layer
        self.layers = nn.ModuleList()

        # Layer đầu tiên: dùng embedding_size làm input
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes))

        # Các hidden layers
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))

        # Output layer: Linear layer áp dụng lên 'target' node
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, g, features):
        """
        Args:
            g (dgl.DGLHeteroGraph): Đồ thị không đồng nhất.
            features (Tensor): Đặc trưng đầu vào của node 'target'.

        Returns:
            Tensor: Output logits cho node 'target'.
        """

        # Tập hợp embedding cho các node không phải 'target'
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}

        # Gán feature thật cho node 'target'
        h_dict['target'] = features

        # Truyền qua các lớp HeteroRGCNLayer
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                # Áp dụng Leaky ReLU cho tất cả node type
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g, h_dict)

        # Áp dụng layer cuối cùng (Linear) lên node 'target' để ra logits
        return self.layers[-1](h_dict['target'])
