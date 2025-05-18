import os
import torch
import time
import copy
from sklearn.metrics import confusion_matrix
import pickle
from gnn.model import *
from gnn.utils import *
from gnn.data import *
from gnn.estimator_fns import *

def normalize(feature_matrix):
    mean = torch.mean(feature_matrix, axis=0)
    stdev = torch.sqrt(torch.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev

def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask,
             device, n_epochs, thresh, compute_metrics=True):
    """
    Huấn luyện mô hình HeteroRGCN.

    Hàm này thực hiện:
    - Huấn luyện qua nhiều epoch.
    - Đánh giá F1 score mỗi epoch.
    - Lưu model tốt nhất (dựa trên loss thấp nhất).
    - Tính và hiển thị các metrics trên test set.

    Args:
        model (nn.Module): Mô hình HeteroRGCN cần huấn luyện.
        optim (torch.optim.Optimizer): Optimizer.
        loss (callable): Hàm loss function (vd: BCEWithLogitsLoss).
        features (Tensor): Đặc trưng đầu vào của node 'target'.
        labels (Tensor): Nhãn thực của node 'target'.
        train_g (dgl.DGLHeteroGraph): Đồ thị dùng để train.
        test_g (dgl.DGLHeteroGraph): Đồ thị dùng để test/inference.
        test_mask (Tensor): Mặt nạ chọn node trong tập test.
        device (torch.device): Thiết bị tính toán (CPU/GPU).
        n_epochs (int): Số epoch huấn luyện.
        thresh (float): Ngưỡng phân lớp để chuyển xác suất → nhãn.
        compute_metrics (bool): Nếu True, tính các metric cuối cùng.

    Returns:
        Tuple:
            - best_model (nn.Module): Mô hình có loss tốt nhất trên train.
            - class_preds (np.ndarray): Dự đoán nhị phân trên test.
            - pred_proba (np.ndarray): Xác suất dự đoán trên test.
    """
    duration = []
    best_loss = 1

    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        # Forward: truyền features qua model với train graph
        pred = model(train_g, features.to(device))

        # Tính loss
        l = loss(pred, labels)

        # Backpropagation
        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l.item()
        duration.append(time.time() - tic)

        # Đánh giá performance hiện tại (vd: F1)
        metric = evaluate_f1(model, train_g, features, labels, device)

        print("Epoch {:05d}, Time(s) {:.4f}, Loss {:.4f}, F1 {:.4f}".format(
            epoch, np.mean(duration), loss_val, metric
        ))

        # Ghi log kết quả vào file
        epoch_result = "{:05d},{:.4f},{:.4f},{:.4f}\n".format(
            epoch, np.mean(duration), loss_val, metric)
        os.makedirs('./output', exist_ok=True)
        with open('./output/results.txt','w') as f:    
            f.write("Epoch,Time(s),Loss,F1\n") 
            f.write(epoch_result)

        # Lưu model tốt nhất
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)

    # Inference trên test set với mô hình tốt nhất
    class_preds, pred_proba = get_model_class_predictions(
        best_model, test_g, features, labels, device, threshold=thresh
    )

    # Tính metrics nếu được yêu cầu
    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(
            class_preds,
            pred_proba,
            labels.numpy(),
            test_mask.numpy(),
            './output/'
        )
        print("Metrics")
        print("""Confusion Matrix:
                                {}
                                f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                             """.format(cm, f1, p, r, acc, roc, pr, ap))

    return best_model, class_preds, pred_proba

def get_precision_recall_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true: A list of labels in 0 or 1: 1 * N
    :param y_pred: A list of labels in 0 or 1: 1 * N
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true, y_pred)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    return precision, recall, f1

def evaluate_f1(model, g, features, labels, device):
    "Compute the F1 value in a binary classification case"

    preds = model(g, features.to(device))
    preds = torch.argmax(preds, axis=1).numpy()
    precision, recall, f1 = get_precision_recall_f1_score(labels, preds)

    return f1

def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    """
    Dự đoán nhãn (class predictions) và xác suất (probabilities) từ mô hình HeteroRGCN.

    Hàm này truyền dữ liệu qua mô hình, lấy xác suất thông qua softmax,
    sau đó trả về:
    - Dự đoán nhãn (argmax hoặc so với threshold).
    - Xác suất lớp dương tính (class 1).

    Args:
        model (nn.Module): Mô hình đã huấn luyện (HeteroRGCN).
        g (dgl.DGLHeteroGraph): Đồ thị để truyền qua mô hình.
        features (Tensor): Đặc trưng đầu vào của node 'target'.
        labels (Tensor): Nhãn thực tế (không dùng trong hàm này nhưng có thể cần cho mở rộng).
        device (torch.device): Thiết bị tính toán (CPU/GPU).
        threshold (float, optional): Ngưỡng xác suất để phân lớp (nếu không dùng argmax).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - class_preds: Dự đoán nhị phân (0/1).
            - pred_proba: Xác suất dự đoán của lớp 1 (positive class).
    """

    # Truyền qua mô hình để lấy đầu ra chưa chuẩn hóa (logits)
    unnormalized_preds = model(g, features.to(device))

    # Chuyển logits thành xác suất bằng softmax
    pred_proba = torch.softmax(unnormalized_preds, dim=-1)

    if not threshold:
        # Nếu không có ngưỡng → dùng argmax để phân lớp (multi-class)
        class_preds = unnormalized_preds.argmax(axis=1).detach().cpu().numpy()
    else:
        # Nếu có ngưỡng → phân loại binary dựa vào prob class 1
        class_preds = np.where(pred_proba.detach().cpu().numpy() > threshold, 1, 0)

    # Trả về cả xác suất lớp 1
    return class_preds, pred_proba[:, 1].detach().cpu().numpy()

def save_model(g, model, model_dir, id_to_node, mean, stdev):
    """
    Lưu toàn bộ thông tin liên quan đến mô hình đã huấn luyện:
    - Tham số mô hình PyTorch.
    - Metadata về cấu trúc đồ thị và thống kê features.
    - Embedding học được cho các node không thuộc 'target' dưới dạng CSV (Grelim format).

    Args:
        g (dgl.DGLHeteroGraph): Đồ thị huấn luyện.
        model (nn.Module): Mô hình đã huấn luyện (HeteroRGCN).
        model_dir (str): Thư mục lưu toàn bộ output.
        id_to_node (dict[str, dict[str, int]]): Ánh xạ node_id gốc → index theo từng node type.
        mean (np.ndarray): Vector trung bình feature của node 'target'.
        stdev (np.ndarray): Vector độ lệch chuẩn của feature node 'target'.
    """

    # 1. Lưu trọng số mô hình PyTorch
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # 2. Lưu thông tin cấu trúc đồ thị và statistics (dùng để khởi tạo lại model RGCN khi inference)
    etype_list = g.canonical_etypes  # Danh sách (src, rel, dst)
    ntype_cnt = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}  # Số lượng node theo từng type
    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'etypes': etype_list,
            'ntype_cnt': ntype_cnt,
            'feat_mean': mean,
            'feat_std': stdev
        }, f)

    # 3. Với mỗi node type (ngoại trừ 'target'), lưu ID gốc vào index và embedding vào file CSV
    for ntype, mapping in id_to_node.items():
        if ntype == 'target':
            continue  # Bỏ qua node 'target' vì đã có feature đầu vào riêng

        # Tách danh sách ID gốc và ID số nguyên (node index trong graph)
        old_id_list, node_id_list = [], []
        for old_id, node_id in mapping.items():
            old_id_list.append(old_id)
            node_id_list.append(node_id)

        # Lấy embedding đã học được từ mô hình
        node_feats = model.embed[ntype].detach().numpy()

        num_nodes = node_feats.shape[0]
        num_feats = node_feats.shape[1]

        # Tạo DataFrame chứa thông tin node ID (gốc và định dạng cho Grelim)
        node_ids_df = pd.DataFrame({'~label': [ntype] * num_nodes})
        node_ids_df['~id_tmp'] = old_id_list
        node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
        node_ids_df['node_id'] = node_id_list

        # Tạo DataFrame chứa embedding, đặt tên cột theo định dạng Grelim: val1:Double, val2:Double, ...
        cols = {
            f'val{i+1}:Double': node_feats[:, i]
            for i in range(num_feats)
        }
        node_feats_df = pd.DataFrame(cols)

        # Gộp ID và embedding thành một bảng duy nhất
        node_id_feats_df = node_ids_df.merge(
            node_feats_df, left_on='node_id', right_on=node_feats_df.index
        )

        # Xóa các cột phụ để đảm bảo định dạng cho Grelim
        node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)

        # Lưu file CSV embedding cho node type hiện tại
        node_id_feats_df.to_csv(
            os.path.join(model_dir, ntype + '.csv'),
            index=False, header=True, encoding='utf-8'
        )

def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):

    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'], n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)

    return model

if __name__ == '__main__':
    # In thông tin phiên bản thư viện
    print('numpy version:{} PyTorch version:{} DGL version:{}'.format(np.__version__,
                                                                      torch.__version__,
                                                                      dgl.__version__))

    # Parse các argument dòng lệnh (định nghĩa trong parse_args)
    args = parse_args()
    print(args)

    # Lấy danh sách edgelist từ thư mục training dựa vào pattern 'relation*'
    args.edges = get_edgelists('relation*', args.training_dir)

    # Xây dựng đồ thị và đặc trưng node từ edgelist + node features
    g, features, target_id_to_node, id_to_node = construct_graph(
        args.training_dir,
        args.edges,
        args.nodes,
        args.target_ntype
    )

    # Chuẩn hóa features cho node 'target'
    mean, stdev, features = normalize(torch.from_numpy(features))
    print('feature mean shape:{}, std shape:{}'.format(mean.shape, stdev.shape))

    # Gán feature chuẩn hóa cho node 'target'
    g.nodes['target'].data['features'] = features

    print("Getting labels")
    # Lấy nhãn và test mask cho các node target (dựa vào file labels + new_accounts)
    n_nodes = g.number_of_nodes('target')
    labels, _, test_mask = get_labels(
        target_id_to_node,
        n_nodes,
        args.target_ntype,
        os.path.join(args.training_dir, args.labels),
        os.path.join(args.training_dir, args.new_accounts)
    )
    print("Got labels")

    # Chuyển về tensor
    labels = torch.from_numpy(labels).float()
    test_mask = torch.from_numpy(test_mask).float()

    # Thống kê số node và số cạnh toàn đồ thị
    n_nodes = torch.sum(torch.tensor([g.number_of_nodes(n_type) for n_type in g.ntypes]))
    n_edges = torch.sum(torch.tensor([g.number_of_edges(e_type) for e_type in g.etypes]))

    print("""----Data statistics------
                #Nodes: {}
                #Edges: {}
                #Features Shape: {}
                #Labeled Test samples: {}""".format(n_nodes,
                                                    n_edges,
                                                    features.shape,
                                                    test_mask.sum()))

    # Thiết lập thiết bị train (GPU nếu có)
    if args.num_gpus:
        cuda = True
        device = torch.device('cuda:0')
    else:
        cuda = False
        device = torch.device('cpu')

    print("Initializing Model")
    in_feats = features.shape[1]    # Số chiều input feature
    n_classes = 2                   # Phân loại nhị phân

    # Số lượng node cho từng loại node type
    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}

    # Khởi tạo mô hình HeteroRGCN
    model = get_model(ntype_dict, g.etypes, vars(args), in_feats, n_classes, device)
    print("Initialized Model")

    # Đưa dữ liệu về đúng device
    features = features.to(device)
    labels = labels.long().to(device)
    test_mask = test_mask.to(device)

    # Hàm mất mát
    loss = torch.nn.CrossEntropyLoss()

    # Khởi tạo optimizer Adam
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting Model training")
    # Ghi log header cho output log file
    if os.path.exists('./output/results.txt'):
        os.remove('./output/results.txt')

    # Huấn luyện mô hình full-graph với dữ liệu và graph
    model, class_preds, pred_proba = train_fg(
        model, optim, loss, features, labels, g, g,
        test_mask, device, args.n_epochs,
        args.threshold, args.compute_metrics
    )

    print("Finished Model training")

    print("Saving model")
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Lưu mô hình + metadata + embeddings ra đĩa
    save_model(g, model, args.model_dir, id_to_node, mean, stdev)
    print("Model and metadata saved")
