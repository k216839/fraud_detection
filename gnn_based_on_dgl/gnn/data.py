import numpy as np
import pandas as pd
import os
import re
import dgl
import torch

def parse_feature(val):
    if val.lower() == 'false':
        return 0.0
    elif val.lower() == 'true':
        return 1.0
    else:
        return float(val)
    
def get_features(id_to_node, node_features_path):
    """
    Load node features từ file và chuyển thành features matrix đúng thứ tự theo DGL node ID.
    params:
        id_to_node: dictionary 
            Ánh xạ từ tên node (string, ví dụ: TransactionID) sang chỉ số
            node trong DGL graph
            Nếu gặp node chưa có trong dict này thì sẽ tự động thêm mới
            vào với ID tăng dần
        node_features_path: str
            Đường dẫn đến file chứa node features
    return:
        features: np.ndarray of shape: (num_nodes, num_features)
            Ma trận feature đã sắp xếp đúng thứ tự theo node ID
        new_nodé: list of int
            Danh sách các node chưa có trong đồ thị ban đầu, được thêm vào với ID mới

    """
    # Danh sách các DGL node ID theo thứ tự 
    indices = []
    # Danh sách các feature vectors và các node mới
    features, new_nodes = [], []

    max_node = max(id_to_node.values())

    with open(node_features_path, 'r') as f:
        for line in f:
            # Tách từng dòng CSV thành mảng string [node_id, f1, f2, ...]
            node_features = line.strip().split(",")

            node_id = node_features[0]
            features_ = np.array(list(map(parse_feature, node_features[1:])), dtype=np.float32)

            features.append(features_)
            # Nếu node_id chưa có trong id_to_node thì thêm mới vào
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)
            # Thêm node_id vào danh sách indices
            indices.append(id_to_node[node_id])
    # Chuyển danh sách features thành mảng numpy (num_nodes x num_features)
    features = np.array(features)
    # Sắp xếp lại theo thứ tự node ID để đảm bảo đúng thứ tự khi gán vào DGL graph
    features = features[np.argsort(indices), :]
    return features, new_nodes

def get_labels(
    id_to_node,
    n_nodes, 
    target_node_type, 
    labels_path,
    masked_nodes_path,
    additional_mask_rate=0
    ):
    """
    Load labels cho các node và tạo train/test mask để dùng trong huấn luyện GNN
    params:
        id_to_node: dictionary 
            Ánh xạ từ tên node (string, ví dụ: TransactionID) sang chỉ số
            node trong DGL graph
        n_nodes: int
            Số lượng node
        target_node_type: str
            Tên cột trong file label chứa node name cần ánh xạ
        labels_path: str
            Đường dẫn đến file CSV chứa labels
        masked_nodes_path: str
            Đường dẫn đến file chứa danh sách node cần mask khi huấn luyện
        additional_mask_rate: float
            Tỉ lệ bổ sung của các node có nhãn sẽ được mask ngẫu nhiên (dùng như validation)
    
    returns:
        labels: np.ndarray of shape: (num_nodes,)
            Mảng label ứng với các node
        train_mask: np.ndarray of shape: (num_nodes,)
            Mảng boolean đánh dấu node nào được dùng để huấn luyện
        test_mask: np.ndarray of shape: (num_nodes,)
            Mảng boolean đánh dấu node nào được dùng để test hoặc validation
    """
    # Đảo ngược từ node ID -> tên node (TransactionID)
    node_to_id = {v: k for k, v in id_to_node.items()}
    # Đọc file labels và đặt cột target_node_type làm index
    user_to_label = pd.read_csv(labels_path).set_index(target_node_type)
    # Lấy node name (theo thứ tự node ID từ 0 → n_nodes-1) → ánh xạ sang label
    target_node_names = pd.Series(node_to_id)[np.arange(n_nodes)].values
    labels = user_to_label.loc[map(int, target_node_names)].values.flatten()
    # Đọc danh sách các node bị mask sẵn (để test hoặc validation)
    masked_nodes = read_masked_nodes(masked_nodes_path)
    # Tao train_mask và test_mask 
    train_mask, test_mask = _get_mask(id_to_node, node_to_id, n_nodes, masked_nodes,
                                      additional_mask_rate=additional_mask_rate)
    return labels, train_mask, test_mask

def read_masked_nodes(masked_nodes_path):
    """
    Đọc danh sách các node (ví dụ: user ID) cần được ẩn (masked) từ một tệp văn bản.

    Chỉ định tập người dùng/test node không được sử dụng trong quá trình huấn luyện.

    Args:
        masked_nodes_path (str): Đường dẫn tới file chứa danh sách các node cần ẩn

    Returns:
        List[str]: Danh sách node ID dưới dạng chuỗi.
    """
    # Mở file và đọc từng dòng, loại bỏ ký tự xuống dòng
    with open(masked_nodes_path, "r") as f:
        masked_nodes = [line.strip() for line in f]

    return masked_nodes

def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
    """
    Tạo mask cho tập train và test.

    - Một số node không có nhãn (test nodes) → bị mask khỏi training.
    - Ngoài ra có thể mask thêm một phần node có nhãn theo tỷ lệ `additional_mask_rate`.

    Args:
        id_to_node (dict): Ánh xạ từ tên node sang chỉ số node trong DGL graph.
        node_to_id (dict): Ánh xạ ngược từ chỉ số node DGL sang tên node.
        num_nodes (int): Tổng số node trong đồ thị.
        masked_nodes (list): Danh sách ID của các node cần bị mask.
        additional_mask_rate (float): Tỷ lệ node có nhãn sẽ bị mask thêm (0.0 đến <1.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - `train_mask`: mảng nhị phân với 1 là dùng để train, 0 là bị mask.
            - `test_mask`: mảng nhị phân với 1 là test node, 0 là không dùng để test.
    """

    # mảng train (1) và test (0) ban đầu
    train_mask = np.ones(num_nodes)
    test_mask = np.zeros(num_nodes)

    # Đánh dấu các node trong masked_nodes
    for node_id in masked_nodes:
        node_idx = id_to_node[node_id]
        train_mask[node_idx] = 0
        test_mask[node_idx] = 1

    # Nếu có yêu cầu mask thêm một phần node có nhãn
    if additional_mask_rate and additional_mask_rate < 1:
        # Lọc ra các node chưa bị mask
        unmasked = np.array([
            idx for idx in range(num_nodes) 
            if node_to_id[idx] not in masked_nodes
        ])
        # Chọn ngẫu nhiên một phần trong số đó để mask thêm
        num_to_mask = int(additional_mask_rate * num_nodes)
        yet_unmasked = np.random.permutation(unmasked)[:num_to_mask]

        # Gán giá trị mask = 0 cho những node được chọn thêm này
        train_mask[yet_unmasked] = 0

    return train_mask, test_mask

def _get_node_idx(id_to_node, node_type, node_id, ptr):
    """
    Gán hoặc truy xuất chỉ số (index) của một node trong đồ thị dựa trên node_type và node_id.

    Hàm này dùng để ánh xạ (hoặc thêm mới nếu chưa có) một node vào một index duy nhất.
    Được dùng trong quá trình xây dựng đồ thị, nơi mỗi node cần có một ID số nguyên duy nhất.

    Args:
        id_to_node (dict): Cấu trúc từ điển 2 cấp: 
                           {node_type: {node_id: node_index}}.
        node_type (str): Loại node (ví dụ: 'user', 'account', 'transaction').
        node_id (str or int): ID cụ thể của node.
        ptr (int): Con trỏ index hiện tại, sẽ tăng nếu thêm node mới.

    Returns:
        Tuple[int, dict, int]: 
            - node_idx: index của node vừa xử lý.
            - id_to_node: dict đã được cập nhật.
            - ptr: con trỏ index mới sau khi xử lý.
    """

    # Nếu node_type đã tồn tại trong dict
    if node_type in id_to_node:
        # Nếu node_id đã có trong từ điển → lấy index ra
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            # Nếu chưa có → gán index mới và tăng ptr
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        # Nếu node_type chưa tồn tại → khởi tạo và gán index mới
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1

    return node_idx, id_to_node, ptr

def parse_edgelist(edges_path, id_to_node, header=False, source_type='user', sink_type='user'):
    """
    Phân tích file danh sách cạnh (edge list) và trả về danh sách cạnh (có hướng + ngược hướng).

    Được sử dụng để xây dựng đồ thị từ file CSV chứa các cặp node, ví dụ: user-user, user-merchant, v.v.

    Args:
        edges_path (str): Đường dẫn đến file CSV chứa các cặp node (source,sink) mỗi dòng là một cạnh.
        id_to_node (dict): Từ điển ánh xạ từ node_id → node_index, dùng để gán index cho node.
        header (bool): Cho biết dòng đầu tiên có phải header không. Nếu có, dòng đầu tiên chứa tên `source_type`, `sink_type`.
        source_type (str): Loại node nguồn (source) nếu không có header.
        sink_type (str): Loại node đích (sink) nếu không có header.

    Returns:
        Tuple:
            - edge_list (list of tuple): Danh sách cạnh (source → sink).
            - rev_edge_list (list of tuple): Danh sách cạnh ngược (sink → source).
            - id_to_node (dict): Cập nhật từ điển ánh xạ node.
            - source_type (str): Loại node nguồn.
            - sink_type (str): Loại node đích.
    """

    edge_list = []
    rev_edge_list = []
    source_pointer, sink_pointer = 0, 0  # Con trỏ cho node index mới nếu cần thêm

    with open(edges_path, "r") as f:
        for i, line in enumerate(f):
            source, sink = line.strip().split(",")

            if i == 0:
                if header:
                    # Nếu có header, dòng đầu là tên node_type
                    source_type, sink_type = source, sink

                # Lấy giá trị lớn nhất hiện tại trong ánh xạ để bắt đầu đếm tiếp
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1

                continue

            # Gán index cho source node (nếu chưa có thì thêm mới)
            source_node, id_to_node, source_pointer = _get_node_idx(
                id_to_node, source_type, source, source_pointer
            )

            # Nếu cùng loại node (graph đơn), dùng chung pointer
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(
                    id_to_node, sink_type, sink, source_pointer
                )
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(
                    id_to_node, sink_type, sink, sink_pointer
                )

            # Lưu cạnh gốc và cạnh ngược
            edge_list.append((source_node, sink_node))
            rev_edge_list.append((sink_node, source_node))

    return edge_list, rev_edge_list, id_to_node, source_type, sink_type

def read_edges(edges_path, nodes_path=None):
    """
    Đọc dữ liệu cạnh và đặc trưng (features) của các node nếu có.

    Hàm này xây dựng danh sách source/sink edges (dưới dạng index) và ánh xạ từ node_id gốc
    sang index dạng số nguyên dùng trong graph (DGL, PyG, v.v.).

    Args:
        edges_path (str): Đường dẫn đến file CSV chứa các cặp node (source,sink), mỗi dòng là một cạnh.
        nodes_path (str, optional): Đường dẫn đến file CSV chứa node_id và đặc trưng.

    Returns:
        Tuple:
            - sources (list[int]): Danh sách node index nguồn.
            - sinks (list[int]): Danh sách node index đích.
            - features (list[np.ndarray]): Danh sách vector đặc trưng tương ứng với node index.
            - id_to_node (dict): Ánh xạ node_id → node_index (dạng số nguyên).
    """
    node_pointer = 0                     # Dùng để cấp phát index duy nhất cho mỗi node
    id_to_node = {}                      # Ánh xạ node_id gốc → index
    features = []                        # Danh sách vector đặc trưng của node
    sources, sinks = [], []              # Danh sách các cạnh

    # Nếu có file node features
    if nodes_path is not None:
        with open(nodes_path, "r") as f:
            for line in f:
                node_feats = line.strip().split(",")  # tách theo dấu phẩy
                node_id = node_feats[0]

                if node_id not in id_to_node:
                    id_to_node[node_id] = node_pointer
                    node_pointer += 1

                    # Nếu có đặc trưng: chuyển phần còn lại sang float array
                    if len(node_feats) > 1:
                        feats = np.array(list(map(float, node_feats[1:])))
                        features.append(feats)

        # Đọc file cạnh, ánh xạ source/sink sang index
        with open(edges_path, "r") as f:
            for line in f:
                source, sink = line.strip().split(",")
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])

    # Nếu không có file nodes, ánh xạ node_id từ file edges
    else:
        with open(edges_path, "r") as f:
            for line in f:
                source, sink = line.strip().split(",")

                # Tạo index cho node nếu chưa có
                if source not in id_to_node:
                    id_to_node[source] = node_pointer
                    node_pointer += 1
                if sink not in id_to_node:
                    id_to_node[sink] = node_pointer
                    node_pointer += 1

                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])

    return sources, sinks, features, id_to_node

# def get_logger(name):
#     logger = logging.getLogger(name)
#     log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
#     logging.basicConfig(format=log_format, level=logging.INFO)
#     logger.setLevel(logging.INFO)
#     return logger
# logging = get_logger(__name__)

def get_edgelists(edgelist_expression, directory):
    """
    Trả về danh sách các tên file edgelist phù hợp với biểu thức regex hoặc danh sách tên file CSV tách bằng dấu phẩy.

    Hàm này hỗ trợ 2 trường hợp:
    1. Nếu `edgelist_expression` là danh sách file ngăn cách bởi dấu phẩy → trả về danh sách file trực tiếp.
    2. Nếu là biểu thức regex → lọc các file trong thư mục `directory` phù hợp với regex đó.

    Args:
        edgelist_expression (str): Biểu thức regex hoặc danh sách file CSV tách bằng dấu phẩy (",").
        directory (str): Đường dẫn đến thư mục chứa các file edgelist.

    Returns:
        list[str]: Danh sách tên file phù hợp.
    """
    # Trường hợp người dùng nhập vào danh sách file (file1.csv,file2.csv,...)
    if "," in edgelist_expression:
        return edgelist_expression.split(",")

    # Trường hợp là biểu thức regex → duyệt toàn bộ file trong thư mục và lọc
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]

def construct_graph(training_dir, edges, nodes, target_node_type):
    """
    Xây dựng một DGL heterograph từ nhiều file edgelist và file node features.

    Hàm này hỗ trợ:
    - Đọc các mối quan hệ giữa các loại node từ các file edgelist.
    - Đọc đặc trưng (features) cho một loại node cụ thể (target).
    - Tạo đồ thị không đồng nhất (heterograph) với các canonical edge types trong DGL.
    - Gán feature và thêm self-loop cho node target.

    Args:
        training_dir (str): Đường dẫn thư mục chứa các file edgelist và node feature.
        edges (List[str]): Danh sách tên file edgelist.
        nodes (str): Tên file chứa node ID và đặc trưng (dành cho node target).
        target_node_type (str): Loại node chính (sẽ được đổi tên thành 'target') để train model.

    Returns:
        Tuple[dgl.DGLHeteroGraph, np.ndarray, Dict[str, int], Dict[str, Dict[str, int]]]:
            - g: DGL heterograph đã khởi tạo.
            - features: ma trận đặc trưng của các node 'target'.
            - target_id_to_node: ánh xạ từ ID thực sang index cho node 'target'.
            - id_to_node: ánh xạ tổng hợp cho toàn bộ các loại node.
    """

    print("Getting relation graphs from the following edge lists : {} ".format(edges))
    edgelists, id_to_node = {}, {}

    for i, edge in enumerate(edges):
        # Đọc edgelist: trả về cả cạnh xuôi, cạnh ngược, và loại node ở hai đầu
        edgelist, rev_edgelist, id_to_node, src, dst = parse_edgelist(
            os.path.join(training_dir, edge), id_to_node, header=True
        )

        # Nếu là node chính (target) → đổi tên thành 'target' để xử lý đồng nhất
        if src == target_node_type:
            src = 'target'
        if dst == target_node_type:
            dst = 'target'

        if src == 'target' and dst == 'target':
            # Nếu là quan hệ nội bộ target → thêm self-loop sau
            print("Will add self loop for target later......")
        else:
            # Lưu cạnh xuôi và cạnh ngược theo canonical edge format
            edgelists[(src, src + '<>' + dst, dst)] = edgelist
            edgelists[(dst, dst + '<>' + src, src)] = rev_edgelist
            print("Read edges for {} from edgelist: {}".format(src + '<' + dst + '>', os.path.join(training_dir, edge)))

    # Đọc node features cho loại node chính (target)
    features, new_nodes = get_features(id_to_node[target_node_type], os.path.join(training_dir, nodes))
    print("Read in features for target nodes")

    # Thêm cạnh tự nối (self-loop) cho tất cả node target
    edgelists[('target', 'self_relation', 'target')] = [
        (t, t) for t in id_to_node[target_node_type].values()
    ]

    # Xây dựng DGL heterograph từ các canonical edge types
    g = dgl.heterograph(edgelists)
    print(
        "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
            g.ntypes, g.canonical_etypes
        )
    )
    print("Number of nodes of type target : {}".format(g.number_of_nodes('target')))

    # Gán feature cho node target
    g.nodes['target'].data['features'] = torch.from_numpy(features)

    # Cập nhật ánh xạ node: đổi tên target_node_type → 'target'
    target_id_to_node = id_to_node[target_node_type]
    id_to_node['target'] = target_id_to_node
    del id_to_node[target_node_type]

    return g, features, target_id_to_node, id_to_node
