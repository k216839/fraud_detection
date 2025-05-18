import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt


def get_metrics(pred, pred_proba, labels, mask, out_dir):
    """
    Hàm này lọc ra các phần tử thuộc tập đánh giá (theo `mask`), sau đó tính các chỉ số:
        - accuracy, f1 score, precision, recall
        - roc_auc, pr_auc
        - average precision
        - confusion_matrix
        Sau đó lưu biểu đồ ROC và Precision-Recall vào thư mục chỉ định.
    Args:
        pred (np.ndarray): Dự đoán nhãn (0/1) của mô hình.
        pred_proba (np.ndarray): Xác suất dự đoán cho lớp positive.
        labels (np.ndarray): Nhãn thật (ground truth) dạng 0/1.
        mask (np.ndarray): Mặt nạ boolean hoặc nhị phân (1/0) để chọn các phần tử cần đánh giá.
        out_dir (str): Đường dẫn thư mục để lưu ảnh ROC và PR curve.

    Returns:
        Tuple:
            - acc (float): Accuracy
            - f1 (float): F1 score
            - precision (float): Precision
            - recall (float): Recall
            - roc_auc (float): Area under ROC curve
            - pr_auc (float): Area under Precision-Recall curve
            - ap (float): Average Precision
            - confusion_matrix (pd.DataFrame)
    """
    # Lọc theo mask
    labels, pred, pred_proba = labels[np.where(mask)], pred[np.where(mask)], pred_proba[np.where(mask)]

    acc = ((pred == labels)).sum() / mask.sum()

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0

    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    ap = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)

    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
    save_pr_curve(prc, rec, pr_auc, ap, os.path.join(out_dir, "pr_curve.png"))

    return acc, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix


def save_roc_curve(fpr, tpr, roc_auc, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_pr_curve(fpr, tpr, pr_auc, ap, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model PR curve: AP={0:0.2f}'.format(ap))
    plt.legend(loc="lower right")
    f.savefig(location)


def save_graph_drawing(g, location):
    """
    Vẽ và lưu biểu đồ đồ thị bằng NetworkX + Matplotlib.

    Mỗi node sẽ được tô màu tùy theo tên node:
    - Nếu chứa từ 'user' → màu sáng hơn.
    - Ngược lại → màu tối hơn.

    Args:
        g (networkx.Graph): Đồ thị cần vẽ.
        location (str): Đường dẫn lưu ảnh vẽ đồ thị (ví dụ: "output/graph.png").
    """
    plt.figure(figsize=(12, 8))

    # Gán màu cho từng node:
    # 'user' → 0.0 (sáng), còn lại → 0.5 (tối)
    node_colors = {
        node: 0.0 if 'user' in str(node) else 0.5
        for node in g.nodes()
    }

    nx.draw(
        g,
        pos=nx.spring_layout(g),       
        node_size=10000, 
        node_color=list(node_colors.values()),
        with_labels=True, 
        font_size=14,
        font_color='white'
    )
    plt.savefig(location, bbox_inches='tight')
