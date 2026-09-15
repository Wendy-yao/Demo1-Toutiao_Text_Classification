import numpy as np


# 计算整体准确率
def whole_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


# 计算每个类别的 TP（预测正确）、FN（预测不是实际是）、FP（误认为是）
def confusion_matrix(y_true, y_pred, num_labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.zeros(num_labels, dtype=np.int64)
    fp = np.zeros(num_labels, dtype=np.int64)
    fn = np.zeros(num_labels, dtype=np.int64)

    for c in range(num_labels):
        tp[c] = int(np.sum((y_true == c) & (y_pred == c)))
        fp[c] = int(np.sum((y_true != c) & (y_pred == c)))
        fn[c] = int(np.sum((y_true == c) & (y_pred != c)))

    return tp, fp, fn


# 返回每个类别的 precision、recall、f1、support
def eval_metric(y_true, y_pred, num_labels):
    tp, fp, fn = confusion_matrix(y_true, y_pred, num_labels)

    precision = np.zeros(num_labels, dtype=np.float64)
    recall = np.zeros(num_labels, dtype=np.float64)
    f1 = np.zeros(num_labels, dtype=np.float64)

    for c in range(num_labels):
        denom_p = tp[c] + fp[c]
        denom_r = tp[c] + fn[c]
        precision[c] = tp[c] / denom_p if denom_p > 0 else 0.0
        recall[c] = tp[c] / denom_r if denom_r > 0 else 0.0
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])

    support = tp + fn
    return precision, recall, f1, support


# 计算 Macro-F1
def Macro_f1(y_true, y_pred, num_labels):
    _, _, f1, _ = eval_metric(y_true, y_pred, num_labels)
    return float(np.mean(f1))


# 计算 Weighted-F1
def Weighted_f1(y_true, y_pred, num_labels):
    _, _, f1, support = eval_metric(y_true, y_pred, num_labels)
    total = support.sum()
    if total == 0:
        return 0.0
    return float(np.sum(f1 * support) / total)


# 返回每个类别的 recall
def recall_every(y_true, y_pred, num_labels):
    _, recall, _, _ = eval_metric(y_true, y_pred, num_labels)
    return recall


# 汇总所有的评价指标
def all_metrics(y_true, y_pred, id2label, prefix: str = ""):
    num_labels = len(id2label)
    acc = whole_accuracy(y_true, y_pred)
    macro_f1 = Macro_f1(y_true, y_pred, num_labels)
    weighted_f1 = Weighted_f1(y_true, y_pred, num_labels)
    recalls = recall_every(y_true, y_pred, num_labels)

    metrics = {
        f"{prefix}accuracy": acc,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}weighted_f1": weighted_f1,
    }
    for label_id, name in id2label.items():
        metrics[f"{prefix}recall_{label_id}_{name}"] = float(recalls[label_id])
    return metrics


# 打印详细评价分类报告
def print_report(y_true, y_pred, id2label):
    precision, recall, f1, support = eval_metric(y_true, y_pred, len(id2label))
    total = int(support.sum())

    lines = [f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}", ""]
    for c, name in id2label.items():
        lines.append(f"{name:>12} {precision[c]:>10.4f} {recall[c]:>10.4f} "
                     f"{f1[c]:>10.4f} {int(support[c]):>10d}")

    def avg(x):
        return float(np.sum(x * support) / total) if total else 0.0

    lines += [
        "",
        f"{'accuracy':>12} {'':>10} {'':>10} {whole_accuracy(y_true, y_pred):>10.4f} {total:>10d}",
        f"{'macro avg':>12} {np.mean(precision):>10.4f} {np.mean(recall):>10.4f} {np.mean(f1):>10.4f} {total:>10d}",
        f"{'weighted avg':>12} {avg(precision):>10.4f} {avg(recall):>10.4f} {avg(f1):>10.4f} {total:>10d}",
    ]
    return "\n".join(lines)
