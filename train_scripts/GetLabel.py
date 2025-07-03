import os

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import os

# 提取所有的氨基酸残基的预测结果
def read_files_to_list(folder_path):
    all_values = []
    names = []
    for filename in os.listdir(folder_path):
        name = filename.split(".")[0]
        names.append(name)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                value = line.strip()  # 去除行尾的换行符等空白字符
                try:
                    value = float(value)  # 尝试将数值转换为浮点数
                except ValueError:
                    print(f"文件{filename}中存在无法转换为数值的行：{line}")
                    continue
                all_values.append(value)
    return all_values, names


# 根据预测的结果的名字，获取相应的蛋白质，并提取标签，返回一个list
def get_labels(folder_path, names):
    all_values = []
    for i in names:
        filepath = i + ".label"
        file_path = os.path.join(folder_path, filepath)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                value = [int(char) for char in line]
                all_values = all_values + value
    return all_values

def set_figure(y_test, y_prob):
    # 绘制ROC曲线
    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 保存为PDF文件
    plt.savefig('/ifs/home/dongyihan/protein/EG-ARG/roc_curve.pdf', format='pdf', bbox_inches='tight')

    # 显示图形
    plt.show()


def set_mutle_figure(y_test, y_prob):

    # 真实标签列表
    y_true = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]

    # 多个方法生成的预测概率列表
    y_probs = {
        'Method1': [0.1, 0.8, 0.3, 0.2, 0.9, 0.1, 0.4, 0.7, 0.6, 0.2],
        'Method2': [0.2, 0.7, 0.4, 0.3, 0.8, 0.2, 0.5, 0.6, 0.7, 0.3],
        'Method3': [0.1, 0.9, 0.2, 0.1, 0.8, 0.1, 0.3, 0.7, 0.5, 0.2]
    }

    # 存储每个方法的ROC曲线数据
    roc_data = []

    for method, y_prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data.append((method, fpr, tpr, roc_auc))

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))

    # 绘制每个方法的ROC曲线
    for method, fpr, tpr, roc_auc in roc_data:
        plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.2f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    # 保存为PDF文件
    plt.savefig('roc_curves.pdf', format='pdf', bbox_inches='tight')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    # 预测文件夹的位置
    folder_path = '/ifs/home/dongyihan/protein/EG-ARG/output_2'
    # 标签文件夹的位置
    label_path = '/ifs/home/dongyihan/protein/EG-ARG/Protein_train_data/label'
    predict_results, names = read_files_to_list(folder_path)
    labels = get_labels(label_path, names)
    print(len(predict_results))
    print(len(labels))

    pred = [1 if pred >= 0.5 else 0 for pred in predict_results]

    # 计算准确率
    acc = accuracy_score(labels, pred)
    print(f"Accuracy: {acc}")

    # 计算AUC
    # 注意：计算AUC时，需要预测概率，而不是预测标签
    auc = roc_auc_score(labels, predict_results)
    print(f"AUC: {auc}")

    # 计算召回率
    recall = recall_score(labels, pred)
    print(f"Recall: {recall}")

    # 计算精确度
    precision = precision_score(labels, pred)
    print(f"Precision: {precision}")

    # 计算f1
    f1 = f1_score(labels, pred)
    
    (labels, pred)
    print(f"f1_score: {f1}")

    # 计算马修斯相关系数
    mcc = matthews_corrcoef(labels, pred)
    print("马修斯相关系数:", mcc)

    max_f1 = 0
    max_epoch = 0
    import numpy as np

    # 生成从 0 到 1 之间，步长为 0.02 的浮点数序列
    values = np.arange(0, 1, 0.01)

    # 打印生成的序列
    for i in values:
        pred = [1 if pred >= i else 0 for pred in predict_results]
        f1 = f1_score(labels, pred)
        precision = precision_score(labels, pred)
        mcc = matthews_corrcoef(labels, pred)
        acc = accuracy_score(labels, pred)
        auc = roc_auc_score(labels, predict_results)
        if f1 > max_f1:
            max_f1 = f1
            max_epoch = i
    print(max_f1)
    print(max_epoch)
    pred = [1 if pred >= max_epoch else 0 for pred in predict_results]
        # 计算准确率
    acc = accuracy_score(labels, pred)
    print(f"Accuracy: {acc}")

    # 计算AUC
    # 注意：计算AUC时，需要预测概率，而不是预测标签
    auc = roc_auc_score(labels, predict_results)
    print(f"AUC: {auc}")

    # 计算召回率
    recall = recall_score(labels, pred)
    print(f"Recall: {recall}")

    # 计算精确度
    precision = precision_score(labels, pred)
    print(f"Precision: {precision}")

    # 计算f1
    f1 = f1_score(labels, pred)
    
    (labels, pred)
    print(f"f1_score: {f1}")

    # 计算马修斯相关系数
    mcc = matthews_corrcoef(labels, pred)
    print("马修斯相关系数:", mcc)