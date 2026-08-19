"""
电池失效分类模块
使用 RandomForestClassifier 对电池失效进行二分类（失效/正常），
在测试集上绘制 precision/recall/f1-score 柱状图。
"""
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def build_failure_model(df: pd.DataFrame):
    """
    构建电池失效分类模型
    :param df: 已清洗的 DataFrame（需含 soc、charge_current、available_energy 等列）
    :return: (X_test_scaled, y_test, y_pred) 三元组
    """
    df = df.copy()

    # 特征工程
    df['voltage_diff'] = df['max_cell_voltage (V)'] - df['min_cell_voltage (V)']
    df['temperature_diff'] = df['max_temperature (℃)'] - df['min_temperature (℃)']
    df['abs_current'] = df['charge_current (A)'].abs()

    # 目标标签：可用能量 < 5kW → 失效
    df['failure'] = df['available_energy (kw)'] < 5

    # 特征与标签
    feature_cols = ['soc', 'voltage_diff', 'temperature_diff', 'abs_current']
    X = df[feature_cols]
    y = df['failure']

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # 预测
    y_pred = rf.predict(X_test)

    return X_test, y_test, y_pred


def _classification_bar_fig(y_test, y_pred, font_prop=None):
    """
    绘制分类报告柱状图（precision/recall/f1-score per class），
    返回 base64 编码的 PNG 字符串。
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

    n_classes = len(precision)
    x_range = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2

    ax.bar(x_range, precision, bar_width, label='Precision', color='#1890FF')
    ax.bar(x_range + bar_width, recall, bar_width, label='Recall', color='#E6A23C')
    ax.bar(x_range + 2 * bar_width, f1, bar_width, label='F1-score', color='#67C23A')

    class_labels = [f'Class {i}' for i in range(n_classes)]
    ax.set_xticks(x_range + bar_width)
    ax.set_xticklabels(class_labels)
    ax.set_xlabel('Classes', fontsize=13)
    ax.set_ylabel('Scores', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.set_title('Classification Report', fontproperties=font_prop, fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    from battery_data_analysis2 import load_cleaned_data
    df = load_cleaned_data()
    X_test, y_test, y_pred = build_failure_model(df)
    img = _classification_bar_fig(y_test, y_pred)
    print(f"[battery_failure5] 分类报告柱状图生成成功，base64 长度: {len(img)}")
