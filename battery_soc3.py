"""
电池 SOC 轨迹分析模块
按小时平均 SOC 折线图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def hourly_socf(df: pd.DataFrame) -> pd.Series:
    """
    按小时计算平均 SOC
    :param df: 已清洗的 DataFrame（含 record_time、soc 列）
    :return: Series，索引=hour，值=mean_soc
    """
    return df.groupby('hour')['soc'].mean()


def _hourly_soc_line_fig(hourly_soc: pd.Series, font_prop=None) -> str:
    """
    绘制按小时平均 SOC 的折线图，返回 base64 编码字符串
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hourly_soc.index, hourly_soc.values,
            linewidth=2, color='#1890FF', marker='o', markersize=4)
    ax.set_title('电池使用轨迹（按小时平均 SOC）', fontproperties=font_prop, fontsize=18)
    ax.set_xlabel('小时 (Hour of Day)', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel('平均 SOC (%)', fontproperties=font_prop, fontsize=14)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    from battery_data_analysis2 import load_cleaned_data
    df = load_cleaned_data()
    hourly = hourly_socf(df)
    print("=== 按小时平均 SOC ===")
    print(hourly)
