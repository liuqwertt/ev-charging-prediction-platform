"""
电池 SOC 热力图分析模块
日期（行）× 小时（列）SOC 热力图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def hourly_soc_pivotf(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 日期×小时 的平均 SOC 透视表
    :param df: 已清洗的 DataFrame（含 record_time、soc 列）
    :return: DataFrame，行=date，列=hour，值=mean_soc
    """
    hourly_soc = df.groupby(['date', 'hour'])['soc'].mean().reset_index()
    return hourly_soc.pivot(index='date', columns='hour', values='soc')


def _heatmap_fig(pivot: pd.DataFrame, font_prop=None) -> str:
    """
    绘制 SOC 热力图（日期×小时），返回 base64 编码字符串
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(pivot, cmap='YlGnBu', linewidths=0.5,
                annot=True, fmt='.1f',
                cbar_kws={'label': '平均 SOC (%)'},
                ax=ax)
    ax.set_title('SOC 热力图（日期 × 小时）', fontproperties=font_prop, fontsize=20)
    ax.set_xlabel('小时 (Hour of Day)', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel('日期 (Date)', fontproperties=font_prop, fontsize=14)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    from battery_data_analysis2 import load_cleaned_data
    df = load_cleaned_data()
    pivot = hourly_soc_pivotf(df)
    print(f"=== SOC 热力图透视表 shape: {pivot.shape} ===")
    print(pivot.head())
