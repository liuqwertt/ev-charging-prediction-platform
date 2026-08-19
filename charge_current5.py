"""
充电时间分布分析模块
按小时 / 日 / 月三个维度统计平均充电电流
充电事件定义为 charge_current (A) < 0
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def charge_time_by_hour_day_month(df: pd.DataFrame):
    """
    计算充电时间分布（小时/日/月平均充电电流）
    :param df: 已清洗的 DataFrame（含 charge_current (A) 列）
    :return: (hourly, daily, monthly) 三个 pd.Series
    """
    charging_df = df[df['charge_current (A)'] < 0].copy()
    hourly = charging_df.groupby('hour')['charge_current (A)'].mean()
    daily = charging_df.groupby('date')['charge_current (A)'].mean()
    # month 列不存在，从 record_time 派生
    charging_df['month'] = charging_df['record_time'].dt.to_period('M').astype(str)
    monthly = charging_df.groupby('month')['charge_current (A)'].mean()
    return hourly, daily, monthly


def _charging_time_fig(hourly, daily, monthly, font_prop=None) -> str:
    """
    绘制充电时间分布三子图，返回 base64 编码字符串
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))

    axes[0].plot(hourly.index, hourly.values,
                 linewidth=2, color='#1890FF', marker='o', markersize=3)
    axes[0].set_title('小时充电时间分布（平均充电电流）',
                      fontproperties=font_prop, fontsize=16)
    axes[0].set_xlabel('小时', fontproperties=font_prop, fontsize=12)
    axes[0].set_ylabel('平均电流 (A)', fontproperties=font_prop, fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(daily.index, daily.values,
                 linewidth=2, color='#722ED1', marker='s', markersize=3)
    axes[1].set_title('每日充电时间分布（平均充电电流）',
                      fontproperties=font_prop, fontsize=16)
    axes[1].set_xlabel('日期', fontproperties=font_prop, fontsize=12)
    axes[1].set_ylabel('平均电流 (A)', fontproperties=font_prop, fontsize=12)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(monthly.index, monthly.values,
                 linewidth=2, color='#52C41A', marker='^', markersize=5)
    axes[2].set_title('每月充电时间分布（平均充电电流）',
                      fontproperties=font_prop, fontsize=16)
    axes[2].set_xlabel('月份', fontproperties=font_prop, fontsize=12)
    axes[2].set_ylabel('平均电流 (A)', fontproperties=font_prop, fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    from battery_data_analysis2 import load_cleaned_data
    df = load_cleaned_data()
    h, d, m = charge_time_by_hour_day_month(df)
    print("=== 小时充电电流均值 ===")
    print(h)
    print("\n=== 日充电电流均值 ===")
    print(d)
    print("\n=== 月充电电流均值 ===")
    print(m)
