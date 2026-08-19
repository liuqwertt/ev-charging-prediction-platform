# 新能源充电桩数据可视化与预测平台

基于 Flask 的充电桩数据分析与可视化 Web 应用，提供机器学习预测功能。

> A Flask-based web platform for EV charging pile data visualization and machine learning prediction (SOC, charging time, charging fee, and charging platform).

## 项目结构

```
ev-charging-prediction-platform/
├── app.py                      # Flask 主应用（含全部路由和预测 API）
├── pyproject.toml              # 项目依赖（uv 管理）
├── uv.lock                     # 依赖锁定文件
├── run_train.sh                # 训练脚本（Linux）
├── run_train.ps1               # 训练脚本（Windows）
├── com/neu/
│   ├── Deshdfs.py              # HDFS 电池数据下载与清洗
│   └── Nvv2thdfs.py            # HDFS 充电桩数据下载与清洗
├── data/                       # 数据和模型目录
│   ├── dsv13r2.csv             # 电池数据（可从 HDFS 下载）
│   ├── nvv2t.csv               # 充电桩数据（可从 HDFS 下载）
│   ├── soc_model.pkl           # SOC 预测模型
│   ├── time_model.pkl          # 充电时间预测模型
│   ├── fee_model.pkl           # 充电费用预测模型
│   ├── platform_model.pkl      # 平台分类模型
│   └── scaler.pkl              # 标准化器
├── templates/
│   ├── base.html               # 主布局
│   ├── _sidebar.html           # 侧边栏导航
│   ├── _topbar.html            # 顶部栏
│   ├── page4_soc.html          # SOC 预测页面
│   ├── page4_time.html         # 充电时间预测页面
│   ├── page4_fee.html          # 充电费用预测页面
│   └── page4_platform.html     # 平台选择预测页面
└── static/
    ├── css/custom.css          # 自定义样式
    ├── js/custom.js            # 自定义脚本
    └── images/                 # 图片资源
```

> 训练脚本（`battery_*.py`、`charge_*.py`、`ds_battery72_1.py`、`nvv_ds_*.py`）位于项目根目录，用于生成 `data/` 下的模型文件。

## 依赖环境

| 依赖 | 版本 | 说明 |
|------|------|------|
| Flask | >=3.1.3 | Web 框架 |
| pandas | >=3.0.3 | 数据处理 |
| scikit-learn | >=1.5.0 | 机器学习算法 |
| lightgbm | >=4.0.0 | 梯度提升框架 |
| joblib | >=1.3.0 | 模型序列化 |
| hdfs | >=2.7.3 | HDFS 数据读取 |
| matplotlib | >=3.11.0 | 图表绘制 |

推荐使用 `uv` 管理依赖。

## 部署步骤（Ubuntu）

### 1. 同步依赖

```bash
cd ev-charging-prediction-platform
uv sync
```

### 2. 下载数据（如需要）

```bash
# 下载电池数据
uv run com/neu/Deshdfs.py

# 下载充电桩数据
uv run com/neu/Nvv2thdfs.py
```

### 3. 训练模型

```bash
# SOC预测模型
uv run ds_battery72_1.py

# 充电时间预测模型
uv run nvv_ds_01_1.py

# 充电费用预测模型
uv run nvv_ds_02_1.py

# 平台选择预测模型
uv run nvv_ds_03_1.py
```

### 4. 启动 Flask 服务

```bash
uv run app.py
```

### 5. 访问预测页面

| 路由 | 页面说明 |
|------|----------|
| http://localhost:5000/ | 首页 |
| http://localhost:5000/predict_soc | 预测剩余电量 |
| http://localhost:5000/predict_time | 预测充电时间 |
| http://localhost:5000/predict_fee | 预测充电费用 |
| http://localhost:5000/predict_platform | 预测充电平台 |

## 模型说明

### SOC预测（ds_battery72_1.py）

- **算法**: 随机森林回归 + GridSearchCV超参数优化
- **数据集**: dsv13r2.csv（约5万条）
- **特征**: 组电压、充电电流、单体电压、温度、能量、容量
- **评估指标**: MAE, MSE, R2

### 充电时间预测（nvv_ds_01_1.py）

- **算法**: 线性回归
- **数据集**: nvv2t.csv（约100万条）
- **特征**: 充电量、费用、开始/结束时间、星期几（独热编码）
- **评估指标**: MSE, RMSE

### 充电费用预测（nvv_ds_02_1.py）

- **算法**: LightGBM回归
- **数据集**: nvv2t.csv
- **特征**: 充电量、充电时长、时间、星期几、设施类型
- **评估指标**: MSE对比（默认参数 vs 调参后）

### 平台选择预测（nvv_ds_03_1.py）

- **算法**: 随机森林分类
- **数据集**: nvv2t.csv
- **特征**: 充电量、费用、时长、用户ID、站ID、星期几
- **目标**: platform (0-iOS, 1-Android)
- **评估指标**: Accuracy, Classification Report

## 数据说明

- **HDFS 路径**:
  - 电池数据: `hdfs://192.168.79.136:9000/Car/dsv13r2.csv`
  - 充电桩数据: `hdfs://192.168.79.136:9000/Car/nvv2t.csv`
- **本地数据**: 放在 `data/` 目录下

## 常见问题

**Q: 模型文件不存在？**
A: 先运行数据下载脚本，然后训练模型脚本。

**Q: 页面样式异常？**
A: 确保 `static/` 目录完整。

**Q: 预测结果异常？**
A: 检查输入数据是否符合要求，确保模型已正确训练。

## License

仅供学习与课程实训使用。
