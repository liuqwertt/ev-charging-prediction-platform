# 新能源充电桩 - 一键训练脚本 (Windows PowerShell)
# 使用方法: .\run_train.ps1

$ErrorActionPreference = "Stop"  # 遇到错误立即退出

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "新能源充电桩数据预测模型 - 一键训练脚本" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# 进入项目目录
Set-Location $PSScriptRoot

Write-Host "[1/7] 同步依赖..." -ForegroundColor Yellow
uv sync

Write-Host ""
Write-Host "[2/7] 下载电池数据..." -ForegroundColor Yellow
uv run com/neu/Deshdfs.py

Write-Host ""
Write-Host "[3/7] 下载充电桩数据..." -ForegroundColor Yellow
uv run com/neu/Nvv2thdfs.py

Write-Host ""
Write-Host "[4/7] 训练SOC预测模型..." -ForegroundColor Yellow
uv run ds_battery72_1.py

Write-Host ""
Write-Host "[5/7] 训练充电时间预测模型..." -ForegroundColor Yellow
uv run nvv_ds_01_1.py

Write-Host ""
Write-Host "[6/7] 训练充电费用预测模型..." -ForegroundColor Yellow
uv run nvv_ds_02_1.py

Write-Host ""
Write-Host "[7/7] 训练平台选择预测模型..." -ForegroundColor Yellow
uv run nvv_ds_03_1.py

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "全部训练完成!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "启动服务命令: uv run app.py" -ForegroundColor White
Write-Host "访问地址: http://localhost:5000" -ForegroundColor White
