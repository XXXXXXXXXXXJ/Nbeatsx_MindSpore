# 🧠 N-BEATSx for Time Series Forecasting (MindSpore Version)

本项目为 N-BEATSx 模型的 MindSpore 实现版本，适用于多变量时间序列预测任务。原始模型来自 PyTorch，本项目完成了核心模块的迁移、GPU 支持和可复现训练、测试流程。

---

## 📁 项目结构说明

```bash
nbeatsx-main/
├── README.md                                                   # 项目说明文档（当前文件）
├── data/                                                       # 数据集文件目录
│       └── epf                
│           └── datasets                
│               ├── BE.csv                                      # BE数据集
│               ├── DE.csv                                      # DE数据集
│               ├── FR.csv                                      # FR数据集
│               ├── NP.csv                                      # NP数据集
│               └── PJM.csv                                     # PJM数据集

├── src/                                                        # 实验脚本目录
|      ├── nbeats                                               # 模型脚本目录
│               ├── nbeats_mindspore.py                         # N-BEATS 模型的 MindSpore 实现（核心结构模块）  
│               ├── nbeats_mindspore_main.py                    # 模型训练/预测的主运行脚本，封装高层逻辑接口
│               └── tcn_mindspore.py                            # TCN在 MindSpore 中的实现，用于 exogenous 模块
|      ├── results                                              # 模型输出结果目录
│               ├── BE                                          # BE数据集结果目录 
│                   └── nbeats_x 
|                           ├── hyperopt_20250614_0_0.p         # 模型超参输出结果
|                           └── result_test_20250614_0_0.p      # 模型运行结果    
│               └── NP                                          # NP数据集结果目录 
│                   └── nbeats_x 
|                           ├── hyperopt_20250614_0_0.p         # 模型超参输出结果
|                           └── result_test_20250614_0_0.p      # 模型运行结果
|      ├── utils                                                # 模型脚本目录
│               ├── data                                        # 数据加载与管理的底层支持模块 
│               ├── experiment            
│                           └── utils_experiment_ms1.py         # 实验辅助函数（训练/测试流程、指标计算）
│               ├── numpy                  
│                           └── metrics.py                      # 时间序列预测模型的误差评估指标 
│               └── pytorch                                     # BE数据集结果目录 
│                       ├── ts_dataset_ms.py                    # 数据集封装模块（滑窗处理）
│                       ├── ts_loader_ms.py                     # 数据加载器（MindSpore Dataset）
│                       └── losses_ms.py                        # 定义了基于 MindSpore 的时间序列预测损失函数
│      └── hyperopt_nbeatsx_mindspore.py                        # 超参数搜索脚本（基于 Hyperopt）
```

---

## ⚙️ 环境依赖

请使用以下环境配置以保证模型可运行：

### ✅ 硬件需求：

GPU: 1*Vnt1(32GB)|CPU: 8核 64GB

### 🧪 软件依赖：

mindspore_2_0:mindspore_2.0.0-cuda_11.6-py_3.9

## 🚀 运行说明

```bash
cd main/
python hyperopt_nbeatsx_mindspore.py --dataset 'NP' --space "nbeats_x" --data_augmentation 0 --random_validation 0 --n_val_weeks 52 --hyperopt_iters 2 --experiment_id "20250614_0_0"
```


## 📌 致谢与参考

- 原始模型论文：[N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting](https://arxiv.org/abs/1905.10437)
- 原始模型代码：https://github.com/cchallu/nbeatsx/tree/main
- 本项目基于 PyTorch 实现版本迁移至 MindSpore，适用于国产AI平台部署与推理。
