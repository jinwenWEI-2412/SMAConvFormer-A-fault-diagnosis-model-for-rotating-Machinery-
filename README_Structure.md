# 项目结构说明

## 文件夹组织

### 📁 `models/`
包含所有主要的故障诊断模型架构：
- `SMAConvformer.py` - 主模型
- `Convformer_NSE.py` - ConvFormer变体
- `CLFormer.py` - 另一变体
- `EWSNet.py`
- `Liconvformer.py`
- `MCSwinT.py`
- `SCSA.py`

### 📁 `baselines/`
包含用于对比的基准模型：
- `ResNet18.py`
- `MobileNet.py`
- `MobileNetV2.py`
- `MK_ResCNN.py`
- `BJTU_rao.py`

### 📁 `datasets/`
包含数据集加载和预处理模块：
- `BJTU_rao.py` - 北京交通大学轴承数据集
- `OU_bearing.py` - OU轴承数据集
- `XJTU_gearbox.py` - 西交齿轮箱数据集
- `XJTU_spurgear.py` - 西交直齿轮数据集
- `data_pre.py` - 数据预处理函数

### 📁 `config/`
包含配置和参数文件：
- `args_diagnosis.py` - 训练和评估参数配置

### 📁 `utils/`
包含工具函数和辅助类

### 📁 `tests/`
包含单元测试和测试工具

### 📁 `results/`
输出目录（在运行时自动创建）：
- `checkpoints/` - 模型检查点
- `logs/` - 训练日志
- `predictions/` - 预测结果

## 文件说明

- `train.py` - 主训练脚本
- `evaluate.py` - 评估脚本
- `requirements.txt` - 项目依赖
- `.gitignore` - Git忽略文件配置
- `framework.jpg` - 模型框架图

## 使用方式

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 导入模型
```python
from models import SMAConvformer
from datasets import load_bjtu_rao
from config import get_args
```

### 3. 运行训练
```bash
python train.py
```

### 4. 运行评估
```bash
python evaluate.py
```
