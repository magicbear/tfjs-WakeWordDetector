# Custom Wake Word Detection

基于TensorFlow.js实现的实时唤醒词检测系统，支持自定义唤醒词训练和浏览器端实时检测。

## 功能特性
- 🎙️ 实时音频流处理与波形可视化
- 🔍 基于梅尔频谱特征的深度学习检测
- ⚙️ 动态阈值调节和检测日志记录
- 🧠 支持自定义唤醒词模型训练
- 🌐 纯浏览器端运行，无需后端服务

## 快速开始

### 使用预训练模型
1. 访问 `https://localhost:8000`
2. 点击"开始监听"授权麦克风访问
3. 实时查看音频波形和检测结果

## 模型训练

### 数据准备
```bash
数据集结构：
data/
├── your_wakeword/
│   ├── audio1.wav
│   └── audio2.wav
└── negative/
    └── background_noise.wav
```

### 训练新模型
```python
python kws_traindata.py --wake_word your_wakeword --epochs 20
```

## 技术栈
- **核心框架**: TensorFlow.js 3.18.0
- **音频处理**: Web Audio API + FFT.js
- **特征提取**: 梅尔频谱分析(MFCC)
- **可视化**: Plotly.js + Canvas API

## 开发文档

### 关键参数配置
```javascript
const WINDOW_SIZE = 200         // 分析窗口(ms) 
const SLIDE_WINDOW_SIZE = 50    // 滑动分析窗口(ms)
const MEL_SPEC_BINS = 40        // 梅尔频带数
const SPEC_HOP_LENGTH = 256
const NUM_FFTS = 512
const LOG_OFFSET = 1e-4         // Log偏移值

const wakeWord = "小丁";

```

### 实时处理流程
1. 音频采集 → 重采样至16kHz
2. 分帧加窗 → 计算梅尔频谱
3. 模型推理 → 动态阈值检测
4. 多阶段唤醒词验证

## 致谢
本项目参考了以下优秀资源：
- [WakeWordDetector](https://github.com/rajashekar/WakeWordDetector) - 基础检测框架
- [tensorflow-kws](https://github.com/yuyun2000/tensorflow-kws) - 模型训练方案

## 许可证
[MIT License](LICENSE)