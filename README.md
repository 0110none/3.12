# 多摄像头人脸跟踪系统 🚀

<div align="center">
  <img src="https://raw.githubusercontent.com/AarambhDevHub/multi-cam-face-tracker/main/assets/logo.png" alt="Logo" width="200" height="200">
</div>

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![InsightFace](https://img.shields.io/badge/ML-InsightFace-orange)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red)
![Telegram](https://img.shields.io/badge/Alerts-Telegram-blue)

一套稳定的实时人脸跟踪系统，支持多摄像头监控、人脸识别、年龄与性别检测，并提供包含 Telegram 在内的多种智能告警能力。

## 🌟 核心特性

### 功能概览
- **多摄像头支持**：同时监听多路视频源（摄像头、RTSP 流、视频文件）
- **实时人脸检测**：基于 InsightFace，可选 GPU 加速
- **人脸识别**：识别已知人员，可配置置信度阈值
- **年龄与性别检测**：为每张人脸输出估计结果

### 告警能力
- ✨ **Telegram 通知**：在手机上实时接收截图与事件信息
- 🔔 **可定制声音与弹窗**：按需启用本地提醒
- 📸 **自动取证**：识别事件自动保存截图
- 📊 **完整日志**：包含时间戳与置信度的详细记录

### 图形界面
- 🖥️ **实时监控面板**：多路画面实时展示
- 👤 **人脸库管理**：上传、删除、更新已知人脸
- ⏱️ **历史记录浏览器**：按日期、摄像头或人员过滤，导出 CSV 报表并清理历史

## 🛠️ 技术栈

| 模块                   | 技术                     |
|------------------------|--------------------------|
| 人脸检测               | InsightFace              |
| 机器学习               | PyTorch                  |
| 计算机视觉             | OpenCV                   |
| GUI 框架               | PyQt5                    |
| 数据库                 | SQLite                   |
| 声音告警               | Pygame                   |
| Telegram 告警          | python-telegram-bot      |
| **告警渠道**           | Telegram Bot             |
| 人口统计               | InsightFace 年龄/性别估计 |

## 📦 安装指南

### 前置条件
- Python 3.8+
- 推荐具备 NVIDIA GPU（可选）
- FFmpeg（用于 RTSP/RTMP 流）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/AarambhDevHub/multi-cam-face-tracker.git
   cd multi-cam-face-tracker
   ```

2. **创建虚拟环境**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置系统**
    - 编辑 `config/config.yaml` 设置应用参数
    - 编辑 `config/camera_config.yaml` 配置摄像头列表

5. **准备目录**
    ```bash
    mkdir -p data/{known_faces,screenshots} config logs
    ```

6. **运行程序**
    ```bash
    python main.py
    ```

## ⚙️ 配置说明

### 应用配置（`config/config.yaml`）
```yaml
app:
  name: "Multi-Cam Face Tracker"
  version: "1.0.0"
  threshold: 0.6
  screenshot_dir: "data/screenshots"
  known_faces_dir: "data/known_faces"
  database_path: "data/database.db"
  alert_sound: "assets/alert.wav"
  log_dir: "logs"
  export_dir: "data/exports"
recognition:
  detection_threshold: 0.5
  recognition_threshold: 0.6
  max_batch_size: 8
  device: "cpu"  # 或 "cuda"
  age_estimation: true
  gender_detection: true
```

### 摄像头配置（`config/camera_config.yaml`）
```yaml
cameras:
  - id: 0
    name: "Front Camera"
    source: 0  # 摄像头索引或 RTSP 地址
    enabled: true
    resolution:
      width: 1280
      height: 720
    fps: 30
    rotate: 0 # 旋转角度 (0,90,180,270)
```

## ⚙️ Telegram 配置
### 在 `config/config.yaml` 添加：
```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"  # 来自 @BotFather
  chat_id: "YOUR_CHAT_ID"      # 可通过 @getidsbot 获取
  rate_limit: 30  # 告警间隔（秒）
```
### 开启步骤
- 使用 @BotFather 创建 Bot
- 通过 @getidsbot 获取聊天 ID
- 将 Bot 设为告警群/频道管理员
- 在配置中启用并重启应用

## 🔍 深入了解
想进一步了解架构与部署经验？欢迎查阅 DeepWiki：
👉 [https://deepwiki.com/AarambhDevHub/multi-cam-face-tracker](https://deepwiki.com/AarambhDevHub/multi-cam-face-tracker)

## 🖥️ 用户手册
### 添加已知人脸
1. 在工具菜单点击「人脸管理」。
2. 选择「添加人脸」，上传清晰正面照片。
3. 填写姓名后保存，系统会提取特征写入数据库。

### 摄像头控制
| 按钮 | 功能 |
| --- | --- |
| ▶️ 启动 | 打开选中摄像头数据流 |
| ⏹️ 停止 | 关闭摄像头处理 |
| ⚙️ 设置 | 调整分辨率/FPS |

### 告警管理
- 在「告警面板」里选择声音文件并调节音量。
- 设置最小置信度阈值（0.5-1.0）。
- 可启用/禁用截图保存。
- 在识别结果旁查看年龄与性别信息。

### 历史记录
1. 打开「历史」标签查看已记录的识别事件。
2. 使用日期、摄像头、人员过滤列表。
3. 点击「选择保存位置」指定导出目录，再用「导出记录」导出筛选结果。
4. 选择「清除历史」可删除筛选结果或全部记录。

## 🚀 性能优化建议
1. **RTSP 流**：
    - 优先使用 TCP 传输以提升稳定性。
    - 示例：`rtsp://user:pass@ip:port/stream?tcp`。

2. **GPU 加速**：
    ```yaml
    recognition:
        device: "cuda"  # 配置在 config.yaml
    ```

3. **性能调优**：
    - 摄像头较少时可适当降低处理间隔。
    - 远距离识别可降低分辨率以减轻负载。
    - RTSP 流建议启用 JPEG 压缩。

## 📊 场景示例
- 办公安防：监控门禁，识别异常人员。
- 智能家居：家庭成员到达时推送提醒。
- 零售分析：统计顾客年龄与性别结构。

## 📜 许可证
MIT License，详见 [LICENSE](LICENSE)。

## 🤝 贡献
欢迎参与贡献！请查阅[贡献指南](CONTRIBUTING.md)。

## ☕ 支持项目
如果本项目对你有帮助，欢迎请作者喝杯咖啡！
[Buy Me a Coffee](https://buymeacoffee.com/aarambhdevhub)
