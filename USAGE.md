# 使用说明

本说明面向需要部署与操作 **Multi-Camera Face Tracker System** 的工程师或运维同事，帮助快速完成环境准备、系统配置与日常操作。

## 1. 环境要求
- Python 3.8 及以上版本
- 建议具备 NVIDIA GPU（若无也可使用 CPU，实时性会下降）
- FFmpeg（用于拉取 RTSP/RTMP 流）
- 系统需具备网络访问权限，以便连接 Telegram Bot 或远程摄像头

## 2. 初始安装
1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/AarambhDevHub/multi-cam-face-tracker.git
   cd multi-cam-face-tracker
   ```
2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 初始化数据目录：
   ```bash
   mkdir -p data/{known_faces,screenshots,exports} logs
   ```

## 3. 配置文件
- `config/config.yaml`：应用基础配置（识别阈值、日志目录、告警参数等）
- `config/camera_config.yaml`：摄像头列表，可设置 `source`、分辨率、帧率及是否启用
- `config/telegram.yaml`（可选）：若需要 Telegram 告警，请在 `config.yaml` 中开启并写入 `bot_token`、`chat_id`

> 修改配置后需重新启动程序方能生效。

## 4. 启动与运行
1. 激活虚拟环境。
2. 执行 `python main.py` 启动 GUI。
3. 首次启动将自动创建 SQLite 数据库及必要的日志文件。

## 5. 图形界面操作
### 5.1 摄像头控制
- **选择摄像头**：左侧列表会显示 `camera_config.yaml` 中启用的所有摄像头。
- **开始/停止**：使用工具栏上的 ▶️ / ⏹️ 控件管理单个摄像头的数据流。
- **设置**：通过“设置”面板调整分辨率、FPS 及旋转角度。

### 5.2 人脸库管理
1. 打开“Face Manager”。
2. 选择“Add Face”并上传清晰正脸照。
3. 填写姓名并保存，系统会提取特征向量并写入数据库。
4. 可在列表中选择条目执行“Delete”或“Update”。

### 5.3 历史记录与导出
- “History Viewer” 支持按日期、摄像头、人员过滤事件。
- 点击“Export CSV”可导出当前筛选结果，默认保存在 `data/exports` 目录。
- “Purge Logs” 允许删除超过保留期的记录（操作不可逆）。

### 5.4 告警与通知
- 在“Alert Panel”中选择声音文件并调节音量；支持启用/禁用本地声音提示。
- 若开启 Telegram 告警，系统将在识别到名单人员后发送截图与基础信息。

## 6. 日志与排查
- **运行日志**：位于 `logs/`，关键文件为 `app.log`，包含摄像头状态与识别信息。
- **截图证据**：保存在 `data/screenshots/`，按日期分目录。
- **数据库**：`data/database.db`，可使用 SQLite 工具查看。

常见问题排查：
| 现象 | 可能原因 | 解决建议 |
|------|----------|----------|
| 摄像头无法启动 | RTSP 地址错误或账号无权限 | 使用 VLC 测试地址、确认网络可达 |
| 识别率低 | 阈值设置过高或样本不清晰 | 适当降低 `recognition_threshold`，重新采集高清人脸 |
| 无法发送 Telegram 告警 | `bot_token`/`chat_id` 配置错误 | 重新生成 Bot Token 并在配置中更新 |

## 7. 维护建议
- 定期备份 `data/` 与 `config/` 目录，防止数据库与配置丢失。
- 每季度检查依赖版本并运行 `pip list --outdated`，评估是否需要升级。
- 对 GPU 部署，监控显存使用情况并根据摄像头数量调整 `max_batch_size`。

如需进一步的架构说明或二次开发指南，请参考 `README.md` 及源码中的中文注释。
