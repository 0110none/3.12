# -*- coding: utf-8 -*-

"""
多摄像头人脸追踪系统
--------------------------------
主程序入口，用于启动人脸识别与追踪应用。
初始化图形界面、摄像头流、人脸检测、识别、警报系统和数据库。

功能：
- 多摄像头实时人脸检测与识别
- 目标人脸匹配与警报提示
- 事件记录（带时间戳与截图）
- 基于 PyQt5 的桌面图形界面

"""

import sys
import yaml
from pathlib import Path
from loguru import logger
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

from ui.main_window import MainWindow

def load_config(config_path: str) -> dict:
    """
    从 YAML 文件加载配置
    - 自动创建截图、人脸库、日志目录
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        Path(config['app']['screenshot_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['app']['known_faces_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['app']['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['app'].get('export_dir', 'data/exports')).mkdir(parents=True, exist_ok=True)

        return config
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        raise

def setup_logging(log_dir: str):
    """
    设置日志系统（loguru）
    - 保存 info 和 error 两类日志
    - 自动轮换与保留历史
    """
    logger.add(
        f"{log_dir}/app.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    logger.add(
        f"{log_dir}/error.log",
        rotation="10 MB",
        retention="30 days",
        level="ERROR"
    )

def show_splash_screen(config: dict) -> QSplashScreen:
    """显示启动画面（Splash Screen）"""
    try:
        logo_path = config.get('app', {}).get('logo', 'assets/logo.png')
        if not Path(logo_path).exists():
            raise FileNotFoundError(f"找不到 Logo 文件: {logo_path}")

        splash_pix = QPixmap(logo_path)
        if splash_pix.isNull():
            raise ValueError(f"无效的图片文件: {logo_path}")

        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setMask(splash_pix.mask())

        splash.showMessage(
            "正在初始化人脸追踪系统...",
            Qt.AlignBottom | Qt.AlignCenter,
            Qt.white
        )
        QApplication.processEvents()
        return splash
    except Exception as e:
        print(f"加载启动画面出错: {e}")
        return QSplashScreen(QPixmap(800, 400))

def main():
    """
    主函数：
    - 加载配置
    - 设置日志
    - 启动 PyQt5 界面
    - 启动事件循环
    """
    try:
        config = load_config('config/config.yaml')
        setup_logging(config['app']['log_dir'])

        app = QApplication(sys.argv)
        splash = show_splash_screen(config)
        splash.show()
        app.processEvents()

        window = MainWindow(config)

        def show_ai_loading():
            splash.showMessage(
                "正在加载 AI 模型...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white
            )
            QApplication.processEvents()
            QTimer.singleShot(1500, lambda: splash.finish(window))

        QTimer.singleShot(2000, show_ai_loading)
        QTimer.singleShot(3500, lambda: window.show())

        logger.info("程序启动成功")

        def on_close():
            window.camera_manager.stop_all_cameras()
            window.alert_system.shutdown()
            app.quit()

        app.aboutToQuit.connect(on_close)
        sys.exit(app.exec_())

    except Exception as e:
        logger.critical(f"程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
