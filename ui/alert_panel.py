# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                             QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt
import time


class AlertPanel(QDialog):
    """告警控制面板，用于查看、管理系统触发的人脸识别告警记录"""

    def __init__(self, alert_system):
        super().__init__()
        self.alert_system = alert_system
        self.setWindowTitle("告警面板")          # 中文标题
        self.setGeometry(300, 300, 600, 400)     # 窗口位置与大小

        self.init_ui()       # 初始化界面控件
        self.load_alerts()   # 加载警报记录列表

    def init_ui(self):
        """初始化界面控件：告警列表、复选框、按钮布局等"""
        layout = QVBoxLayout()

        # 告警记录列表（显示时间、人脸名、摄像头等）
        self.alert_list = QListWidget()
        layout.addWidget(self.alert_list)

        # 控制区（横向排列）
        controls_layout = QHBoxLayout()

        # 启用/禁用警报声音
        self.enable_alerts_check = QCheckBox("启用声音告警")
        self.enable_alerts_check.setChecked(self.alert_system.alert_enabled)
        self.enable_alerts_check.stateChanged.connect(self.toggle_alerts)
        controls_layout.addWidget(self.enable_alerts_check)

        # 启用/禁用截图保存
        self.enable_screenshots_check = QCheckBox("保存截图")
        self.enable_screenshots_check.setChecked(self.alert_system.screenshot_enabled)
        self.enable_screenshots_check.stateChanged.connect(self.toggle_screenshots)
        controls_layout.addWidget(self.enable_screenshots_check)

        # 清空告警按钮
        self.clear_btn = QPushButton("清空告警")
        self.clear_btn.clicked.connect(self.clear_alerts)
        controls_layout.addWidget(self.clear_btn)

        # 关闭窗口按钮
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.close_btn)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

    def load_alerts(self):
        """从告警系统中获取最近的 50 条告警记录并显示"""
        self.alert_list.clear()
        alerts = self.alert_system.get_recent_alerts(50)

        for alert in alerts:
            # 格式化时间
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))
            # 组合显示文本
            item_text = f"{time_str} - {alert.face_name} 出现在 {alert.camera_name} (置信度: {alert.confidence:.2f})"
            self.alert_list.addItem(item_text)

    def toggle_alerts(self, state):
        """启用或禁用声音告警"""
        self.alert_system.enable_alerts(state == Qt.Checked)

    def toggle_screenshots(self, state):
        """启用或禁用截图保存功能"""
        self.alert_system.enable_screenshots(state == Qt.Checked)

    def clear_alerts(self):
        """弹出确认框，清空所有告警记录"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有告警记录吗？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.alert_system.clear_alerts()
            self.alert_list.clear()
            QMessageBox.information(self, "完成", "所有告警记录已清空。")
