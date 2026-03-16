# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QGridLayout,
    QSizePolicy,
    QFrame,
    QProgressBar,
)


class MonitorToolbar(QGroupBox):
    """顶部工具栏组件：摄像头控制 + 阈值 + 快捷动作"""

    def __init__(self, cameras, threshold_value: int, on_start, on_stop, on_threshold, on_refresh, on_emergency):
        super().__init__("监控控制")
        layout = QHBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 14, 16, 14)

        layout.addWidget(QLabel("摄像头"))
        self.camera_combo = QComboBox()
        for cam_id, cam_config in cameras.items():
            self.camera_combo.addItem(f"摄像头 {cam_id}: {cam_config.name}", cam_id)
        self.camera_combo.setMinimumWidth(260)
        layout.addWidget(self.camera_combo)

        self.start_btn = QPushButton("启动摄像头")
        self.start_btn.clicked.connect(on_start)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止摄像头")
        self.stop_btn.setObjectName("secondaryButton")
        self.stop_btn.clicked.connect(on_stop)
        layout.addWidget(self.stop_btn)

        threshold_wrap = QVBoxLayout()
        threshold_title = QLabel("识别阈值")
        threshold_title.setObjectName("fieldTitle")
        threshold_wrap.addWidget(threshold_title)

        row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(threshold_value)
        self.threshold_slider.valueChanged.connect(on_threshold)
        row.addWidget(self.threshold_slider, 1)

        self.threshold_value_label = QLabel(f"{threshold_value / 100:.2f}")
        self.threshold_value_label.setObjectName("keyMetric")
        row.addWidget(self.threshold_value_label)

        threshold_wrap.addLayout(row)
        layout.addLayout(threshold_wrap, 1)

        self.quick_refresh_btn = QPushButton("快速刷新")
        self.quick_refresh_btn.setObjectName("secondaryButton")
        self.quick_refresh_btn.clicked.connect(on_refresh)
        layout.addWidget(self.quick_refresh_btn)

        self.emergency_btn = QPushButton("应急模式")
        self.emergency_btn.setObjectName("warningButton")
        self.emergency_btn.clicked.connect(on_emergency)
        layout.addWidget(self.emergency_btn)


class CameraDisplayArea(QWidget):
    """主摄像头显示区组件"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(18)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)


class SummaryCardsWidget(QGroupBox):
    """系统摘要统计卡片"""

    def __init__(self):
        super().__init__("系统摘要")
        self.online_value = QLabel("0/0")
        self.known_value = QLabel("0")
        self.alert_value = QLabel("0")
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.hide()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(self._build_card("在线摄像头", self.online_value))
        layout.addWidget(self._build_card("已知人脸", self.known_value))
        layout.addWidget(self._build_card("最近告警", self.alert_value))
        layout.addWidget(self.loading_bar)

    @staticmethod
    def _build_card(title: str, value_label: QLabel) -> QWidget:
        card = QFrame()
        card.setObjectName("summaryCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        value_label.setObjectName("cardValue")
        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        return card


class AlertListWidget(QGroupBox):
    """最新告警列表（固定表头 + 空数据提示）"""

    def __init__(self):
        super().__init__("最新告警")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["时间", "人员", "摄像头", "置信度", "状态"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionsClickable(False)

        self.table.setWordWrap(False)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setShowGrid(False)

        layout.addWidget(self.table)

    def show_empty(self):
        self.table.setRowCount(1)
        self.table.setSpan(0, 0, 1, self.table.columnCount())
        empty_item = QTableWidgetItem("暂无告警记录")
        empty_item.setFlags(Qt.ItemIsEnabled)
        empty_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(0, 0, empty_item)


class StatusNoticeWidget(QFrame):
    """状态提示区域"""

    def __init__(self, on_clear):
        super().__init__()
        self.setObjectName("statusNotice")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)

        self.notice_label = QLabel("系统运行正常，暂无未处理异常。")
        self.notice_label.setWordWrap(True)
        layout.addWidget(self.notice_label, 1)

        self.clear_btn = QPushButton("清除提示")
        self.clear_btn.setObjectName("secondaryButton")
        self.clear_btn.clicked.connect(on_clear)
        layout.addWidget(self.clear_btn)
