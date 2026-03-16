# -*- coding: utf-8 -*-

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Tuple

import numpy as np
from loguru import logger
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.alert_system import AlertSystem
from core.camera_manager import CameraManager
from core.database import FaceDatabase
from core.face_detection import FaceDetector
from core.utils import draw_face_info, numpy_to_pixmap
from .alert_panel import AlertPanel
from .components import (
    AlertListWidget,
    CameraDisplayArea,
    MonitorToolbar,
    StatusNoticeWidget,
    SummaryCardsWidget,
)
from .face_manager import FaceManagerDialog
from .history_viewer import HistoryViewer


class MainWindow(QMainWindow):
    FONT_SIZE_OPTIONS = {"中": 18, "大": 22, "特大": 26}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle(f"{config['app']['name']} v{config['app']['version']}")
        self.setWindowIcon(QIcon(config['app']['logo']))
        self.setMinimumSize(1366, 820)
        self.resize(1920, 1080)

        self.processing_interval = 2.0
        self.base_font_size = 22
        self.is_emergency_mode = False

        self.face_detector = FaceDetector(config)
        self.camera_manager = CameraManager('config/camera_config.yaml')
        self.alert_system = AlertSystem(config)
        self.database = FaceDatabase(config['app']['database_path'])

        max_workers = max(1, len(self.camera_manager.cameras))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_futures: Dict[int, Future] = {}
        self.latest_processed_frames: Dict[int, np.ndarray] = {}

        self.face_detector.load_known_faces(config['app']['known_faces_dir'])

        self.init_ui()
        self.camera_manager.start_all_cameras()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)
        self.last_processed: Dict[int, float] = {}

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)

        self.tab_widget = QTabWidget()
        self.tab_widget.tabBar().setExpanding(True)
        main_layout.addWidget(self.tab_widget)

        self.setup_monitor_tab()
        self.setup_controls_tab()
        self.setup_history_tab()

        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addPermanentWidget(self.status_label)

        self.setup_menu_bar()
        self.apply_styles()

    def setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        exit_action = file_menu.addAction('退出')
        exit_action.triggered.connect(self.close)

        tools_menu = menubar.addMenu('工具')
        tools_menu.addAction('人脸管理').triggered.connect(self.open_face_manager)
        tools_menu.addAction('告警面板').triggered.connect(self.open_alert_panel)

        view_menu = menubar.addMenu('视图')
        view_menu.addAction('切换全屏').triggered.connect(self.toggle_fullscreen)

    def setup_monitor_tab(self):
        monitor_tab = QWidget()
        self.tab_widget.addTab(monitor_tab, "监控")
        layout = QVBoxLayout(monitor_tab)
        layout.setSpacing(16)
        layout.setContentsMargins(18, 18, 18, 18)

        threshold_value = int(self.config['recognition']['recognition_threshold'] * 100)
        self.monitor_toolbar = MonitorToolbar(
            self.camera_manager.cameras,
            threshold_value,
            self.start_selected_camera,
            self.stop_selected_camera,
            self.update_threshold,
            self.handle_quick_refresh,
            self.handle_emergency_mode,
        )
        layout.addWidget(self.monitor_toolbar)

        self.camera_combo = self.monitor_toolbar.camera_combo
        self.threshold_slider = self.monitor_toolbar.threshold_slider
        self.threshold_value = self.monitor_toolbar.threshold_value_label
        self.start_btn = self.monitor_toolbar.start_btn
        self.stop_btn = self.monitor_toolbar.stop_btn
        self.quick_refresh_btn = self.monitor_toolbar.quick_refresh_btn
        self.emergency_btn = self.monitor_toolbar.emergency_btn

        content_layout = QHBoxLayout()
        content_layout.setSpacing(18)

        self.camera_area = CameraDisplayArea()
        self.camera_grid = self.camera_area.grid
        content_layout.addWidget(self.camera_area, 7)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(14)
        self.summary_widget = SummaryCardsWidget()
        self.loading_bar = self.summary_widget.loading_bar
        right_panel.addWidget(self.summary_widget)

        self.alert_widget = AlertListWidget()
        self.alert_table = self.alert_widget.table
        right_panel.addWidget(self.alert_widget, 1)

        content_layout.addLayout(right_panel, 3)
        layout.addLayout(content_layout, 1)

        self.status_notice = StatusNoticeWidget(self.reset_notice)
        self.notice_label = self.status_notice.notice_label
        self.clear_notice_btn = self.status_notice.clear_btn
        layout.addWidget(self.status_notice)

        self.max_display_cameras = 4
        self.camera_labels = {}
        self.build_camera_displays()

    def reset_notice(self):
        self.notice_label.setText("系统运行正常，暂无未处理异常。")

    def build_camera_displays(self):
        for i in reversed(range(self.camera_grid.count())):
            item = self.camera_grid.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.camera_labels.clear()
        camera_ids = [cid for cid, cam in self.camera_manager.cameras.items() if cam.enabled]
        if not camera_ids:
            camera_ids = list(self.camera_manager.cameras.keys())
        display_ids = camera_ids[:self.max_display_cameras]

        if not display_ids:
            placeholder = QLabel("未配置摄像头")
            placeholder.setAlignment(Qt.AlignCenter)
            self.camera_grid.addWidget(placeholder, 0, 0)
            return

        columns = 2 if len(display_ids) > 1 else 1
        for index, cam_id in enumerate(display_ids):
            frame = QFrame()
            frame.setObjectName("cameraDisplay")
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.setSpacing(0)

            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(360, 240)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setObjectName("cameraFeed")
            frame_layout.addWidget(label)

            overlay = QLabel(self.camera_manager.cameras[cam_id].name, label)
            overlay.setObjectName("cameraOverlay")
            overlay.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            overlay.setMargin(10)
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents)

            self.camera_labels[cam_id] = label
            self.camera_grid.addWidget(frame, index // columns, index % columns)

        rows = (len(display_ids) + columns - 1) // columns
        for row in range(rows):
            self.camera_grid.setRowStretch(row, 1)
        for col in range(columns):
            self.camera_grid.setColumnStretch(col, 1)

    def setup_controls_tab(self):
        controls_tab = QWidget()
        self.tab_widget.addTab(controls_tab, "控制")
        layout = QVBoxLayout(controls_tab)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout(camera_group)
        self.ctrl_camera_combo = QComboBox()
        for cam_id, cam_config in self.camera_manager.cameras.items():
            self.ctrl_camera_combo.addItem(f"摄像头 {cam_id}: {cam_config.name}", cam_id)
        camera_layout.addWidget(self.ctrl_camera_combo)

        btn_layout = QHBoxLayout()
        self.ctrl_start_btn = QPushButton("启动摄像头")
        self.ctrl_start_btn.clicked.connect(self.start_selected_camera)
        btn_layout.addWidget(self.ctrl_start_btn)
        self.ctrl_stop_btn = QPushButton("停止摄像头")
        self.ctrl_stop_btn.setObjectName("secondaryButton")
        self.ctrl_stop_btn.clicked.connect(self.stop_selected_camera)
        btn_layout.addWidget(self.ctrl_stop_btn)
        camera_layout.addLayout(btn_layout)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("识别阈值："))
        self.ctrl_threshold_slider = QSlider(Qt.Horizontal)
        self.ctrl_threshold_slider.setRange(50, 100)
        self.ctrl_threshold_slider.setValue(int(self.config['recognition']['recognition_threshold'] * 100))
        self.ctrl_threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.ctrl_threshold_slider)
        self.ctrl_threshold_value = QLabel(f"{self.ctrl_threshold_slider.value() / 100:.2f}")
        self.ctrl_threshold_value.setObjectName("keyMetric")
        threshold_layout.addWidget(self.ctrl_threshold_value)
        camera_layout.addLayout(threshold_layout)
        layout.addWidget(camera_group)

        interval_group = QGroupBox("处理参数")
        interval_layout = QHBoxLayout(interval_group)
        interval_layout.addWidget(QLabel("处理间隔（毫秒）："))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 5000)
        self.interval_spin.setValue(int(self.processing_interval * 1000))
        self.interval_spin.valueChanged.connect(self.update_processing_interval)
        interval_layout.addWidget(self.interval_spin)
        layout.addWidget(interval_group)

        font_group = QGroupBox("显示设置")
        font_layout = QHBoxLayout(font_group)
        font_layout.addWidget(QLabel("字体大小："))
        self.font_size_combo = QComboBox()
        for label, size in self.FONT_SIZE_OPTIONS.items():
            self.font_size_combo.addItem(f"{label}（{size}px）", size)
        self.font_size_combo.currentIndexChanged.connect(self.update_font_size)
        default_index = self.font_size_combo.findData(self.base_font_size)
        if default_index >= 0:
            self.font_size_combo.setCurrentIndex(default_index)
        font_layout.addWidget(self.font_size_combo, 1)
        layout.addWidget(font_group)

        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        self.status_display = QLabel("正在加载状态...")
        self.status_display.setWordWrap(True)
        status_layout.addWidget(self.status_display)
        layout.addWidget(status_group)
        layout.addStretch(1)

    def setup_history_tab(self):
        self.history_viewer = HistoryViewer(self.database, self.config)
        self.tab_widget.addTab(self.history_viewer, "历史记录")

    def open_face_manager(self):
        dialog = FaceManagerDialog(self.face_detector, self.config['app']['known_faces_dir'])
        dialog.exec_()
        self.face_detector.load_known_faces(self.config['app']['known_faces_dir'])

    def open_alert_panel(self):
        AlertPanel(self.alert_system).exec_()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.notice_label.setText("已退出全屏模式。")
        else:
            self.showFullScreen()
            self.notice_label.setText("已进入全屏模式。")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            self.notice_label.setText("已退出全屏模式（Esc）。")
            event.accept()
            return
        super().keyPressEvent(event)

    def handle_quick_refresh(self):
        self.loading_bar.setVisible(True)
        self.notice_label.setText("正在刷新监控数据，请稍候...")
        QTimer.singleShot(650, self._finish_quick_refresh)

    def _finish_quick_refresh(self):
        self.loading_bar.setVisible(False)
        self.notice_label.setText(f"刷新完成：{QDateTime.currentDateTime().toString('HH:mm:ss')}")

    def handle_emergency_mode(self):
        mode_text = "关闭" if self.is_emergency_mode else "开启"
        reply = QMessageBox.warning(
            self,
            "应急模式",
            f"确定{mode_text}应急模式？\n应急模式将触发高优先级告警通知。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.notice_label.setText("已取消应急模式操作。")
            return

        self.is_emergency_mode = not self.is_emergency_mode
        if self.is_emergency_mode:
            self.emergency_btn.setText("应急模式（已开启）")
            self.notice_label.setText("应急模式已开启，请优先处理异常。")
        else:
            self.emergency_btn.setText("应急模式")
            self.notice_label.setText("应急模式已关闭。")

    def start_selected_camera(self):
        combo = self.focusWidget() if isinstance(self.focusWidget(), QComboBox) else None
        cam_source = combo if combo in (self.camera_combo, self.ctrl_camera_combo) else self.camera_combo
        cam_id = cam_source.currentData() if cam_source else None
        if cam_id is None:
            self.notice_label.setText("未找到可启动的摄像头。")
            return
        if self.camera_manager.start_camera(cam_id):
            self.status_label.setText(f"已启动摄像头 {cam_id}")
            self.notice_label.setText(f"摄像头 {cam_id} 启动成功。")
        else:
            self.notice_label.setText(f"摄像头 {cam_id} 启动失败，请检查设备连接。")

    def stop_selected_camera(self):
        combo = self.focusWidget() if isinstance(self.focusWidget(), QComboBox) else None
        cam_source = combo if combo in (self.camera_combo, self.ctrl_camera_combo) else self.camera_combo
        cam_id = cam_source.currentData() if cam_source else None
        if cam_id is None:
            self.notice_label.setText("未找到可停止的摄像头。")
            return

        reply = QMessageBox.question(self, "确认操作", f"确定停止摄像头 {cam_id} 吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            self.notice_label.setText("已取消停止操作。")
            return

        if self.camera_manager.stop_camera(cam_id):
            self.status_label.setText(f"已停止摄像头 {cam_id}")
            self.notice_label.setText(f"摄像头 {cam_id} 已停止。")
        else:
            self.notice_label.setText(f"摄像头 {cam_id} 停止失败。")

    def update_threshold(self, value):
        threshold = value / 100
        self.face_detector.recognition_threshold = threshold
        self.threshold_value.setText(f"{threshold:.2f}")
        self.ctrl_threshold_value.setText(f"{threshold:.2f}")
        if self.threshold_slider.value() != value:
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(value)
            self.threshold_slider.blockSignals(False)
        if self.ctrl_threshold_slider.value() != value:
            self.ctrl_threshold_slider.blockSignals(True)
            self.ctrl_threshold_slider.setValue(value)
            self.ctrl_threshold_slider.blockSignals(False)

    def update_processing_interval(self, value):
        self.processing_interval = value / 1000

    def update_font_size(self, _index):
        selected_size = self.font_size_combo.currentData()
        if selected_size is None:
            return
        self.base_font_size = int(selected_size)
        self.apply_styles()

    def update(self):
        try:
            frames = self.camera_manager.get_all_frames()
            for cam_id, frame in frames.items():
                if frame is None:
                    continue

                current_time = time.time()
                last_time = self.last_processed.get(cam_id, 0)
                future = self.processing_futures.get(cam_id)

                if future and future.done():
                    try:
                        processed_frame, _ = future.result()
                        self.latest_processed_frames[cam_id] = processed_frame
                    except Exception as e:
                        logger.error(f"摄像头 {cam_id} 处理结果获取失败: {e}")
                    finally:
                        self.processing_futures[cam_id] = None

                display_frame = self.latest_processed_frames.get(cam_id, frame)
                self.display_frame(cam_id, display_frame)

                if (current_time - last_time >= self.processing_interval and (self.processing_futures.get(cam_id) is None)):
                    self.processing_futures[cam_id] = self.executor.submit(self.process_frame, cam_id, frame.copy())
                    self.last_processed[cam_id] = current_time

            self.update_status()
        except Exception as e:
            logger.error(f"更新循环错误: {e}")
            self.status_label.setText(f"错误: {str(e)}")

    def process_frame(self, cam_id: int, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        alert_triggered = False
        try:
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return frame, False

            recognized_faces = self.face_detector.recognize_faces(faces)
            for face, known_face, confidence in recognized_faces:
                camera_name = self.camera_manager.cameras[cam_id].name
                if known_face:
                    frame = draw_face_info(frame, face.bbox, known_face.name, confidence, camera_name, face.age, face.gender, time.time())
                    alert_event = self.alert_system.trigger_alert(cam_id, camera_name, known_face.name, face, confidence, frame)
                else:
                    frame = draw_face_info(frame, face.bbox, "未知", confidence, camera_name, timestamp=time.time())
                    alert_event = self.alert_system.trigger_alert(cam_id, camera_name, "未知", face, confidence, frame)

                if (not getattr(alert_event, "is_cooldown", False) and getattr(alert_event, "screenshot_path", None)):
                    alert_triggered = True
                    self.database.log_face_event(alert_event)
        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
        return frame, alert_triggered

    def display_frame(self, cam_id: int, frame: np.ndarray):
        try:
            if frame is None:
                return
            target_label = self.camera_labels.get(cam_id)
            if target_label is None:
                return
            pixmap = numpy_to_pixmap(frame)
            if pixmap is None:
                return
            target_label.setPixmap(pixmap.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logger.error(f"显示帧错误: {e}")

    def update_status(self):
        try:
            status_text = []
            running_count = 0
            status_text.append("=== 摄像头状态 ===")
            for cam_id, cam_config in self.camera_manager.cameras.items():
                running = cam_id in self.camera_manager.capture_threads
                if running:
                    running_count += 1
                status_text.append(f"摄像头 {cam_id}（{cam_config.name}）：{'运行中' if running else '已停止'}")

            status_text.append("\n=== 人脸数据库 ===")
            face_count = len(self.face_detector.known_faces)
            status_text.append(f"已知人脸数量：{face_count}")

            status_text.append("\n=== 告警记录 ===")
            recent_alerts = self.alert_system.get_recent_alerts(30)
            if recent_alerts:
                for alert in recent_alerts[:5]:
                    t = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
                    status_text.append(f"{t}: {alert.face_name} 出现在 {alert.camera_name} (置信度: {alert.confidence:.2f})")
            else:
                status_text.append("暂无告警")

            self.status_display.setText("\n".join(status_text))
            self.summary_widget.online_value.setText(f"{running_count}/{len(self.camera_manager.cameras)}")
            self.summary_widget.known_value.setText(str(face_count))
            self.summary_widget.alert_value.setText(str(len(recent_alerts)))

            self.alert_table.setUpdatesEnabled(False)
            self.alert_table.clearSpans()
            if not recent_alerts:
                self.alert_widget.show_empty()
            else:
                self.alert_table.setRowCount(len(recent_alerts))
                for row, alert in enumerate(recent_alerts):
                    time_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
                    values = [time_str, alert.face_name, alert.camera_name, f"{alert.confidence:.2f}", "告警" if alert.face_name in ("未知", "Unknown") else "提示"]
                    for col, text in enumerate(values):
                        item = QTableWidgetItem(text)
                        item.setToolTip(text)
                        if col == 4 and values[4] == "告警":
                            item.setBackground(Qt.red)
                            item.setForeground(Qt.white)
                        self.alert_table.setItem(row, col, item)
                self.alert_table.resizeRowsToContents()
            self.alert_table.setUpdatesEnabled(True)
        except Exception as e:
            logger.error(f"更新状态失败: {e}")
            self.notice_label.setText(f"状态更新失败：{str(e)}")

    def closeEvent(self, event):
        try:
            self.camera_manager.stop_all_cameras()
            self.update_timer.stop()
            self.executor.shutdown(wait=False)
            event.accept()
        except Exception as e:
            logger.error(f"关闭程序时出错: {e}")
            event.accept()

    def apply_styles(self):
        base = self.base_font_size
        title = base + 8
        control_height = max(50, int(base * 2.2))

        self.setStyleSheet(f"""
        QMainWindow {{ background-color: #0f172a; color: #e5e7eb; }}
        QMenuBar, QStatusBar {{ background: #111827; color: #f9fafb; font-size: {base}px; }}
        QWidget {{ font-family: 'Microsoft YaHei UI', 'PingFang SC', 'Noto Sans CJK SC', sans-serif; }}
        QLabel {{ color: #f3f4f6; font-size: {base}px; }}
        QLabel#fieldTitle {{ color: #bfdbfe; font-size: {base}px; font-weight: 700; }}
        QLabel#keyMetric {{ font-size: {title}px; font-weight: 800; color: #fbbf24; min-width: 78px; }}

        QGroupBox {{
            border: 1px solid #334155; border-radius: 10px; margin-top: 14px; padding: 12px;
            font-size: {base + 2}px; font-weight: 700; color: #e2e8f0; background: #111827;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}

        QPushButton {{
            min-height: {control_height}px; min-width: 148px; border-radius: 8px; padding: 6px 14px;
            background-color: #2563eb; color: #ffffff; border: 1px solid #1d4ed8; font-size: {base}px;
        }}
        QPushButton#secondaryButton {{ background-color: #1f2937; border: 1px solid #475569; }}
        QPushButton#warningButton {{ background-color: #b91c1c; border: 2px solid #ef4444; font-weight: 800; }}
        QPushButton:focus {{ border: 3px solid #facc15; }}

        QComboBox, QSpinBox, QLineEdit {{
            min-height: {control_height}px; border: 1px solid #475569; border-radius: 8px;
            background: #0b1220; color: #f8fafc; padding: 0 10px; font-size: {base}px;
        }}

        QSlider::groove:horizontal {{ height: 12px; border-radius: 6px; background: #334155; }}
        QSlider::handle:horizontal {{ background: #fbbf24; border: 1px solid #f59e0b; width: 26px; margin: -7px 0; border-radius: 13px; }}

        QFrame#cameraDisplay {{ border: 1px solid #334155; border-radius: 10px; background: #020617; }}
        QLabel#cameraFeed {{ background-color: #020617; border: none; border-radius: 10px; }}
        QLabel#cameraOverlay {{ font-size: {base}px; font-weight: 700; color: #ffffff; background-color: rgba(15,23,42,0.65); border-radius: 6px; padding: 4px 10px; }}

        QTableWidget {{
            border: 1px solid #334155; border-radius: 8px; background: #0b1220; color: #e5e7eb;
            alternate-background-color: #111827; selection-background-color: #1d4ed8; font-size: {base}px;
        }}
        QHeaderView::section {{
            background: #1e293b; color: #f8fafc; padding: 10px; border: none; border-bottom: 1px solid #334155;
            font-weight: 800; min-height: {control_height + 8}px; font-size: {base}px;
        }}
        QTableWidget::item {{ padding: 10px; min-height: {control_height}px; }}

        QFrame#statusNotice {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; }}
        QFrame#summaryCard {{ background: #0b1220; border: 1px solid #334155; border-radius: 10px; }}
        QLabel#cardTitle {{ font-size: {base}px; color: #93c5fd; font-weight: 700; }}
        QLabel#cardValue {{ font-size: {title + 2}px; color: #f8fafc; font-weight: 800; }}
        """)
