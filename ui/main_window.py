# -*- coding: utf-8 -*-

import time
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTabWidget, QScrollArea, QGridLayout,
                             QComboBox, QSlider, QSpinBox, QFrame, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from loguru import logger
from typing import Dict, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future

# 导入核心模块
from core.face_detection import FaceDetector
from core.camera_manager import CameraManager
from core.alert_system import AlertSystem
from core.database import FaceDatabase
from core.utils import numpy_to_pixmap, draw_face_info

# 导入子界面模块
from .face_manager import FaceManagerDialog
from .alert_panel import AlertPanel
from .history_viewer import HistoryViewer


class MainWindow(QMainWindow):
    """
    系统主窗口（MainWindow）
    ------------------------
    负责整合摄像头、人脸识别、告警系统、数据库、历史记录等所有功能模块，
    提供统一的图形界面控制与监控视图。
    """

    def __init__(self, config):
        """初始化主界面及所有核心组件"""
        super().__init__()
        self.config = config
        self.setWindowTitle(f"{config['app']['name']} v{config['app']['version']}")
        self.setWindowIcon(QIcon(config['app']['logo']))
        self.setGeometry(100, 100, 1200, 800)

        self.processing_interval = 2.0  # 图像处理间隔时间（秒）

        # --- 初始化核心组件 ---
        self.face_detector = FaceDetector(config)                         # 人脸检测与识别模块
        self.camera_manager = CameraManager('config/camera_config.yaml')  # 摄像头管理模块
        self.alert_system = AlertSystem(config)                            # 告警模块
        self.database = FaceDatabase(config['app']['database_path'])       # 数据库模块

        max_workers = max(1, len(self.camera_manager.cameras))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_futures: Dict[int, Future] = {}
        self.latest_processed_frames: Dict[int, np.ndarray] = {}

        # 加载已知人脸库
        self.face_detector.load_known_faces(config['app']['known_faces_dir'])

        # --- 初始化 UI ---
        self.init_ui()

        # 启动摄像头线程
        self.camera_manager.start_all_cameras()

        # 启动定时更新器（刷新画面）
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)  # 约 30 FPS

        # 每个摄像头上次处理时间记录
        self.last_processed: Dict[int, float] = {}

    # ------------------------------
    # 初始化与 UI 构建部分
    # ------------------------------
    def init_ui(self):
        """设置主界面布局：Tab页 + 状态栏 + 菜单栏"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标签页（Tab）
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 添加三个功能页
        self.setup_monitor_tab()   # 摄像头监控界面
        self.setup_controls_tab()  # 控制参数界面
        self.setup_history_tab()   # 历史记录界面

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addPermanentWidget(self.status_label)

        # 菜单栏
        self.setup_menu_bar()

        # 统一应用样式
        self.apply_styles()

    def setup_menu_bar(self):
        """创建菜单栏（文件、工具、视图）"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')
        exit_action = file_menu.addAction('退出')
        exit_action.triggered.connect(self.close)

        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        face_manager_action = tools_menu.addAction('人脸管理')
        face_manager_action.triggered.connect(self.open_face_manager)
        alert_panel_action = tools_menu.addAction('告警面板')
        alert_panel_action.triggered.connect(self.open_alert_panel)

        # 视图菜单
        view_menu = menubar.addMenu('视图')
        fullscreen_action = view_menu.addAction('切换全屏')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)

    def setup_monitor_tab(self):
        """监控界面：显示摄像头实时画面"""
        monitor_tab = QWidget()
        self.tab_widget.addTab(monitor_tab, "监控")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # 摄像头画面容器
        self.camera_container = QWidget()
        self.camera_grid = QGridLayout(self.camera_container)
        self.camera_grid.setSpacing(10)
        self.camera_grid.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.camera_container)

        layout = QVBoxLayout(monitor_tab)
        title = QLabel("实时监控")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(title)
        layout.addWidget(scroll)

        # 摄像头显示
        self.max_display_cameras = 4
        self.camera_labels = {}
        self.build_camera_displays()

    def build_camera_displays(self):
        """根据摄像头数量创建（或更新）显示区域，最多 4 个"""
        # 清空旧控件
        for i in reversed(range(self.camera_grid.count())):
            item = self.camera_grid.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.camera_labels.clear()

        camera_ids = [
            cam_id for cam_id, cam in self.camera_manager.cameras.items() if cam.enabled
        ]
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
            label.setMinimumSize(240, 180)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setObjectName("cameraFeed")
            frame_layout.addWidget(label)

            overlay = QLabel(self.camera_manager.cameras[cam_id].name, label)
            overlay.setObjectName("cameraOverlay")
            overlay.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            overlay.setMargin(8)
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents)

            self.camera_labels[cam_id] = label

            row = index // columns
            col = index % columns
            self.camera_grid.addWidget(frame, row, col)

    def setup_controls_tab(self):
        """控制界面：用于调节识别阈值、处理间隔、摄像头启停等"""
        controls_tab = QWidget()
        self.tab_widget.addTab(controls_tab, "控制")
        layout = QVBoxLayout(controls_tab)

        # 摄像头控制区
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(12)

        self.camera_combo = QComboBox()
        for cam_id, cam_config in self.camera_manager.cameras.items():
            self.camera_combo.addItem(f"摄像头 {cam_id}: {cam_config.name}", cam_id)
        camera_layout.addWidget(self.camera_combo)

        # 启动/停止按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动摄像头")
        self.start_btn.clicked.connect(self.start_selected_camera)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止摄像头")
        self.stop_btn.clicked.connect(self.stop_selected_camera)
        btn_layout.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_layout)

        # 识别阈值控制
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("识别阈值：")
        threshold_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(int(self.config['recognition']['recognition_threshold'] * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value = QLabel(f"{self.threshold_slider.value() / 100:.2f}")
        threshold_layout.addWidget(self.threshold_value)
        camera_layout.addLayout(threshold_layout)

        layout.addWidget(camera_group)

        # 处理间隔控制
        interval_group = QGroupBox("处理参数")
        interval_layout = QHBoxLayout(interval_group)
        interval_layout.addWidget(QLabel("处理间隔（毫秒）："))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 5000)
        self.interval_spin.setValue(int(self.processing_interval * 1000))
        self.interval_spin.valueChanged.connect(self.update_processing_interval)
        interval_layout.addWidget(self.interval_spin)
        layout.addWidget(interval_group)

        # 系统状态显示
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)
        self.status_display = QLabel("正在加载状态...")
        self.status_display.setWordWrap(True)
        status_layout.addWidget(self.status_display)
        layout.addWidget(status_group)

    def setup_history_tab(self):
        """历史记录界面：整合 HistoryViewer 模块"""
        self.history_viewer = HistoryViewer(self.database, self.config)
        self.tab_widget.addTab(self.history_viewer, "历史记录")

    # ------------------------------
    # 菜单动作
    # ------------------------------
    def open_face_manager(self):
        """打开人脸管理窗口"""
        dialog = FaceManagerDialog(self.face_detector, self.config['app']['known_faces_dir'])
        dialog.exec_()
        self.face_detector.load_known_faces(self.config['app']['known_faces_dir'])

    def open_alert_panel(self):
        """打开告警管理面板"""
        dialog = AlertPanel(self.alert_system)
        dialog.exec_()

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ------------------------------
    # 摄像头控制与参数调节
    # ------------------------------
    def start_selected_camera(self):
        """启动所选摄像头"""
        cam_id = self.camera_combo.currentData()
        if self.camera_manager.start_camera(cam_id):
            self.status_label.setText(f"已启动摄像头 {cam_id}")

    def stop_selected_camera(self):
        """停止所选摄像头"""
        cam_id = self.camera_combo.currentData()
        if self.camera_manager.stop_camera(cam_id):
            self.status_label.setText(f"已停止摄像头 {cam_id}")

    def update_threshold(self, value):
        """调整识别置信度阈值"""
        threshold = value / 100
        self.face_detector.recognition_threshold = threshold
        self.threshold_value.setText(f"{threshold:.2f}")

    def update_processing_interval(self, value):
        """调整图像处理间隔"""
        self.processing_interval = value / 1000

    # ------------------------------
    # 主循环与图像处理
    # ------------------------------
    def update(self):
        """主循环（每30ms执行一次）：获取帧→识别→显示→更新状态"""
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

                if (current_time - last_time >= self.processing_interval and
                        (self.processing_futures.get(cam_id) is None)):
                    self.processing_futures[cam_id] = self.executor.submit(
                        self.process_frame, cam_id, frame.copy()
                    )
                    self.last_processed[cam_id] = current_time

            self.update_status()

        except Exception as e:
            logger.error(f"更新循环错误: {e}")
            self.status_label.setText(f"错误: {str(e)}")

    def process_frame(self, cam_id: int, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        图像识别流程：检测 → 识别 → 画框 → 触发告警

        逻辑：
        - 已知人：画框 + 触发“已知人”告警
        - 未知人：画框 + 触发“未知人”告警
        - 只有在「非冷却」并且「成功保存了截图」的情况下，才写入数据库
        """
        alert_triggered = False
        try:
            # 1. 检测人脸
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return frame, False

            # 2. 进行人脸识别（与已知人脸库比对）
            recognized_faces = self.face_detector.recognize_faces(faces)

            for face, known_face, confidence in recognized_faces:
                camera_name = self.camera_manager.cameras[cam_id].name

                # ---------- 已知人脸 ----------
                if known_face:
                    # 在画面上叠加信息（实时）
                    frame = draw_face_info(
                        frame,
                        face.bbox,
                        name=known_face.name,
                        confidence=confidence,
                        camera_name=camera_name,
                        age=face.age,
                        gender=face.gender,
                        timestamp=time.time()
                    )

                    # 触发“已知人”告警（AlertSystem 内部会处理冷却、截图、声音等）
                    alert_event = self.alert_system.trigger_alert(
                        cam_id,
                        camera_name,
                        known_face.name,
                        face,
                        confidence,
                        frame
                    )

                    # ✅ 只有满足：
                    #   1. 不是冷却事件（is_cooldown=False）
                    #   2. 截图路径存在（screenshot_path 不为 None / 空）
                    #   才写入数据库和认为“真正触发报警”
                    if (
                            not getattr(alert_event, "is_cooldown", False)
                            and getattr(alert_event, "screenshot_path", None)
                    ):
                        alert_triggered = True
                        self.database.log_face_event(alert_event)

                # ---------- 未知人脸 ----------
                else:
                    # 画面上显示为“未知”（同样是实时覆盖在画面上）
                    frame = draw_face_info(
                        frame,
                        face.bbox,
                        name="未知",
                        confidence=confidence,
                        camera_name=camera_name,
                        timestamp=time.time()
                    )

                    # 触发“未知人”告警：
                    # AlertSystem 中会根据 face_name == "未知" / "Unknown" 判断为陌生人，
                    # 使用陌生人警报音，并应用冷却逻辑。
                    alert_event = self.alert_system.trigger_alert(
                        cam_id,
                        camera_name,
                        "未知",  # face_name：用于区分陌生人
                        face,
                        confidence,
                        frame
                    )

                    # 未知人也遵守同样规则：只有非冷却 + 有截图 才写入数据库
                    if (
                            not getattr(alert_event, "is_cooldown", False)
                            and getattr(alert_event, "screenshot_path", None)
                    ):
                        alert_triggered = True
                        self.database.log_face_event(alert_event)

        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
        return frame, alert_triggered

    def display_frame(self, cam_id: int, frame: np.ndarray):
        """将处理后的画面显示到对应摄像头窗口"""
        try:
            if frame is None:
                return
            target_label = self.camera_labels.get(cam_id)
            if target_label is None:
                return
            pixmap = numpy_to_pixmap(frame)
            if pixmap is None:
                return
            scaled = pixmap.scaled(
                target_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            target_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"显示帧错误: {e}")

    def update_status(self):
        """更新系统状态信息：摄像头、人脸库、告警"""
        try:
            status_text = []

            # 摄像头状态
            status_text.append("=== 摄像头状态 ===")
            for cam_id, cam_config in self.camera_manager.cameras.items():
                running = cam_id in self.camera_manager.capture_threads
                status_text.append(
                    f"摄像头 {cam_id}（{cam_config.name}）：{'运行中' if running else '已停止'}"
                )

            # 已知人脸状态
            status_text.append("\n=== 人脸数据库 ===")
            status_text.append(f"已知人脸数量：{len(self.face_detector.known_faces)}")

            # 告警信息
            status_text.append("\n=== 告警记录 ===")
            recent_alerts = self.alert_system.get_recent_alerts(3)
            if recent_alerts:
                for alert in recent_alerts:
                    time_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
                    status_text.append(
                        f"{time_str}: {alert.face_name} 出现在 {alert.camera_name} "
                        f"(置信度: {alert.confidence:.2f})"
                    )
            else:
                status_text.append("暂无告警")

            self.status_display.setText("\n".join(status_text))

        except Exception as e:
            logger.error(f"更新状态失败: {e}")

    def closeEvent(self, event):
        """程序退出时释放资源：停止摄像头、定时器"""
        try:
            self.camera_manager.stop_all_cameras()
            self.update_timer.stop()
            self.executor.shutdown(wait=False)
            event.accept()
        except Exception as e:
            logger.error(f"关闭程序时出错: {e}")
            event.accept()

    def apply_styles(self):
        """统一设置应用的样式和色彩风格"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1f2233;
                color: #f4f5f7;
            }
            QLabel {
                color: #f4f5f7;
            }
            QLabel#sectionTitle {
                font-size: 20px;
                font-weight: 600;
                padding: 8px 0;
            }
            QLabel#cameraFeed {
                background-color: #0f1120;
                border: 1px solid #2b2f44;
                border-radius: 12px;
            }
            QLabel#cameraOverlay {
                font-size: 14px;
                font-weight: 600;
                color: #ffffff;
                background-color: rgba(0, 0, 0, 0.45);
                border-radius: 6px;
                padding: 4px 10px;
            }
            QTabWidget::pane {
                border: 1px solid #2b2f44;
                border-radius: 6px;
            }
            QTabBar::tab {
                padding: 10px 20px;
                background-color: transparent;
                color: #d0d3dc;
            }
            QTabBar::tab:selected {
                background-color: #2b2f44;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3f5c;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #50567a;
            }
            QPushButton:pressed {
                background-color: #2d324c;
            }
            QGroupBox {
                border: 1px solid #2b2f44;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: white;
            }
            QScrollArea {
                border: none;
            }
            QFrame#cameraDisplay {
                background-color: transparent;
            }
        """)
