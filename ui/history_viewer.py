# -*- coding: utf-8 -*-

from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                             QLabel, QDateEdit, QComboBox, QSpacerItem, QSizePolicy,
                             QSplitter, QFrame, QMessageBox, QDialog, QFileDialog)
from PyQt5.QtCore import Qt, QDate
from loguru import logger
from datetime import datetime, timedelta
import cv2
import csv

from core.database import FaceLogEntry
from core.utils import numpy_to_pixmap


class HistoryViewer(QWidget):
    """
    历史记录查看器（HistoryViewer）
    --------------------------------
    在图形界面中查看系统识别记录，包括时间、摄像头、人脸名称、置信度等。
    支持按时间范围、人脸名、摄像头筛选，并支持查看对应截图。
    """

    def __init__(self, database, config):
        """初始化界面与数据"""
        super().__init__()
        self.database = database      # 数据库对象（FaceDatabase 实例）
        self.config = config
        self.current_entry = None     # 当前选中的记录项
        self.current_entries = []     # 当前筛选条件下的所有记录
        self.export_dir = self._resolve_export_directory()

        # 初始化界面与数据
        self.setup_ui()
        self.load_camera_list()
        self.load_face_list()
        self.refresh_history()

    def _resolve_export_directory(self) -> Path:
        """根据配置或默认值创建导出目录"""
        app_config = self.config.get('app', {}) if isinstance(self.config, dict) else {}
        export_dir = app_config.get('export_dir')
        if export_dir:
            export_path = Path(export_dir).expanduser()
        else:
            export_path = Path.home() / '历史记录导出'

        try:
            export_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"创建导出目录失败，使用用户目录: {exc}")
            export_path = Path.home()

        return export_path

    def _current_filter_conditions(self):
        """返回当前筛选条件对应的查询参数"""
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate() + timedelta(days=1)

        start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_timestamp = datetime.combine(end_date, datetime.min.time()).timestamp()

        return {
            'camera_id': self.camera_combo.currentData(),
            'face_name': self.face_combo.currentData(),
            'start_time': start_timestamp,
            'end_time': end_timestamp
        }

    def update_export_path_label(self):
        """同步界面上的导出目录标签，确保用户了解最终保存位置"""
        if hasattr(self, 'export_path_label'):
            display_path = str(self.export_dir)
            self.export_path_label.setText(display_path)
            # 通过 ToolTip 显示完整路径，便于查看长目录
            self.export_path_label.setToolTip(display_path)

    def setup_ui(self):
        """初始化界面组件：筛选区 + 历史列表 + 详情面板"""
        main_layout = QVBoxLayout()

        # --- 筛选条件区 ---
        filter_layout = QHBoxLayout()

        # 时间范围筛选
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("日期范围："))

        date_range_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-7))  # 默认最近 7 天
        self.start_date.setCalendarPopup(True)
        date_range_layout.addWidget(self.start_date)

        date_range_layout.addWidget(QLabel("至"))

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_range_layout.addWidget(self.end_date)

        date_layout.addLayout(date_range_layout)
        filter_layout.addLayout(date_layout)

        # 摄像头筛选
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("摄像头："))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("全部摄像头", None)
        camera_layout.addWidget(self.camera_combo)
        filter_layout.addLayout(camera_layout)

        # 人脸筛选
        face_layout = QVBoxLayout()
        face_layout.addWidget(QLabel("人脸："))
        self.face_combo = QComboBox()
        self.face_combo.addItem("全部人脸", None)
        face_layout.addWidget(self.face_combo)
        filter_layout.addLayout(face_layout)

        # 操作按钮
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_history)
        filter_layout.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("导出记录")
        self.export_btn.clicked.connect(self.export_history)
        filter_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("清除历史")
        self.clear_btn.clicked.connect(self.clear_history)
        filter_layout.addWidget(self.clear_btn)

        filter_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(filter_layout)

        # 导出目录选择
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("保存位置："))
        self.export_path_label = QLabel()
        self.export_path_label.setObjectName("exportPathLabel")
        self.export_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_layout.addWidget(self.export_path_label, 1)

        self.choose_dir_btn = QPushButton("选择保存位置")
        self.choose_dir_btn.clicked.connect(self.choose_export_directory)
        path_layout.addWidget(self.choose_dir_btn)
        main_layout.addLayout(path_layout)

        # --- 主体区：历史列表 + 详情面板 ---
        splitter = QSplitter(Qt.Horizontal)

        # 历史记录列表
        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self.on_history_item_selected)
        splitter.addWidget(self.history_list)

        # 右侧详情面板
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.StyledPanel)
        details_layout = QVBoxLayout()

        # 图片展示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        details_layout.addWidget(self.image_label)

        # 文本详情
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)

        # 查看截图按钮
        self.view_screenshot_btn = QPushButton("查看截图")
        self.view_screenshot_btn.clicked.connect(self.view_screenshot)
        details_layout.addWidget(self.view_screenshot_btn)

        details_frame.setLayout(details_layout)
        splitter.addWidget(details_frame)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.update_export_path_label()

    def load_camera_list(self):
        """从配置文件加载摄像头列表到下拉框"""
        try:
            import yaml
            with open('config/camera_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            for camera in config.get('cameras', []):
                self.camera_combo.addItem(
                    f"摄像头 {camera['id']}: {camera.get('name', '')}",
                    camera['id']
                )
        except Exception as e:
            logger.error(f"加载摄像头配置失败: {e}")

    def load_face_list(self):
        """从数据库加载已知人脸列表到下拉框"""
        try:
            known_faces = self.database.get_known_faces()
            for face in known_faces:
                self.face_combo.addItem(face['name'], face['name'])
        except Exception as e:
            logger.error(f"加载人脸列表失败: {e}")

    def refresh_history(self):
        """根据筛选条件刷新历史记录列表"""
        try:
            # 获取筛选条件
            filters = self._current_filter_conditions()

            # 从数据库获取符合条件的记录
            entries = self.database.get_face_logs(
                limit=1000,
                camera_id=filters['camera_id'],
                face_name=filters['face_name'],
                start_time=filters['start_time'],
                end_time=filters['end_time']
            )

            # 显示到列表中
            self.history_list.clear()
            self.current_entries = entries
            for entry in entries:
                try:
                    timestamp = float(entry.timestamp)
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    item_text = f"{time_str} - {entry.face_name} 出现在 {entry.camera_name}"
                    self.history_list.addItem(item_text)
                    self.history_list.item(self.history_list.count() - 1).setData(Qt.UserRole, entry)
                except Exception as e:
                    logger.error(f"历史记录解析错误: {e}")
                    continue
        except Exception as e:
            logger.error(f"刷新历史记录失败: {e}")

    def export_history(self):
        """将当前列表导出为 CSV"""
        if not self.current_entries:
            QMessageBox.information(self, "无记录", "没有可导出的历史记录。")
            return

        default_path = self.export_dir
        default_path.mkdir(parents=True, exist_ok=True)
        filename = default_path / f"历史记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出历史记录",
            str(filename),
            "CSV 文件 (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    '编号', '时间', '摄像头名称', '摄像头ID', '人脸名称', '年龄', '性别', '置信度', '截图路径'
                ])
                for entry in self.current_entries:
                    writer.writerow([
                        entry.id,
                        datetime.fromtimestamp(float(entry.timestamp)).strftime('%Y-%m-%d %H:%M:%S'),
                        entry.camera_name,
                        entry.camera_id,
                        entry.face_name,
                        entry.age if entry.age is not None else '',
                        entry.gender if entry.gender else '',
                        f"{float(entry.confidence):.2f}",
                        entry.screenshot_path or ''
                    ])
            QMessageBox.information(self, "导出成功", f"历史记录已导出到：\n{path}")
        except Exception as e:
            logger.error(f"导出历史记录失败: {e}")
            QMessageBox.critical(self, "错误", f"导出历史记录失败：{str(e)}")

    def clear_history(self):
        """提供清除筛选记录或全部记录的能力"""
        if not self.current_entries:
            QMessageBox.information(self, "无记录", "当前没有可清除的记录。")
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("确认清除")
        msg_box.setText("请选择要清除的记录范围：")
        filtered_btn = msg_box.addButton("清除筛选结果", QMessageBox.ActionRole)
        all_btn = msg_box.addButton("清除全部记录", QMessageBox.DestructiveRole)
        cancel_btn = msg_box.addButton(QMessageBox.Cancel)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.exec_()

        clicked = msg_box.clickedButton()
        if clicked == cancel_btn:
            return

        try:
            deleted = 0
            if clicked == filtered_btn:
                filters = self._current_filter_conditions()
                deleted = self.database.delete_face_logs(
                    camera_id=filters['camera_id'],
                    face_name=filters['face_name'],
                    start_time=filters['start_time'],
                    end_time=filters['end_time']
                )
            elif clicked == all_btn:
                deleted = self.database.delete_face_logs()

            if deleted > 0:
                QMessageBox.information(self, "已清除", f"成功清除 {deleted} 条历史记录。")
            else:
                QMessageBox.information(self, "无记录", "没有符合条件的记录被删除。")
            self.refresh_history()
        except Exception as e:
            logger.error(f"清除历史记录失败: {e}")
            QMessageBox.critical(self, "错误", "清除历史记录时出现问题。")

    def choose_export_directory(self):
        """允许用户自定义 CSV 保存位置"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择导出目录",
            str(self.export_dir)
        )
        if directory:
            self.export_dir = Path(directory)
            self.update_export_path_label()

    def on_history_item_selected(self, current, previous):
        """当用户选择一条历史记录时，显示详细信息"""
        try:
            if current is None:
                self.current_entry = None
                self.image_label.clear()
                self.details_label.clear()
                self.view_screenshot_btn.setEnabled(False)
                return

            entry = current.data(Qt.UserRole)
            if not isinstance(entry, FaceLogEntry):
                return

            self.current_entry = entry

            # 格式化时间
            try:
                timestamp = float(entry.timestamp)
                time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_str = "未知时间"

            # 格式化置信度
            try:
                confidence = float(entry.confidence)
                confidence_str = f"{confidence:.2f}"
            except Exception:
                confidence_str = "N/A"

            # 构建详情文本（HTML）
            details_text = (
                f"<b>时间：</b> {time_str}<br>"
                f"<b>摄像头：</b> {entry.camera_name} （ID: {entry.camera_id}）<br>"
                f"<b>人脸：</b> {entry.face_name}<br>"
                f"<b>年龄：</b> {entry.age if entry.age else '未知'}<br>"
                f"<b>性别：</b> {entry.gender if entry.gender else '未知'}<br>"
                f"<b>置信度：</b> {confidence_str}<br>"
            )
            self.details_label.setText(details_text)

            # 若截图存在则启用按钮
            self.view_screenshot_btn.setEnabled(
                entry.screenshot_path is not None and len(str(entry.screenshot_path)) > 0
            )

        except Exception as e:
            logger.error(f"显示记录详情失败: {e}")
            self.details_label.setText("加载记录详情时出错")
            self.view_screenshot_btn.setEnabled(False)

    def view_screenshot(self):
        """打开截图预览窗口"""
        if self.current_entry is None or not self.current_entry.screenshot_path:
            QMessageBox.information(self, "无截图", "该记录没有保存截图。")
            return

        try:
            screenshot_path = Path(self.current_entry.screenshot_path)
            if not screenshot_path.exists():
                QMessageBox.warning(self, "文件不存在", f"截图文件不存在：{screenshot_path}")
                return

            image = cv2.imread(str(screenshot_path))
            if image is None:
                raise ValueError("无法读取截图文件")

            pixmap = numpy_to_pixmap(image)

            # 弹出对话框显示截图
            dialog = QDialog(self)
            dialog.setWindowTitle("截图预览")
            layout = QVBoxLayout()

            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(
                800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(image_label)

            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)

            dialog.setLayout(layout)
            dialog.exec_()

        except Exception as e:
            logger.error(f"查看截图失败: {e}")
            QMessageBox.critical(self, "错误", f"加载截图失败：{str(e)}")
