# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt
from loguru import logger
import cv2
from pathlib import Path

from core.utils import numpy_to_pixmap


class FaceManagerDialog(QDialog):
    """
    人脸管理窗口：用于管理已知人脸数据库（添加、更新、删除）
    """

    def __init__(self, face_detector, known_faces_dir):
        """
        初始化人脸管理界面

        Args:
            face_detector: 人脸检测/识别对象（FaceDetector 实例）
            known_faces_dir (str or Path): 存放已知人脸图片的目录
        """
        super().__init__()
        self.face_detector = face_detector
        self.known_faces_dir = known_faces_dir
        self.current_image = None  # 当前导入或显示的图片

        self.setWindowTitle("人脸管理")          # 窗口标题
        self.setGeometry(200, 200, 800, 600)  # 窗口大小

        self.init_ui()          # 初始化界面
        self.load_face_list()   # 加载人脸列表

    def init_ui(self):
        """设置界面组件，包括人脸列表、预览、输入框和按钮"""
        layout = QVBoxLayout()

        # 顶部区域：左边人脸列表 + 右边预览图
        top_layout = QHBoxLayout()

        # 人脸文件列表
        self.face_list = QListWidget()
        self.face_list.currentItemChanged.connect(self.on_face_selected)
        top_layout.addWidget(self.face_list, 3)

        # 人脸预览窗口
        self.face_preview = QLabel()
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setMinimumSize(300, 300)
        top_layout.addWidget(self.face_preview, 2)

        layout.addLayout(top_layout)

        # 中间：输入框（人脸名称）
        middle_layout = QHBoxLayout()
        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel("姓名："))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        middle_layout.addLayout(name_layout)
        layout.addLayout(middle_layout)

        # 底部按钮区
        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("添加人脸")
        self.add_btn.clicked.connect(self.add_face)
        button_layout.addWidget(self.add_btn)

        self.update_btn = QPushButton("更新人脸")
        self.update_btn.clicked.connect(self.update_face)
        button_layout.addWidget(self.update_btn)

        self.delete_btn = QPushButton("删除人脸")
        self.delete_btn.clicked.connect(self.delete_face)
        button_layout.addWidget(self.delete_btn)

        self.import_btn = QPushButton("导入图片")
        self.import_btn.clicked.connect(self.import_image)
        button_layout.addWidget(self.import_btn)

        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_face_list(self):
        """加载已知人脸目录中的图片文件名到列表中"""
        self.face_list.clear()
        known_faces_dir = Path(self.known_faces_dir)

        if not known_faces_dir.exists():
            logger.warning(f"已知人脸目录不存在: {known_faces_dir}")
            return

        for face_file in known_faces_dir.glob('*.*'):
            if face_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.face_list.addItem(face_file.stem)

    def on_face_selected(self, current, previous):
        """当选择列表中的人脸时，加载并显示对应图像"""
        if current is None:
            self.face_preview.clear()
            self.name_input.clear()
            return

        face_name = current.text()
        self.name_input.setText(face_name)

        # 拼接文件路径（支持多种扩展名）
        face_path = Path(self.known_faces_dir) / f"{face_name}{self.get_face_extension(face_name)}"
        if not face_path.exists():
            QMessageBox.warning(self, "错误", f"文件未找到: {face_path}")
            return

        try:
            image = cv2.imread(str(face_path))
            if image is None:
                raise ValueError("无法读取图像")

            self.current_image = image
            pixmap = numpy_to_pixmap(image)
            self.face_preview.setPixmap(pixmap.scaled(
                self.face_preview.width(), self.face_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            logger.error(f"加载人脸图像错误: {e}")

    def get_face_extension(self, face_name: str) -> str:
        """查找指定人脸文件的扩展名"""
        known_faces_dir = Path(self.known_faces_dir)
        for ext in ['.jpg', '.jpeg', '.png']:
            if (known_faces_dir / f"{face_name}{ext}").exists():
                return ext
        return ''

    def add_face(self):
        """添加新的人脸图片并更新人脸识别器"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "错误", "请先输入人脸名称")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "错误", "请先导入或选择一张图片")
            return

        # 检查是否已存在同名人脸
        existing_files = list(Path(self.known_faces_dir).glob(f"{name}.*"))
        if existing_files:
            QMessageBox.warning(self, "错误", f"'{name}' 已存在")
            return

        # 调用检测器添加
        success = self.face_detector.add_known_face(
            self.current_image, name, self.known_faces_dir)

        if success:
            QMessageBox.information(self, "成功", f"人脸 '{name}' 添加成功")
            self.load_face_list()
        else:
            QMessageBox.warning(self, "错误", "添加失败")

    def update_face(self):
        """更新已有人脸（重命名或替换图片）"""
        current_item = self.face_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "错误", "请先选择一个人脸")
            return

        old_name = current_item.text()
        new_name = self.name_input.text().strip()

        if not new_name:
            QMessageBox.warning(self, "错误", "请输入人脸名称")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "错误", "请导入或选择一张图片")
            return

        # 如果名称变化则重命名文件
        if old_name != new_name:
            old_path = Path(self.known_faces_dir) / f"{old_name}{self.get_face_extension(old_name)}"
            new_path = Path(self.known_faces_dir) / f"{new_name}{old_path.suffix}"

            if new_path.exists():
                QMessageBox.warning(self, "错误", f"'{new_name}' 已存在")
                return

            try:
                old_path.rename(new_path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"重命名失败: {str(e)}")
                return

        # 更新图片
        try:
            current_path = Path(self.known_faces_dir) / f"{new_name}{self.get_face_extension(new_name)}"
            cv2.imwrite(str(current_path), self.current_image)

            # 重新加载识别器中的人脸库
            self.face_detector.load_known_faces(self.known_faces_dir)

            QMessageBox.information(self, "成功", "人脸更新成功")
            self.load_face_list()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新失败: {str(e)}")

    def delete_face(self):
        """删除选中的人脸文件"""
        current_item = self.face_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "错误", "请先选择要删除的人脸")
            return

        name = current_item.text()

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除人脸 '{name}' 吗？",
            QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.No:
            return

        face_path = Path(self.known_faces_dir) / f"{name}{self.get_face_extension(name)}"
        try:
            face_path.unlink()
            self.face_detector.load_known_faces(self.known_faces_dir)
            self.load_face_list()
            QMessageBox.information(self, "成功", "删除成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")

    def import_image(self):
        """导入外部图片，显示预览并自动填充名称"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)")

        if not file_path:
            return

        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("无法读取图像")

            self.current_image = image
            pixmap = numpy_to_pixmap(image)
            self.face_preview.setPixmap(pixmap.scaled(
                self.face_preview.width(), self.face_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # 默认使用文件名作为建议名称
            suggested_name = Path(file_path).stem
            self.name_input.setText(suggested_name)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            logger.error(f"导入图片出错: {e}")
