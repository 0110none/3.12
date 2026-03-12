from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from loguru import logger
from PIL import ImageFont, ImageDraw, Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import time


def draw_face_info(
    image: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    name: Optional[str] = None,
    confidence: Optional[float] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    camera_name: Optional[str] = None,
    timestamp: Optional[float] = None,
    ) -> np.ndarray:
    """
    在图像上绘制人脸边框与识别信息（支持中文显示）
    """
    try:
        img = image.copy()
        x1, y1, x2, y2 = map(int, face_bbox)

        # 判断是否为陌生人（没有名字或叫“未知”）
        is_unknown = (name is None) or (str(name) in ["未知", "unknown", "Unknown"])
        color = (0, 0, 255) if is_unknown else (0, 255, 0)  # 红色：陌生人，绿色：已知人

        # 绘制人脸框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 组合文字信息（中文）
        info_text = []
        if timestamp:
            info_text.append(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
        if camera_name:
            info_text.append(f"摄像头：{camera_name}")
        if gender:
            g = str(gender).lower()
            if "male" in g:
                gender_cn = "男"
            elif "female" in g:
                gender_cn = "女"
            else:
                gender_cn = "未知"
            info_text.append(f"性别：{gender_cn}")
        if age:
            info_text.append(f"年龄：{age}")
        if confidence is not None:
            info_text.append(f"置信度：{confidence:.2f}")
        if name:
            info_text.append(f"姓名：{name}")

        # 使用 PIL 绘制中文文字（兼容多平台字体路径）
        font = _load_chinese_font()

        # 将 OpenCV 图像转为 PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        text_y = y1 - 25 if y1 - 25 > 10 else y2 + 20
        for i, text in enumerate(info_text):
            text_position = (x1 + 3, text_y - i * 22)
            draw.text(text_position, text, font=font, fill=(0, 255, 0))  # 绿色文字

        # 转回 OpenCV 图像
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    except Exception as e:
        logger.error(f"绘制人脸信息出错: {e}")
        return image


def _load_chinese_font() -> ImageFont.ImageFont:
    """
    在不同系统上寻找可用的中文字体，保证中文文本正常显示。
    优先选择常见的中文字体文件，若均不存在则回退到默认字体。
    """
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),                 # Windows 微软雅黑
        Path("/System/Library/Fonts/STHeiti Light.ttc"),   # macOS 黑体
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),  # macOS 黑体
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), 18)
            except Exception as exc:  # pragma: no cover - 防御性处理
                logger.warning(f"加载字体 {font_path} 失败，尝试下一个候选：{exc}")

    logger.warning("未找到可用的中文字体，回退到默认字体显示。")
    return ImageFont.load_default()


def numpy_to_pixmap(image: np.ndarray) -> "QPixmap":
    """将 numpy 图像数据转换为 QPixmap（用于 PyQt 显示）"""
    try:
        if image is None:
            return QPixmap()

        if len(image.shape) == 2:  # 灰度图
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:  # 彩色图（BGR 格式）
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)

        # 转换为 QPixmap，并保持比例缩放
        return QPixmap.fromImage(qimg).scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    except Exception as e:
        logger.error(f"numpy 转 QPixmap 出错: {e}")
        return QPixmap()


def resize_image(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """按比例缩放图像，防止图像过大导致显示异常"""
    try:
        if image is None:
            return None

        h, w = image.shape[:2]

        # 若尺寸在限制内则不缩放
        if w <= max_width and h <= max_height:
            return image

        ratio = min(max_width / w, max_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    except Exception as e:
        logger.error(f"调整图像尺寸出错: {e}")
        return image
