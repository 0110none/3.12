# -*- coding: utf-8 -*-

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict   # ★ 增加 Dict
from loguru import logger
import cv2
import numpy as np
from pathlib import Path
from pygame import mixer

from .telegram_manager import TelegramManager
from .face_detection import Face


# 警报事件的数据结构，自动生成 __init__ 等方法
@dataclass
class AlertEvent:
    camera_id: int
    camera_name: str
    face_name: str
    confidence: float
    timestamp: float
    age: Optional[int] = None
    gender: Optional[str] = None  # 性别：'Male' 或 'Female'
    screenshot_path: Optional[str] = None
    is_cooldown: bool = False


class AlertSystem:
    """
    警报系统
    --------
    负责：
    - 播放警报音（区分已知人/陌生人）
    - 保存截图
    - 发送 Telegram 通知
    - 维护本地警报历史记录
    """

    def __init__(self, config: dict):
        self.config = config

        # 已知人警报音
        self.alert_sound = config['app']['alert_sound']
        # 陌生人警报音（如果没有配置，就退回用同一份）
        self.unknown_alert_sound = config['app'].get(
            'unknown_alert_sound',
            self.alert_sound
        )

        # 截图保存目录
        self.screenshot_dir = Path(config['app']['screenshot_dir'])
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # 内存中的警报历史
        self.alert_history: List[AlertEvent] = []

        # 功能开关
        self.alert_enabled = True       # 是否启用声音警报
        self.screenshot_enabled = True  # 是否保存截图

        # ★★★ 冷却相关：同一摄像头+同一姓名 10 秒内只触发一次“真正报警”
        self.last_alert_time: Dict[str, float] = {}  # key: f"{camera_id}_{face_name}"
        self.alert_cooldown: float = 10.0            # 冷却时间（秒）

        # Telegram 相关
        self.telegram = None
        if config.get('telegram', {}).get('enabled', False):
            self.telegram = TelegramManager(
                config['telegram']['bot_token'],
                config['telegram']['chat_id'],
                config['telegram']['rate_limit']
            )

        # 初始化声音播放模块
        mixer.init()

    def trigger_alert(
        self,
        camera_id: int,
        camera_name: str,
        face_name: str,
        face: Face,
        confidence: float,
        frame: np.ndarray
    ) -> AlertEvent:
        """
        触发一次警报（识别到人脸时调用）

        根据 face_name 判断是否为陌生人：
        - 已知人：使用 alert_sound
        - 陌生人（Unknown / 未知 / 陌生人）：使用 unknown_alert_sound
        """
        now = time.time()

        # ===== 冷却逻辑：同一摄像头 + 同一名字，10 秒内只做一次重操作 =====
        key = f"{camera_id}_{face_name}"
        last = self.last_alert_time.get(key, 0)
        in_cooldown = (now - last) < self.alert_cooldown

        is_unknown = face_name in ("Unknown", "未知", "陌生人")

        # ★ 冷却期：只记录事件，is_cooldown=True，不截图、不声音、不Telegram
        if in_cooldown:
            event = AlertEvent(
                camera_id=camera_id,
                camera_name=camera_name,
                face_name=face_name,
                age=face.age,
                gender=face.gender,
                confidence=confidence,
                timestamp=now,
                screenshot_path=None,
                is_cooldown=True  # ★ 标记为冷却事件
            )
            self.alert_history.append(event)
            logger.info(
                f"警报冷却中（{self.alert_cooldown:.0f}s）："
                f"{face_name} 在 {camera_name} 被检测到，置信度 {confidence:.2f}"
            )
            return event

        # ★ 只有真正报警时才更新 last_alert_time
        self.last_alert_time[key] = now

        # ===== 非冷却期：正常执行报警 =====
        timestamp = now
        screenshot_path = None

        if self.screenshot_enabled:
            screenshot_path = self._capture_screenshot(
                frame, camera_id, face_name, timestamp
            )

        event = AlertEvent(
            camera_id=camera_id,
            camera_name=camera_name,
            face_name=face_name,
            age=face.age,
            gender=face.gender,
            confidence=confidence,
            timestamp=timestamp,
            screenshot_path=str(screenshot_path) if screenshot_path is not None else None,
            is_cooldown=False  # ★ 标记为真正报警
        )

        logger.debug(f"事件创建成功，截图路径: {event.screenshot_path}")
        self.alert_history.append(event)

        # 播放警报音
        if self.alert_enabled:
            self._play_alert_sound(is_unknown=is_unknown)

        # 发送 Telegram 通知
        if self.telegram:
            message_lines = [
                "检测到人脸!",
                f"姓名: {face_name}"
            ]

            if face.age:
                message_lines.append(f"年龄: 约 {face.age} 岁")
            if face.gender:
                message_lines.append(f"性别: {face.gender}")

            message_lines.extend([
                f"摄像头: {camera_name}",
                f"置信度: {confidence:.2%}",
                f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            ])

            message = "\n".join(message_lines)

            self.telegram.send_alert(
                message=message,
                image_path=screenshot_path
            )

        logger.info(
            f"警报触发：{face_name} 在 {camera_name} 被检测到，置信度 {confidence:.2f}"
            + ("（陌生人）" if is_unknown else "（已知人）")
        )
        return event

    def _play_alert_sound(self, is_unknown: bool = False) -> None:
        """
        播放警报音

        参数:
            is_unknown (bool): 是否为陌生人
                - False：播放已知人警报音 alert_sound
                - True ：播放陌生人警报音 unknown_alert_sound
        """
        try:
            # 根据类型选择声音文件
            sound_path = self.unknown_alert_sound if is_unknown else self.alert_sound

            if os.path.exists(sound_path):
                mixer.music.load(sound_path)
                mixer.music.play()
            else:
                logger.warning(f"警报音文件不存在: {sound_path}")
        except Exception as e:
            logger.error(f"播放警报音失败: {e}")

    def _capture_screenshot(
            self,
            frame: np.ndarray,
            camera_id: int,
            face_name: str,
            timestamp: float
    ) -> Optional[Path]:
        """
        保存当前帧为截图文件，并返回路径

        规则：
        - 已知人：YYYYMMDD_HHMMSS_cam0_姓名.jpg
        - 陌生人：YYYYMMDD_HHMMSS_cam0_unknown.jpg
        """
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))

            # 1. 统一“陌生人”的文件名为 unknown
            if face_name is None:
                label = "unknown"
            else:
                name_str = str(face_name)
                if name_str in ["未知", "unknown", "Unknown", "陌生人"]:
                    label = "unknown"
                else:
                    label = name_str

            # 2. “安全化”文件名
            safe_name = "".join(
                c if c.isalnum() else "_"  # 不是字母数字就换成 _
                for c in label
            )

            if not safe_name:
                safe_name = "unknown"

            filename = f"{timestamp_str}_cam{camera_id}_{safe_name}.jpg"
            filepath = self.screenshot_dir / filename

            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件
            success = cv2.imwrite(str(filepath), frame)
            if not success:
                logger.error(f"截图保存失败: {filepath}")
                return None

            logger.info(f"截图已保存: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"截图出错: {e}")
            return None

    def get_recent_alerts(self, limit: int = 10) -> List[AlertEvent]:
        """获取最近的警报记录（按时间倒序，返回前 limit 条）"""
        return sorted(
            self.alert_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]

    def clear_alerts(self) -> None:
        """清空内存中的警报历史记录"""
        self.alert_history.clear()

    def enable_alerts(self, enabled: bool) -> None:
        """启用或禁用声音警报"""
        self.alert_enabled = enabled
        logger.info(f"声音警报 {'已启用' if enabled else '已禁用'}")

    def enable_screenshots(self, enabled: bool) -> None:
        """启用或禁用截图功能"""
        self.screenshot_enabled = enabled
        logger.info(f"截图功能 {'已启用' if enabled else '已禁用'}")

    def shutdown(self):
        """关闭并释放资源（目前主要是 Telegram 管理器）"""
        if hasattr(self, 'telegram') and self.telegram:
            self.telegram.shutdown()
