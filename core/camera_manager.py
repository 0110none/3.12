import cv2
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger
import time
import threading
import queue
import yaml
from typing import List, Dict

# -*- coding: utf-8 -*-
# 摄像头配置数据结构
@dataclass
class CameraConfig:
    id: int
    name: str
    source: str        # 视频源路径或摄像头编号
    enabled: bool      # 是否启用
    width: int         # 分辨率宽度
    height: int        # 分辨率高度
    fps: int           # 帧率
    rotate: int        # 旋转角度（0, 90, 180, 270）

class CameraManager:
    """摄像头管理类：负责多摄像头的启动、捕获、停止与状态管理"""

    def __init__(self, config_path: str):
        self.cameras: Dict[int, CameraConfig] = {}         # 存储摄像头配置
        self.capture_threads: Dict[int, threading.Thread] = {}  # 捕获线程
        self.stop_event = threading.Event()                # 停止信号
        self.frame_queues: Dict[int, queue.Queue] = {}     # 每个摄像头的帧队列
        self.load_config(config_path)

    def _cleanup_camera_thread(self, cam_id: int):
        """清理指定摄像头的线程与缓存"""
        if cam_id in self.capture_threads:
            self.stop_event.set()  # 发出停止信号
            thread = self.capture_threads.pop(cam_id)
            thread.join(timeout=2.0)

            if thread.is_alive():
                logger.warning(f"摄像头 {cam_id} 的线程未正常退出")

            # 清空帧队列
            if cam_id in self.frame_queues:
                try:
                    while True:
                        self.frame_queues[cam_id].get_nowait()
                except queue.Empty:
                    pass
                del self.frame_queues[cam_id]

            logger.debug(f"已清理摄像头 {cam_id} 的资源")

    def load_config(self, config_path: str) -> None:
        """从 YAML 文件加载摄像头配置"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.cameras.clear()
            for cam_config in config.get('cameras', []):
                cam_id = cam_config['id']
                self.cameras[cam_id] = CameraConfig(
                    id=cam_id,
                    name=cam_config.get('name', f'Camera {cam_id}'),
                    source=cam_config['source'],
                    enabled=cam_config.get('enabled', True),
                    width=cam_config['resolution']['width'],
                    height=cam_config['resolution']['height'],
                    fps=cam_config.get('fps', 30),
                    rotate=cam_config.get('rotate', 0)
                )
            logger.info(f"成功加载 {len(self.cameras)} 个摄像头配置")
        except Exception as e:
            logger.error(f"加载摄像头配置失败: {e}")
            raise

    def start_all_cameras(self) -> None:
        """启动所有启用状态的摄像头"""
        self.stop_event.clear()
        for cam_id, cam_config in self.cameras.items():
            if cam_config.enabled:
                self.start_camera(cam_id)

    def stop_all_cameras(self) -> None:
        """停止所有摄像头线程"""
        self.stop_event.set()
        for thread in self.capture_threads.values():
            thread.join(timeout=2)
        self.capture_threads.clear()
        self.frame_queues.clear()
        logger.info("所有摄像头线程已停止")

    def start_camera(self, cam_id: int) -> bool:
        """启动单个摄像头"""
        if cam_id not in self.cameras:
            logger.error(f"未找到摄像头 {cam_id} 的配置")
            return False

        if not self.cameras[cam_id].enabled:
            logger.warning(f"摄像头 {cam_id} 在配置中被禁用")
            return False

        # 清理旧线程
        if cam_id in self.capture_threads:
            self._cleanup_camera_thread(cam_id)

        # 创建帧队列与捕获线程
        self.frame_queues[cam_id] = queue.Queue(maxsize=1)
        self.stop_event.clear()

        self.capture_threads[cam_id] = threading.Thread(
            target=self._capture_frames,
            args=(cam_id,),
            daemon=True,
            name=f"CameraThread-{cam_id}"
        )
        self.capture_threads[cam_id].start()
        logger.info(f"摄像头 {cam_id} 已启动")
        return True

    def stop_camera(self, cam_id: int) -> bool:
        """停止单个摄像头"""
        if cam_id in self.capture_threads:
            self._cleanup_camera_thread(cam_id)
            logger.info(f"摄像头 {cam_id} 已停止")
            return True
        return False

    def _capture_frames(self, cam_id: int) -> None:
        """摄像头捕获线程函数"""
        cam_config = self.cameras[cam_id]
        cap = None

        try:
            # 支持数字ID或视频路径
            source = int(cam_config.source) if str(cam_config.source).isdigit() else cam_config.source
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.error(f"无法打开摄像头 {cam_id}，源：{cam_config.source}")
                return

            # 设置分辨率和帧率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config.height)
            cap.set(cv2.CAP_PROP_FPS, cam_config.fps)

            logger.info(f"摄像头 {cam_id} 打开成功")

            # 不断读取帧
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"摄像头 {cam_id} 读取失败")
                    time.sleep(1)
                    continue

                # 按配置旋转画面
                if cam_config.rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif cam_config.rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif cam_config.rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # 更新帧队列（仅保留最新一帧）
                if self.frame_queues[cam_id].full():
                    try:
                        self.frame_queues[cam_id].get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queues[cam_id].put(frame)

        except Exception as e:
            logger.error(f"摄像头 {cam_id} 捕获线程出错: {e}")
        finally:
            if cap is not None:
                cap.release()
            logger.info(f"摄像头 {cam_id} 捕获线程退出")

    def get_frame(self, cam_id: int) -> Optional[np.ndarray]:
        """获取指定摄像头的最新帧"""
        if cam_id not in self.frame_queues:
            return None
        try:
            return self.frame_queues[cam_id].get_nowait()
        except queue.Empty:
            return None

    def get_all_frames(self) -> Dict[int, np.ndarray]:
        """获取所有摄像头的最新帧"""
        frames = {}
        for cam_id in self.frame_queues:
            frame = self.get_frame(cam_id)
            if frame is not None:
                frames[cam_id] = frame
        return frames

    def get_camera_status(self, cam_id: int) -> Dict:
        """获取指定摄像头状态"""
        if cam_id not in self.cameras:
            return {}
        return {
            'id': cam_id,
            'name': self.cameras[cam_id].name,
            'running': cam_id in self.capture_threads,
            'frame_queue_size': self.frame_queues.get(cam_id, queue.Queue()).qsize(),
            'enabled': self.cameras[cam_id].enabled
        }

    def get_all_camera_status(self) -> List[Dict]:
        """获取所有摄像头的状态"""
        return [self.get_camera_status(cam_id) for cam_id in self.cameras]
