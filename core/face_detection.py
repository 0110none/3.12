import cv2
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import time
# -*- coding: utf-8 -*-
# 人脸信息数据结构（单张检测结果）
@dataclass
class Face:
    bbox: np.ndarray          # 人脸位置（边框坐标）
    kps: np.ndarray           # 关键点坐标（眼、鼻、嘴）
    det_score: float          # 检测置信度
    embedding: np.ndarray     # 人脸特征向量
    age: Optional[int] = None
    gender: Optional[str] = None  # 'Male' / 'Female'
    face_img: Optional[np.ndarray] = None  # 裁剪后的人脸图像

# 已知人脸数据结构
@dataclass
class KnownFace:
    name: str
    embedding: np.ndarray
    image_path: str


class FaceDetector:
    """人脸检测与识别模块：负责加载模型、检测人脸、提取特征与匹配识别"""

    def __init__(self, config: dict):
        self.config = config
        self.recognition_threshold = config['recognition']['recognition_threshold']  # 识别阈值
        self.detection_threshold = config['recognition']['detection_threshold']      # 检测阈值
        self.max_batch_size = config['recognition']['max_batch_size']
        self.device = config['recognition'].get('device', 'cuda').lower()            # CPU / GPU
        self.analysis_enabled = config['recognition'].get('analysis_enabled', True) # 是否启用年龄性别分析
        self.model = self._load_model()   # 加载 InsightFace 模型
        self.known_faces: List[KnownFace] = []  # 已知人脸列表

    def _cuda_available(self) -> bool:
        """检查当前环境是否可用 CUDA。"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def _load_model(self) -> FaceAnalysis:
        """加载 InsightFace 模型"""
        preferred_cuda = self.device in ('cuda', 'gpu', 'auto')
        run_on_cuda = preferred_cuda and self._cuda_available()

        if preferred_cuda and not run_on_cuda:
            logger.warning("未检测到可用 CUDA，自动切换为 CPU 模式")

        contexts = [0, -1] if run_on_cuda else [-1]
        last_error = None

        for ctx_id in contexts:
            try:
                model = FaceAnalysis(
                    name='buffalo_l',
                    root='./',
                    allowed_modules=['detection', 'recognition', 'genderage']
                )
                model.prepare(
                    ctx_id=ctx_id,
                    det_thresh=self.detection_threshold,
                    det_size=(640, 640)
                )
                self.device = 'cuda' if ctx_id == 0 else 'cpu'
                logger.success(f"人脸检测模型加载成功（设备: {self.device.upper()}）")
                return model
            except Exception as e:
                last_error = e
                logger.warning(
                    f"使用 {'CUDA' if ctx_id == 0 else 'CPU'} 加载模型失败，"
                    f"将尝试其他可用设备: {e}"
                )

        logger.error(f"加载人脸检测模型失败: {last_error}")
        raise last_error

    def load_known_faces(self, known_faces_dir: str) -> None:
        """从目录加载已知人脸库"""
        try:
            self.known_faces.clear()
            known_faces_dir = Path(known_faces_dir)

            if not known_faces_dir.exists():
                logger.warning(f"目录不存在: {known_faces_dir}")
                return

            for face_file in known_faces_dir.glob('*.*'):
                if face_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                try:
                    img = cv2.imread(str(face_file))
                    if img is None:
                        logger.warning(f"无法读取图片: {face_file}")
                        continue

                    faces = self.model.get(img)
                    if len(faces) == 0:
                        logger.warning(f"未检测到人脸: {face_file}")
                        continue

                    # 使用第一张检测到的人脸
                    face = faces[0]
                    name = face_file.stem
                    self.known_faces.append(KnownFace(
                        name=name,
                        embedding=face.embedding,
                        image_path=str(face_file)
                    ))
                    logger.info(f"已加载人脸: {name}")

                except Exception as e:
                    logger.error(f"处理文件 {face_file} 出错: {e}")

            logger.info(f"共加载 {len(self.known_faces)} 张已知人脸")

        except Exception as e:
            logger.error(f"加载人脸库出错: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """检测输入图像中的所有人脸"""
        try:
            faces = self.model.get(image)
            results = []

            for face in faces:
                face_img = self._extract_face_image(image, face.bbox)
                results.append(Face(
                    bbox=face.bbox,
                    kps=face.kps,
                    det_score=face.det_score,
                    embedding=face.embedding,
                    age=self._get_age(face),
                    gender=self._get_gender(face),
                    face_img=face_img
                ))
            return results
        except Exception as e:
            logger.error(f"检测人脸出错: {e}")
            return []

    def recognize_faces(self, faces: List[Face]) -> List[Tuple[Face, Optional[KnownFace], float]]:
        """将检测到的人脸与已知人脸库进行匹配"""
        results = []

        if not self.known_faces:
            return [(face, None, 0.0) for face in faces]

        try:
            # 提取已知人脸的所有 embedding 向量
            known_embeddings = np.array([kf.embedding for kf in self.known_faces])

            for face in faces:
                if face.embedding is None or len(face.embedding) == 0:
                    results.append((face, None, 0.0))
                    continue

                # 计算余弦相似度（Cosine Similarity）
                similarities = np.dot(known_embeddings, face.embedding) / (
                    np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(face.embedding)
                )

                max_idx = np.argmax(similarities)
                max_similarity = similarities[max_idx]

                # 判断是否超过识别阈值
                if max_similarity > self.recognition_threshold:
                    results.append((face, self.known_faces[max_idx], max_similarity))
                else:
                    results.append((face, None, max_similarity))

        except Exception as e:
            logger.error(f"识别人脸出错: {e}")
            return [(face, None, 0.0) for face in faces]

        return results

    def _extract_face_image(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """根据边框提取人脸区域图像"""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x1 >= x2 or y1 >= y2:
            return np.array([])

        return image[y1:y2, x1:x2].copy()

    def add_known_face(self, image: np.ndarray, name: str, save_dir: str) -> bool:
        """添加新的已知人脸到人脸库"""
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            faces = self.detect_faces(image)
            if not faces:
                logger.warning("未检测到人脸，添加失败")
                return False

            # 使用第一张检测到的人脸
            face = faces[0]

            # 保存人脸图片
            timestamp = int(time.time())
            face_path = save_dir / f"{name}_{timestamp}.jpg"
            cv2.imwrite(str(face_path), image)

            # 添加到已知人脸列表
            self.known_faces.append(KnownFace(
                name=name,
                embedding=face.embedding,
                image_path=str(face_path)
            ))

            logger.info(f"新的人脸已添加: {name}")
            return True

        except Exception as e:
            logger.error(f"添加人脸出错: {e}")
            return False

    def _get_age(self, face) -> Optional[int]:
        """提取年龄预测结果"""
        if not self.analysis_enabled:
            return None
        return int(face.age) if hasattr(face, 'age') else None

    def _get_gender(self, face) -> Optional[str]:
        """提取性别预测结果"""
        if not self.analysis_enabled:
            return None
        if not hasattr(face, 'sex') or face.sex is None:
            return None
        return 'Female' if np.argmax(face.sex) == 1 else 'Male'
