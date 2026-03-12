import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from loguru import logger
import time
# -*- coding: utf-8 -*-
# 人脸识别事件数据结构
@dataclass
class FaceLogEntry:
    id: int
    timestamp: float
    camera_id: int
    camera_name: str
    face_name: str
    age: Optional[int]
    gender: Optional[str]
    confidence: float
    screenshot_path: Optional[str]

    def __post_init__(self):
        """修正 timestamp 的类型（从字节或字符串转为浮点数）"""
        if isinstance(self.timestamp, bytes):
            self.timestamp = float(self.timestamp.decode('utf-8'))
        elif isinstance(self.timestamp, str):
            self.timestamp = float(self.timestamp)


class FaceDatabase:
    """人脸数据库类：负责事件记录与人脸数据的存储与查询"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库与表结构"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 创建“人脸日志”表（记录每次识别事件）
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        camera_id INTEGER NOT NULL,
                        camera_name TEXT NOT NULL,
                        face_name TEXT NOT NULL,
                        age INTEGER,
                        gender TEXT,
                        confidence REAL NOT NULL,
                        screenshot_path TEXT
                    )
                ''')

                # 创建“已知人脸”表（保存人脸特征向量）
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS known_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        embedding BLOB NOT NULL,
                        image_path TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                ''')

                conn.commit()
                logger.success("数据库初始化成功")

        except Exception as e:
            logger.error(f"数据库初始化出错: {e}")
            raise

    def log_face_event(self, event) -> int:
        """将识别事件写入日志表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO face_logs (
                        timestamp, camera_id, camera_name, face_name,
                        age, gender, confidence, screenshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    float(event.timestamp),
                    int(event.camera_id),
                    str(event.camera_name),
                    str(event.face_name),
                    int(event.age) if event.age else None,
                    str(event.gender) if event.gender else None,
                    float(event.confidence),
                    str(event.screenshot_path) if event.screenshot_path else None
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"记录识别事件失败: {e}")
            raise

    def get_face_logs(self, limit: int = 100,
                      camera_id: Optional[int] = None,
                      face_name: Optional[str] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> List[FaceLogEntry]:
        """按条件查询人脸识别日志"""
        try:
            query = '''
                SELECT id, timestamp, camera_id, camera_name, face_name, age, gender, confidence, screenshot_path
                FROM face_logs
            '''
            params = []
            conditions = []

            # 根据条件拼接 SQL 查询
            if camera_id is not None:
                conditions.append("camera_id = ?")
                params.append(camera_id)
            if face_name is not None:
                conditions.append("face_name = ?")
                params.append(face_name)
            if start_time is not None:
                conditions.append("timestamp >= ?")
                params.append(float(start_time))
            if end_time is not None:
                conditions.append("timestamp <= ?")
                params.append(float(end_time))

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)

                entries = []
                for row in cursor.fetchall():
                    try:
                        entries.append(FaceLogEntry(
                            id=row['id'],
                            timestamp=float(row['timestamp']),
                            camera_id=row['camera_id'],
                            camera_name=row['camera_name'],
                            face_name=row['face_name'],
                            age=row['age'],
                            gender=row['gender'],
                            confidence=float(row['confidence']),
                            screenshot_path=row['screenshot_path']
                        ))
                    except Exception as e:
                        logger.error(f"行数据转换失败: {dict(row)}，错误: {e}")
                        continue

                return entries
        except Exception as e:
            logger.error(f"查询人脸日志失败: {e}")
            return []

    def clear_face_logs(self) -> bool:
        """清空人脸识别日志表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM face_logs")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"清空历史记录失败: {e}")
            return False

    def delete_face_logs(self,
                         camera_id: Optional[int] = None,
                         face_name: Optional[str] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> int:
        """按条件删除人脸识别日志，返回被删除的记录数"""
        try:
            query = "DELETE FROM face_logs"
            params = []
            conditions = []

            if camera_id is not None:
                conditions.append("camera_id = ?")
                params.append(camera_id)
            if face_name is not None:
                conditions.append("face_name = ?")
                params.append(face_name)
            if start_time is not None:
                conditions.append("timestamp >= ?")
                params.append(float(start_time))
            if end_time is not None:
                conditions.append("timestamp <= ?")
                params.append(float(end_time))

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"删除历史记录失败: {e}")
            return 0

    def add_known_face(self, name: str, embedding: bytes, image_path: str) -> bool:
        """添加新的已知人脸（特征 + 图像路径）"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO known_faces (name, embedding, image_path, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (name, embedding, image_path, time.time()))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"人脸 '{name}' 已存在")
            return False
        except Exception as e:
            logger.error(f"添加人脸失败: {e}")
            return False

    def get_known_faces(self) -> List[dict]:
        """获取数据库中所有已知人脸"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT name, embedding, image_path FROM known_faces
                ''')
                return [{
                    'name': row[0],
                    'embedding': row[1],
                    'image_path': row[2]
                } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取已知人脸失败: {e}")
            return []

    def delete_known_face(self, name: str) -> bool:
        """删除指定名称的人脸记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM known_faces WHERE name = ?
                ''', (name,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"删除人脸失败: {e}")
            return False
