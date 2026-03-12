import logging
from telegram import Bot
from telegram.error import TelegramError
from typing import Optional
from pathlib import Path
import asyncio
import time
# -*- coding: utf-8 -*-
class TelegramManager:
    """Telegram 消息管理模块：用于通过 Telegram 发送人脸识别警报信息"""

    def __init__(self, token: str, chat_id: str, rate_limit: int):
        self.token = token               # Bot 的 API Token
        self.chat_id = chat_id           # 接收消息的聊天 ID
        self.bot = None                  # Bot 实例
        self.last_sent = 0               # 上次发送的时间戳
        self.min_interval = rate_limit   # 发送最小间隔（秒）
        self.loop = asyncio.new_event_loop()  # 创建异步事件循环

    async def _initialize_bot(self):
        """异步初始化 Telegram Bot"""
        try:
            self.bot = Bot(token=self.token)
            await self.bot.get_me()  # 测试连接是否成功
            logging.info("Telegram 机器人初始化成功")
        except TelegramError as e:
            logging.error(f"Telegram 机器人初始化失败: {e}")
            self.bot = None

    def send_alert(self, message: str, image_path: Optional[Path] = None):
        """发送警报信息（同步包装函数）"""
        # 确保 Bot 已初始化
        if not self.bot:
            try:
                self.loop.run_until_complete(self._initialize_bot())
            except Exception as e:
                logging.error(f"Telegram 连接失败: {e}")
                return

        async def _send():
            """异步发送消息或图片"""
            now = time.time()
            # 限制发送频率
            if now - self.last_sent < self.min_interval:
                logging.warning(f"Telegram 发送频率限制 ({self.min_interval}s)")
                return
            try:
                # 若有图片则发送图片+文字
                if image_path and image_path.exists():
                    with open(image_path, 'rb') as photo:
                        await self.bot.send_photo(
                            chat_id=self.chat_id,
                            photo=photo,
                            caption=message
                        )
                else:
                    # 否则仅发送文字消息
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message
                    )
                self.last_sent = now
                logging.info("Telegram 警报发送成功")
            except TelegramError as e:
                logging.error(f"发送 Telegram 警报失败: {e}")

        # 执行异步任务
        try:
            self.loop.run_until_complete(_send())
        except Exception as e:
            logging.error(f"Telegram 警报发送出错: {str(e)}")
            # 如果发送失败，将警报内容保存到本地日志文件
            with open("failed_alerts.log", "a") as f:
                f.write(f"{time.ctime()}: {message}\n")

            # 若有图片，也保存到备用文件夹
            if image_path:
                backup_dir = Path("failed_alert_images")
                backup_dir.mkdir(exist_ok=True)
                new_path = backup_dir / f"alert_{int(time.time())}.jpg"
                try:
                    image_path.rename(new_path)
                except Exception as e:
                    logging.error(f"备份警报图片失败: {e}")

    def shutdown(self):
        """关闭并清理资源"""
        self._shutdown = True
        self.last_sent = 0
        if self.loop.is_running():
            self.loop.stop()

        # 取消所有未完成的异步任务
        pending = asyncio.all_tasks(loop=self.loop)
        for task in pending:
            task.cancel()

        self.loop.close()
