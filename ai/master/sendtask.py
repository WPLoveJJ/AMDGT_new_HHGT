import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import base64

# 配置更详细的日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WeComTaskProcessor")

# 企业微信配置
CORPID = "wwb0728887ce23a4ce"
CORPSECRET = "vZ7mge0BomfhLaza43spNe9Wb8EmBdjmWQzxrhD10j4"

# 智能表API配置
API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Accept": "application/json"
}


class WeComTaskHandler:
    """企业微信任务处理类（完整版）"""

    def __init__(self, corpid: str, corpsecret: str):
        self.corpid = corpid
        self.corpsecret = corpsecret
        self.access_token = None
        self.token_expires_at = 0
        self.timeout = aiohttp.ClientTimeout(total=60)
        self._session = None  # 延迟创建会话

    async def _get_session(self):
        """创建或返回现有会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            logger.debug("✅ 创建新的aiohttp会话")
        return self._session

    async def _get_access_token(self) -> Optional[str]:
        """获取企业微信访问令牌（内部方法）"""
        logger.debug("尝试获取AccessToken...")
        if self.access_token and self.token_expires_at > datetime.now().timestamp():
            logger.debug("✅ 使用缓存的AccessToken")
            return self.access_token

        session = await self._get_session()
        token_url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
            f"?corpid={self.corpid}"
            f"&corpsecret={self.corpsecret}"
        )

        logger.debug(f"请求AccessToken: {token_url}")
        try:
            async with session.get(token_url, timeout=self.timeout) as resp:
                result = await resp.json()
                logger.debug(f"AccessToken响应: {json.dumps(result)}")
                if result.get("errcode") == 0:
                    self.access_token = result["access_token"]
                    self.token_expires_at = datetime.now().timestamp() + 7100
                    logger.info("✅ 获取企微AccessToken成功")
                    return self.access_token
                else:
                    error_msg = f"获取AccessToken失败：{result.get('errmsg')}（错误码：{result.get('errcode')}）"
                    logger.error(error_msg)
                    return None
        except Exception as e:
            logger.error(f"获取AccessToken异常：{str(e)}")
            return None

    async def upload_temp_media(self, img_data: bytes, file_name: str) -> Optional[str]:
        """使用临时素材接口上传图片（无需docid）"""
        logger.debug(f"临时素材上传: {file_name} (大小: {len(img_data) // 1024}KB)")
        access_token = await self._get_access_token()
        if not access_token:
            logger.error("❌ 无AccessToken，素材上传失败")
            return None

        url = f"https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token={access_token}&type=image"

        try:
            form_data = aiohttp.FormData()
            form_data.add_field('media', img_data, filename=file_name)

            async with self._session.post(url, data=form_data) as resp:
                # 先获取文本响应避免JSON解析错误
                resp_text = await resp.text()
                try:
                    result = json.loads(resp_text)
                except json.JSONDecodeError:
                    logger.error(f"❌ 素材接口返回非JSON: {resp_text[:200]}...")
                    return None

                if result.get("errcode") == 0:
                    media_id = result.get("media_id")
                    logger.info(f"✅ 临时素材上传成功: {file_name} → media_id: {media_id[:20]}...")
                    return media_id
                else:
                    error_msg = result.get("errmsg", "未知错误")
                    logger.error(f"❌ 素材上传失败（{result['errcode']}）: {error_msg}")
        except Exception as e:
            logger.error(f"素材上传异常: {str(e)}")
        return None

    async def create_mass_task(
            self,
            external_userid: List[str],
            sender: str,
            content: str,
            task_name: str,
            image_urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """创建群发任务（带图片：下载→压缩→微盘上传→构造附件）"""
        logger.info(f"📨 创建群发任务: {task_name} (发送人: {sender}, 接收人: {len(external_userid)}个)")
        access_token = await self._get_access_token()
        if not access_token:
            return {"success": False, "error": "无法获取AccessToken"}

        session = await self._get_session()
        full_content = content
        attachments = []  # 图片附件列表（最终要传给群发接口）

        # 修改后正确的代码部分：
        # ---------------------- 核心：图片处理+临时素材上传 ----------------------
        if image_urls:
            logger.info(f"准备处理 {len(image_urls)} 张图片（临时素材上传模式）...")
            for img_url in image_urls:
                if not img_url:
                    continue

                try:
                    # 1. 下载图片（300秒超时）
                    logger.debug(f"下载图片: {img_url}")
                    img_timeout = aiohttp.ClientTimeout(total=300)
                    async with session.get(img_url, timeout=img_timeout) as resp:
                        if resp.status != 200:
                            logger.error(f"❌ 图片下载失败（状态码{resp.status}）: {img_url}")
                            continue
                        img_data = await resp.read()
                        file_name = img_url.split("/")[-1] or "temp_image.png"

                    # 2. 压缩图片（必须≤2MB，避免上传失败）
                    max_size = 2 * 1024 * 1024  # 2MB限制
                    if len(img_data) > max_size:
                        logger.warning(f"⚠️ 图片过大({len(img_data) // 1024}KB)，开始压缩: {file_name}")
                        try:
                            from PIL import Image
                            import io

                            # 步骤1：按比例缩放（最大1000×1000像素）
                            with Image.open(io.BytesIO(img_data)) as img:
                                max_dim = 1000
                                width, height = img.size
                                if width > max_dim or height > max_dim:
                                    ratio = min(max_dim / width, max_dim / height)
                                    new_size = (int(width * ratio), int(height * ratio))
                                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                                    logger.debug(f"📏 图片缩放: {width}×{height} → {new_size[0]}×{new_size[1]}")

                                # 步骤2：按格式压缩（PNG用无损压缩，JPEG用质量控制）
                                img_byte_arr = io.BytesIO()
                                if img.format == "PNG":
                                    img.save(img_byte_arr, format="PNG", compress_level=9, optimize=True)
                                else:
                                    quality = 80
                                    while True:
                                        img_byte_arr.seek(0)
                                        img.save(img_byte_arr, format="JPEG", quality=quality, optimize=True)
                                        compressed_data = img_byte_arr.getvalue()
                                        if len(compressed_data) <= max_size or quality <= 30:
                                            break
                                        quality -= 5

                                img_data = img_byte_arr.getvalue()
                                logger.debug(
                                    f"✅ 压缩完成: {len(img_data) // 1024}KB（质量{quality if img.format != 'PNG' else '无损'}）")
                        except Exception as e:
                            logger.error(f"❌ 图片压缩失败: {str(e)}，跳过此图")
                            continue

                    # 3. 使用临时素材接口上传（无需docid）
                    media_id = await self.upload_temp_media(img_data, file_name)  # 关键修改点
                    if not media_id:
                        logger.warning(f"⚠️ 跳过图片: {file_name}（素材上传失败）")
                        continue

                    # 4. 构造群发图片附件（关键：符合企微群发接口格式）
                    attachments.append({
                        "msgtype": "image",
                        "image": {"media_id": media_id}
                    })
                    logger.debug(f"✅ 图片附件添加成功: {file_name}（media_id: {media_id[:20]}...）")

                except Exception as e:
                    logger.error(f"图片处理异常: {str(e)}（URL: {img_url}）")
                    continue
        # ---------------------- 图片处理结束 ----------------------

        # 5. 构造群发请求体（attachments不为空即带图片）
        payload = {
            "chat_type": "single",
            "external_userid": external_userid,
            "sender": sender,
            "text": {"content": full_content},
            "attachments": attachments  # 图片附件在这里！
        }
        logger.debug(f"群发请求体（含附件）: {json.dumps(payload, ensure_ascii=False)[:500]}...")

        # 6. 调用群发接口（修复缩进+规范变量名）
        try:
            # 替换中文变量名为英文（更规范）
            mass_url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/add_msg_template?access_token={access_token}"
            async with session.post(mass_url, json=payload, timeout=self.timeout) as resp:
                result = await resp.json()
                logger.debug(f"群发响应: {json.dumps(result)}")
                if result.get("errcode") == 0:
                    logger.info(f"✅ 群发任务创建成功: {task_name}（msgid: {result['msgid']}）")
                    return {
                        "success": True,
                        "msgid": result["msgid"],
                        "attachments_count": len(attachments)
                    }
                else:
                    error_msg = result.get("errmsg", "未知错误")
                    logger.error(f"❌ 群发任务失败: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "errcode": result.get("errcode")
                    }
        # 关键：except 与上方 try 缩进一致！
        except Exception as e:
            logger.error(f"群发请求异常: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


    async def close(self):
        """关闭会话，释放资源"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("✅ 已关闭企业微信会话")
            self._session = None


async def fetch_task_table(docid: str, sheet_id: str, view_id: str) -> List[Dict]:
    """从智能表查询任务数据"""
    logger.debug(f"查询任务表: docid={docid}, sheet_id={sheet_id}, view_id={view_id}")
    query_params = {
        "action": "通用查询表单",
        "company": "拉伸大师",
        "WordList": {
            "docid": docid,
            "sheet_id": sheet_id,
            "view_id": view_id
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            logger.debug(f"请求任务数据到: {API_URL}")
            logger.debug(f"请求参数: {json.dumps(query_params, ensure_ascii=False)}")

            async with session.post(
                    API_URL,
                    headers=HEADERS,
                    json=query_params,
                    timeout=60
            ) as resp:
                result = await resp.json()
                logger.debug(f"任务表响应: {json.dumps(result, ensure_ascii=False)[:1000]}...")

                if result.get("success") and isinstance(result.get("data"), list):
                    logger.info(f"✅ 成功查询到 {len(result['data'])} 条任务数据")
                    return result["data"]
                else:
                    error_msg = result.get("message", "未知错误")
                    logger.error(f"❌ 查询任务表失败: {error_msg}")
                    if "data" in result:
                        logger.debug(f"响应数据: {json.dumps(result['data'], ensure_ascii=False)[:500]}...")
                    return []
    except Exception as e:
        logger.error(f"❌ 查询任务表异常: {str(e)}")
        return []


def parse_date_field(date_value):
    """解析日期字段，支持时间戳和日期字符串格式"""
    # 第一步：提取日期原始字符串（处理字典/列表/直接字符串三种场景）
    if isinstance(date_value, dict):
        # 场景1：字典格式（如{"text": "2025年8月29日"}）
        date_str = date_value.get("text", "")
    elif isinstance(date_value, list) and len(date_value) > 0 and isinstance(date_value[0], dict):
        # 场景2：列表包裹的字典（如[{"text": "2025-08-29"}]）
        date_str = date_value[0].get("text", "")
    else:
        # 场景3：直接字符串（如"1756396800000"或"2025年8月29日"）
        date_str = str(date_value).strip()

    # 第二步：尝试解析为毫秒时间戳
    try:
        timestamp = int(date_str)
        timestamp_sec = timestamp / 1000  # 毫秒转秒
        # UTC时间戳转本地时区时间（自动适配UTC+8）
        dt_local = datetime.fromtimestamp(timestamp_sec)
        return dt_local.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # 第三步：解析为常规日期字符串（支持多种格式）
        date_formats = ["%Y-%m-%d", "%Y年%m月%d日", "%Y/%m/%d"]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # 无法解析时返回原始字符串（便于日志排查）
        return date_str

async def process_tasks(wecom_handler: WeComTaskHandler, docid: str, sheet_id: str, view_id: str):
    """
    核心流程：读取任务表并创建群发任务
    """
    # 1. 从任务表获取任务数据
    logger.info("⌛ 正在查询任务表...")
    tasks = await fetch_task_table(docid, sheet_id, view_id)
    if not tasks:
        logger.warning("⚠️ 未获取到任务数据，流程终止")
        return

    logger.info(f"开始处理 {len(tasks)} 条任务...")

    # 获取今天的日期字符串（YYYY-MM-DD）
    today = datetime.now().strftime("%Y-%m-%d")
    logger.debug(f"今天的日期: {today}")

    valid_task_count = 0


    # 2. 遍历每条任务
    for task in tasks:
        logger.debug("=" * 80)
        logger.debug(f"任务原始数据: {json.dumps(task, ensure_ascii=False)[:500]}...")

        try:
            values = task.get("values", {})
            logger.debug(f"任务值: {json.dumps(values, ensure_ascii=False)[:500]}...")

            # 任务名称过滤
            task_name_cell = values.get("任务名", [{}])[0]
            task_name = task_name_cell.get("text", "")
            logger.debug(f"任务名称: {task_name} (类型: {type(task_name)}")

            if task_name != "领取礼品":
                logger.debug(f"跳过任务: 名称不匹配 ({task_name} != 测试群发任务)")
                continue

            # 发送日期过滤：修复取值格式错误
            send_date_raw = values.get("任务发送日期", "")
            # 处理「直接字符串」或「列表字典」两种格式
            if isinstance(send_date_raw, list) and len(send_date_raw) > 0:
                send_date_cell = send_date_raw[0]
            else:
                send_date_cell = send_date_raw
            # 解析日期
            send_date = parse_date_field(send_date_cell)

            logger.debug(f"发送日期: {send_date} (今天: {today})")

            if send_date != today:
                logger.debug(f"跳过任务: 日期不匹配 ({send_date} != {today})")
                continue

            # 提取所需字段
            # 外部用户ID（逗号分隔的字符串转换为列表）
            external_userid_cell = values.get("externalUserid", [{}])[0]
            external_userid_str = external_userid_cell.get("text", "")
            logger.debug(f"原始external_userid: {external_userid_str}")
            external_userid = [x.strip() for x in external_userid_str.split(",") if x.strip()]
            logger.debug(f"解析后external_userid: {external_userid}")

            # 话术内容
            content_cell = values.get("话术", [{}])[0]
            content = content_cell.get("text", "")
            logger.debug(f"话术内容: {content[:50]}...")

            # 任务图片：从字典列表中提取image_url（智能表图片字段的标准格式）
            task_images = values.get("任务图片", [])
            image_urls = []
            for img_info in task_images:
                if isinstance(img_info, dict):
                    img_url = img_info.get("image_url")
                    if img_url and img_url.startswith(("http://", "https://")):
                        image_urls.append(img_url)
            logger.debug(f"解析到图片URL: {image_urls}")

            # 发送人ID
            sender_list = values.get("待发送", [])
            logger.debug(f"原始待发送数据: {sender_list}")

            sender_ids = []
            for member in sender_list:
                if "user_id" in member:
                    sender_ids.append(member["user_id"])
                    logger.debug(f"从成员字段获取user_id: {member['user_id']}")
                elif "text" in member:
                    # 如果字段是文本类型而不是成员类型
                    user_ids = [id.strip() for id in member["text"].split(",")]
                    sender_ids.extend(user_ids)
                    logger.debug(f"从文本字段解析user_id: {user_ids}")

            # 如果sender_ids为空则使用空列表
            sender_ids = sender_ids if sender_ids else []
            logger.debug(f"发送人ID列表: {sender_ids}")

            valid_task_count += 1
            logger.info(f"🎯 找到匹配任务: {task_name} (发送日期: {send_date})")

        except Exception as e:
            logger.error(f"⚠️ 任务数据处理异常: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

        # 3. 为每个发送人创建群发任务
        if not sender_ids:
            logger.warning("⚠️ 没有找到发送人ID，跳过此任务")
            continue

        for sender in sender_ids:
            if external_userid and content and sender:
                logger.info(f"📨 开始处理任务: {task_name} (发送人: {sender})")

                # 创建群发任务
                logger.info(f"🕒 准备发送消息给 {len(external_userid)} 位客户...")
                result = await wecom_handler.create_mass_task(
                    external_userid=external_userid,
                    sender=sender,
                    content=content,
                    task_name=task_name,
                    image_urls=image_urls
                )

                # 处理结果
                if result["success"]:
                    logger.info(f"✅ 任务完成: {task_name} (msgid: {result.get('msgid')})")
                else:
                    logger.error(f"❌ 任务失败: {task_name} (错误: {result.get('error')})")
            else:
                missing = []
                if not external_userid: missing.append("external_userid")
                if not content: missing.append("content")
                if not sender: missing.append("sender")
                logger.warning(f"⚠️ 缺少必要字段: {', '.join(missing)}")

    logger.info(f"处理完成: 共处理 {valid_task_count} 个有效任务")
    if valid_task_count == 0:
        logger.warning("⚠️ 没有找到符合条件的任务")


async def main():
    """主函数"""
    logger.info("===== 🚀 企微任务表群发流程开始 =====")

    # 初始化企微处理器
    wecom_handler = WeComTaskHandler(CORPID, CORPSECRET)

    try:
        # 配置任务表参数
        docid = "dcPbCgiFT361NMXCjtOXHJRssdGcQcFBNmx-ej23sFFCjZJO1PmrZOGHDn_4dRUnUw1Nt-SD5-3fxIhNB42H1Gbw"
        sheet_id = "tyTqJV"
        view_id = "v5MarV"

        logger.info(f"任务表配置: docid={docid[:10]}..., sheet_id={sheet_id}, view_id={view_id}")

        # 执行核心流程
        await process_tasks(wecom_handler, docid, sheet_id, view_id)

    except Exception as e:
        logger.error(f"❌ 流程执行异常: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        # 关闭会话
        logger.info("正在清理资源...")
        await wecom_handler.close()
        logger.info("===== 🏁 流程执行结束 =====")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"主程序异常: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())