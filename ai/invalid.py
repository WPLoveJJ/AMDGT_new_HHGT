import asyncio
import aiohttp
from aiohttp import ClientTimeout
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class WeComTaskManager:
    """企业微信群发任务管理类（专注于任务查询与失效功能）"""

    def __init__(self, corpid: str, corpsecret: str):
        self.corpid = corpid
        self.corpsecret = corpsecret
        self.access_token = None
        self.token_expires_at = 0  # 令牌过期时间（时间戳）
        # 企业微信接口地址
        self.list_url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get_groupmsg_list"
        self.cancel_url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/cancel_groupmsg_send"
        self.timeout = ClientTimeout(total=30)

    async def _get_access_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """获取并缓存企业微信访问令牌"""
        # 检查令牌有效性
        if self.access_token and self.token_expires_at > asyncio.get_event_loop().time():
            return self.access_token

        token_url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
            f"?corpid={self.corpid}"
            f"&corpsecret={self.corpsecret}"
        )

        try:
            async with session.get(token_url, timeout=self.timeout) as resp:
                result = await resp.json()
                if result.get("errcode") == 0:
                    self.access_token = result["access_token"]
                    # 提前100秒过期，避免网络延迟问题
                    self.token_expires_at = asyncio.get_event_loop().time() + 7100
                    print("获取AccessToken成功")
                    return self.access_token
                else:
                    print(f"获取AccessToken失败：{result['errmsg']}（错误码：{result['errcode']}）")
                    return None
        except Exception as e:
            print(f"获取AccessToken异常：{str(e)}")
            return None

    async def get_tasks_created_today(self) -> List[str]:
        """
        查询今天创建的所有群发任务ID列表
        :return: 任务msgid列表
        """
        # 获取今天的日期范围（00:00:00 到 23:59:59）
        today = datetime.now().date()
        start_time = int(datetime.combine(today, datetime.min.time()).timestamp())
        end_time = int(datetime.combine(today, datetime.max.time()).timestamp() - 1)

        print(f"查询今天创建的任务范围: {today} {start_time}-{end_time}")

        async with aiohttp.ClientSession() as session:
            access_token = await self._get_access_token(session)
            if not access_token:
                print("无法获取访问令牌，查询任务失败")
                return []

            all_msgids = []
            cursor = ""  # 分页游标

            # 分页查询所有任务
            while True:
                payload = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": 100,  # 最大每页100条
                    "cursor": cursor,
                    "chat_type": "single"
                }

                try:
                    url = f"{self.list_url}?access_token={access_token}"
                    async with session.post(url, json=payload, timeout=self.timeout) as resp:
                        result = await resp.json()

                        if result.get("errcode") != 0:
                            print(f"查询任务失败：{result['errmsg']}（错误码：{result['errcode']}）")
                            break

                        # 提取当前页的任务ID
                        current_tasks = result.get("group_msg_list", [])
                        all_msgids.extend([task["msgid"] for task in current_tasks])

                        print(f"查询到 {len(current_tasks)} 个任务，总计 {len(all_msgids)} 个")

                        # 检查是否有下一页
                        cursor = result.get("next_cursor", "")
                        if not cursor:
                            break  # 无更多数据，退出循环

                except Exception as e:
                    print(f"查询任务时发生异常：{str(e)}")
                    break

            print(f"查询到今天创建的群发任务共 {len(all_msgids)} 个")
            return all_msgids

    async def cancel_tasks(self, msgids: List[str]) -> Dict[str, Any]:
        """
        批量停止群发任务
        :param msgids: 任务ID列表
        :return: 操作结果汇总
        """
        if not msgids:
            return {"success": True, "message": "没有需要停止的任务", "details": {}}

        async with aiohttp.ClientSession() as session:
            access_token = await self._get_access_token(session)
            if not access_token:
                return {"success": False, "message": "无法获取访问令牌", "details": {}}

            result_details = {}
            success_count = 0

            for msgid in msgids:
                try:
                    url = f"{self.cancel_url}?access_token={access_token}"
                    payload = {"msgid": msgid}

                    async with session.post(url, json=payload, timeout=self.timeout) as resp:
                        result = await resp.json()

                        if result.get("errcode") == 0:
                            success_count += 1
                            result_details[msgid] = {"success": True, "message": "停止成功"}
                        else:
                            result_details[msgid] = {
                                "success": False,
                                "message": result.get("errmsg"),
                                "errcode": result.get("errcode")
                            }

                except Exception as e:
                    result_details[msgid] = {"success": False, "message": f"请求异常：{str(e)}"}

                # 添加短暂延迟避免触发API限流
                await asyncio.sleep(0.1)

            return {
                "success": success_count > 0,
                "total": len(msgids),
                "success_count": success_count,
                "details": result_details
            }

    async def cancel_tasks_created_today(self) -> Dict[str, Any]:
        """
        停止今天创建的所有群发任务
        :return: 操作结果
        """
        # 1. 查询今天创建的任务
        msgids = await self.get_tasks_created_today()
        if not msgids:
            return {"success": True, "message": "今天没有查询到新创建的群发任务"}

        # 2. 停止查询到的任务
        return await self.cancel_tasks(msgids)


async def main():
    # 配置企业微信参数（替换为实际值）
    CORPID = "wwb0728887ce23a4ce"
    CORPSECRET = "vZ7mge0BomfhLaza43spNe9Wb8EmBdjmWQzxrhD10j4"

    manager = WeComTaskManager(CORPID, CORPSECRET)
    result = await manager.cancel_tasks_created_today()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
    print("操作完成")