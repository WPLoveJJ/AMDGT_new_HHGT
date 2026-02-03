import asyncio
import aiohttp
from aiohttp import ClientTimeout
import json
from typing import Optional, Dict, Any


class WeComMassSender:
    """企业微信群发任务工具类"""

    def __init__(self, corpid: str, corpsecret: str):
        """
        初始化企业微信配置

        :param corpid: 企业ID
        :param corpsecret: 应用密钥
        """
        self.corpid = corpid
        self.corpsecret = corpsecret
        self.access_token = None
        self.token_expires_at = 0  # 令牌过期时间（时间戳）
        self.mass_url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/add_msg_template"
        self.timeout = ClientTimeout(total=30)

    async def _get_access_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """获取企业微信访问令牌（内部方法）"""
        # 检查令牌是否有效，有效则直接返回
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
                    # 设置过期时间（提前100秒过期，避免网络延迟问题）
                    self.token_expires_at = asyncio.get_event_loop().time() + 7100
                    print(f"获取AccessToken成功，有效期7200秒")
                    return self.access_token
                else:
                    print(f"获取AccessToken失败：{result['errmsg']}（错误码：{result['errcode']}）")
                    return None
        except Exception as e:
            print(f"获取AccessToken异常：{str(e)}")
            return None

    async def create_mass_task(self,
                               external_userid: str,
                               sender: str,
                               content: str,
                               task_name: str = "测试任务") -> Dict[str, Any]:
        """
        创建企业微信群发任务

        :param external_userid: 接收人externalUserid
        :param sender: 发送人userid
        :param content: 发送内容
        :param task_name: 任务名称
        :return: 接口响应结果字典
        """
        async with aiohttp.ClientSession() as session:
            # 获取访问令牌
            access_token = await self._get_access_token(session)
            if not access_token:
                return {"success": False, "error": "无法获取AccessToken"}

            # 构建请求参数
            payload = {
                "chat_type": "single",
                "external_userid": [external_userid],
                "sender": sender,
                "allow_select": True,
                "text": {
                    "content": f"【{task_name}】\n{content}"
                },
                "attachments": []
            }

            # 发送请求
            try:
                url = f"{self.mass_url}?access_token={access_token}"
                async with session.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                ) as resp:
                    result = await resp.json()
                    print(f"接口响应: {json.dumps(result, ensure_ascii=False)}")

                    if result.get("errcode") == 0:
                        return {
                            "success": True,
                            "msgid": result.get("msgid"),
                            "response": result
                        }
                    else:
                        return {
                            "success": False,
                            "error": result.get("errmsg"),
                            "errcode": result.get("errcode"),
                            "response": result
                        }
            except Exception as e:
                error_msg = f"请求异常: {str(e)}"
                print(error_msg)
                return {"success": False, "error": error_msg}


async def main():
    """测试主函数"""
    # 企业微信配置（请替换为实际值）
    CORPID = "ww6fffc827ac483f35"
    CORPSECRET = "DxTJu-VblBUVmeQHGaEKvtEzXTRHFSgSfbJIfP39okQ"

    # 测试参数
    TEST_PARAMS = {
        "external_userid": "wmbb8UCgAAuOWhtSKPf4zIgFq9unW6tw",
        "sender": "wobb8UCgAA14GxUz9oQkGVa1n0R-3Sfg",
        "content": "您好！花山镇卫生院妇保科家庭医生提醒您",
        "task_name": "调试测试任务"
    }

    # 初始化发送器并测试
    sender = WeComMassSender(CORPID, CORPSECRET)
    result = await sender.create_mass_task(
        external_userid=TEST_PARAMS["external_userid"],
        sender=TEST_PARAMS["sender"],
        content=TEST_PARAMS["content"],
        task_name=TEST_PARAMS["task_name"]
    )

    # 输出测试结果
    if result["success"]:
        print(f"测试成功！群发任务ID: {result['msgid']}")
    else:
        print(f"测试失败！错误: {result['error']} (错误码: {result.get('errcode')})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
    print("测试结束")
