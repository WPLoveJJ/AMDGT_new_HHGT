from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # 添加CORS支持
import ctypes
import json
import base64
from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA
import time
import os
import hashlib
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 启用CORS支持

class WeWorkFinanceSDK:
    # 您的现有SDK实现保持不变
    def __init__(self, corpid, chat_secret, private_key):
        self.corpid = corpid
        self.chat_secret = chat_secret
        self.private_key = private_key
        
        # 设置DLL搜索路径
        self.dll_dir = r'E:\AMDGT\plus - 副本\AMDGT-main\AMDGT-main\ai\unload\小成功会话存档\小成功会话存档'
        self.download_dir = r'E:\AMDGT\plus - 副本\AMDGT-main\AMDGT-main\ai\unload\小成功会话存档\小成功会话存档\downloads'
        os.environ['PATH'] = self.dll_dir + os.pathsep + os.environ['PATH']
        # 先加载依赖的DLL
        self._load_dependencies()
        
        # 加载主DLL
        self.dll_path = os.path.join(self.dll_dir, 'WeWorkFinanceSdk.dll')
        try:
            self.lib = ctypes.windll.LoadLibrary(self.dll_path)
            self._setup_function_prototypes()
        except Exception as e:
            raise Exception(f"加载 DLL 失败: {e}")
    
    def _load_dependencies(self):
        """加载依赖的DLL"""
        try:
            # 加载 libcrypto-3-x64.dll
            libcrypto_path = os.path.join(self.dll_dir, 'libcrypto-3-x64.dll')
            if os.path.exists(libcrypto_path):
                ctypes.windll.LoadLibrary(libcrypto_path)
                print("成功加载 libcrypto-3-x64.dll")
            
            # 加载 libcurl-x64.dll
            libcurl_path = os.path.join(self.dll_dir, 'libcurl-x64.dll')
            if os.path.exists(libcurl_path):
                ctypes.windll.LoadLibrary(libcurl_path)
                print("成功加载 libcurl-x64.dll")
                
        except Exception as e:
            print(f"加载依赖DLL时出错: {e}")
    
    def _setup_function_prototypes(self):
        """设置函数原型"""
        # 基本函数
        self.lib.NewSdk.restype = ctypes.c_void_p
        self.lib.DestroySdk.argtypes = [ctypes.c_void_p]
        self.lib.Init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.GetChatData.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint, 
                                       ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.NewSlice.restype = ctypes.c_void_p
        self.lib.FreeSlice.argtypes = [ctypes.c_void_p]
        self.lib.GetContentFromSlice.restype = ctypes.c_char_p
        self.lib.GetContentFromSlice.argtypes = [ctypes.c_void_p]
        self.lib.GetSliceLen.argtypes = [ctypes.c_void_p]
        self.lib.DecryptData.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        
        # 媒体数据相关函数
        try:
            # 创建和释放MediaData的函数
            self.lib.NewMediaData.restype = ctypes.c_void_p
            self.lib.FreeMediaData.argtypes = [ctypes.c_void_p]
            
            # 获取MediaData内容的函数
            self.lib.GetOutIndexBuf.restype = ctypes.c_char_p
            self.lib.GetOutIndexBuf.argtypes = [ctypes.c_void_p]
            
            self.lib.GetData.restype = ctypes.c_void_p
            self.lib.GetData.argtypes = [ctypes.c_void_p]
            
            self.lib.GetDataLen.restype = ctypes.c_int
            self.lib.GetDataLen.argtypes = [ctypes.c_void_p]
            
            self.lib.IsMediaDataFinish.restype = ctypes.c_int
            self.lib.IsMediaDataFinish.argtypes = [ctypes.c_void_p]
            
            # 正确的GetMediaData函数参数顺序
            self.lib.GetMediaData.argtypes = [
                ctypes.c_void_p,  # sdk
                ctypes.c_char_p,  # indexbuf
                ctypes.c_char_p,  # sdkfileid
                ctypes.c_char_p,  # proxy
                ctypes.c_char_p,  # passwd
                ctypes.c_int,     # timeout
                ctypes.c_void_p   # media_data
            ]
            self.lib.GetMediaData.restype = ctypes.c_int
            
            self.has_download_function = True
            print("检测到 GetMediaData 函数")
        except AttributeError as e:
            print(f"警告: DLL 中缺少某些函数: {e}")
            self.has_download_function = False
    
    def get_chat_data(self, seq=0, limit=100):
        """获取聊天数据"""
        sdk = None
        slice_ptr = None
        
        try:
            sdk = self.lib.NewSdk()
            if not sdk:
                raise Exception("创建 SDK 实例失败")
            
            ret = self.lib.Init(sdk, self.corpid.encode('utf-8'), self.chat_secret.encode('utf-8'))
            if ret != 0:
                raise Exception(f"初始化 SDK 失败, 返回值: {ret}")
            
            proxy = b""
            passwd = b""
            timeout = 10
            
            slice_ptr = self.lib.NewSlice()
            
            ret = self.lib.GetChatData(sdk, seq, limit, proxy, passwd, timeout, slice_ptr)
            if ret != 0:
                raise Exception(f"获取聊天数据失败, 返回值: {ret}")
            
            content_ptr = self.lib.GetContentFromSlice(slice_ptr)
            content_len = self.lib.GetSliceLen(slice_ptr)
            
            if content_len == 0:
                return [], seq
                
            content_data = ctypes.string_at(content_ptr, content_len)
            json_str = content_data.decode('utf-8')
            
            data = json.loads(json_str)
            
            if 'errcode' in data and data['errcode'] != 0:
                error_msg = data.get('errmsg', '未知错误')
                raise Exception(f"API 错误: {error_msg}")
            
            chat_data_list = data.get('chatdata', [])
            
            decrypted_messages = []
            for i, chat_data in enumerate(chat_data_list):
                try:
                    decrypted_msg = self.decrypt_message(chat_data)
                    decrypted_messages.append(decrypted_msg)
                except Exception as e:
                    print(f"解密消息失败: {e}")
                    continue
            
            return decrypted_messages, data.get('next_seq', seq + len(chat_data_list))
            
        except Exception as e:
            print(f"获取聊天数据时发生错误: {e}")
            return [], seq
        finally:
            if slice_ptr:
                self.lib.FreeSlice(slice_ptr)
            if sdk:
                self.lib.DestroySdk(sdk)
    


    def get_chat_data_by_userid(self, userid: str, limit=100, room_flag=None):
        """
        按用户 ID 拉取消息
        :param userid:   企业微信 userid
        :param limit:    拉取条数上限
        :param room_flag:
            None   -> 不区分单聊/群聊（默认）
            False  -> 只要单聊（roomid 为空）
            True   -> 只要群聊（roomid 非空）
        """
        seq = 0
        collected = []
        batch = 100
        max_rounds = 20  # 兜底

        for _ in range(max_rounds):
            if len(collected) >= limit:
                break

            msgs, next_seq = self.get_chat_data(seq=seq, limit=batch)
            if not msgs:
                break

            for m in msgs:
                if userid not in [m.get('from')] + m.get('tolist', []):
                    continue

                # 根据 room_flag 过滤
                is_room = bool(m.get('roomid'))
                if room_flag is None or is_room == room_flag:
                    collected.append(m)
                    if len(collected) >= limit:
                        break

            seq = next_seq

        return collected        

    def decrypt_message(self, chat_data):
        """解密单条消息"""
        encrypt_random_key = base64.b64decode(chat_data['encrypt_random_key'])
        encrypt_chat_msg = chat_data['encrypt_chat_msg']
        
        random_key = self.rsa_decrypt(encrypt_random_key)
        
        decrypt_slice = self.lib.NewSlice()
        try:
            ret = self.lib.DecryptData(random_key.encode('utf-8'), encrypt_chat_msg.encode('utf-8'), decrypt_slice)
            if ret != 0:
                raise Exception(f"解密数据失败, 返回值: {ret}")
            
            content_ptr = self.lib.GetContentFromSlice(decrypt_slice)
            content_len = self.lib.GetSliceLen(decrypt_slice)
            
            if content_len == 0:
                raise Exception("解密后内容为空")
                
            content_data = ctypes.string_at(content_ptr, content_len)
            decrypted_json = content_data.decode('utf-8')
            
            return json.loads(decrypted_json)
        finally:
            self.lib.FreeSlice(decrypt_slice)
    
    def rsa_decrypt(self, encrypted_data):
        """使用 RSA 私钥解密数据"""
        key = RSA.import_key(self.private_key)
        cipher = PKCS1_v1_5.new(key)
        decrypted_data = cipher.decrypt(encrypted_data, None)
        return decrypted_data.decode('utf-8')
    
    def download_media(self, sdkfileid, save_dir=None):
        """下载媒体文件"""
        if not self.has_download_function:
            raise Exception("当前DLL版本不支持下载功能")
            
        if save_dir is None:
            save_dir = self.download_dir
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 生成唯一的文件名
        timestamp = int(time.time())
        file_hash = hashlib.md5(sdkfileid.encode()).hexdigest()[:8]
        filename = f"image_{timestamp}_{file_hash}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        print(f"开始下载文件: {sdkfileid[:50]}...")
        
        sdk = None
        # 分片下载媒体文件
        indexbuf = b""  # 首次下载为空
        proxy = b""
        passwd = b""
        timeout = 30
        
        all_data = b""
        download_complete = False
        retry_count = 0
        max_retries = 3
        
        try:
            sdk = self.lib.NewSdk()
            self.lib.Init(sdk, self.corpid.encode('utf-8'), self.chat_secret.encode('utf-8'))
            
            while not download_complete and retry_count < max_retries:
                media_data = self.lib.NewMediaData()
                if not media_data:
                    raise Exception("创建MediaData失败")
                    
                try:
                    ret = self.lib.GetMediaData(
                        sdk, 
                        indexbuf, 
                        sdkfileid.encode('utf-8'), 
                        proxy, 
                        passwd, 
                        timeout, 
                        media_data
                    )
                    
                    print(f"GetMediaData 返回值: {ret}")
                    
                    if ret != 0:
                        print(f"下载文件失败, 返回值: {ret}, 重试 {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        time.sleep(1)
                        continue
                    
                    # 获取数据指针和长度
                    data_ptr = self.lib.GetData(media_data)
                    data_len = self.lib.GetDataLen(media_data)
                    
                    print(f"获取到数据长度: {data_len}")
                    
                    if data_ptr and data_len > 0:
                        # 读取数据
                        chunk_data = ctypes.string_at(data_ptr, data_len)
                        if chunk_data:
                            all_data += chunk_data
                            print(f"收到数据块: {len(chunk_data)} 字节, 累计: {len(all_data)} 字节")
                        else:
                            print("未读取到有效数据")
                    
                    # 检查是否下载完成
                    is_finish = self.lib.IsMediaDataFinish(media_data)
                    print(f"下载完成状态: {is_finish}")
                    
                    if is_finish == 1:
                        download_complete = True
                        print(f"文件下载完成，总大小: {len(all_data)} 字节")
                    else:
                        # 获取下一次的索引
                        next_indexbuf = self.lib.GetOutIndexBuf(media_data)
                        if next_indexbuf:
                            indexbuf = next_indexbuf
                            print(f"下一次索引: {indexbuf}")
                        else:
                            print("无法获取下一次索引，可能下载完成")
                            download_complete = True
                            
                except Exception as e:
                    print(f"下载过程中出错: {e}")
                    retry_count += 1
                    time.sleep(1)
                finally:
                    if media_data:
                        self.lib.FreeMediaData(media_data)
            
            if not all_data:
                raise Exception("下载的文件内容为空")
            
            # 保存文件
            with open(save_path, "wb") as f:
                f.write(all_data)
            
            print(f"文件保存成功: {save_path} (大小: {len(all_data)} 字节)")
            
            # 验证文件是否是有效的图片
            if self.is_valid_image_file(save_path):
                print("图片文件验证成功")
                return save_path
            else:
                print("图片文件验证失败，但已保存原始数据")
                # 即使验证失败也返回路径，因为可能是其他格式的文件
                return save_path
                
        except Exception as e:
            print(f"下载失败: {e}")
            raise
        finally:
            if sdk:
                self.lib.DestroySdk(sdk)

    def is_valid_image_file(self, file_path):
        """检查文件是否为有效的图片"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(20)
                
            # 检查常见的图片格式
            if header.startswith(b'\xFF\xD8\xFF'):  # JPEG
                print("检测到JPEG格式")
                return True
            elif header.startswith(b'\x89PNG'):  # PNG
                print("检测到PNG格式")
                return True
            elif header.startswith(b'GIF'):  # GIF
                print("检测到GIF格式")
                return True
            elif header.startswith(b'BM'):  # BMP
                print("检测到BMP格式")
                return True
            elif header.startswith(b'\x49\x49\x2A\x00') or header.startswith(b'\x4D\x4D\x00\x2A'):  # TIFF
                print("检测到TIFF格式")
                return True
            elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':  # WebP
                print("检测到WebP格式")
                return True
            else:
                print(f"未知文件格式: {header.hex()}")
                # 即使格式未知，如果文件大小合理也认为是有效的
                file_size = os.path.getsize(file_path)
                if file_size > 100:  # 至少100字节才认为是有效文件
                    return True
                return False
                
        except Exception as e:
            print(f"验证文件失败: {e}")
            return False

# 配置参数
config = {
    'corpid': 'ww7ef6355ab47b84bf',
    'chat_secret': 'K-QqC18EWV-79hQAuL1nAi6dKinw9_Aba72bZda9w40',
    'private_key': '''-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAntJD9K4Lcr0OE+YQ/JSi2tnAqZcFbql0z4pSSlKut7hu/s0g
9w51ztpuQRT8V8PX5644bxPr0KrwVJGbl0txHaeh6xbPdSvvoyGmymUD4sLn0ovH
QYGRGnKE4JPjSm0tVkB4sVKXvuOukyOslGxPnrP/xxMx5E+1ruEQSWAOme2+pZRI
ShpG1Tvx8w3bMgS/NDAgs7d9FAKK8DvYeKr6WnrvARoU3dAhPK5TmPkbzVIRHvWv
2kY4oimFcCZUv3bUv9MavSpNMJWowpNSehBtISTjugNYLYCg1o6hdZH6JSotcScf
kUYAdKFlgth5kahexQbyvwLb8Fa+wyEw5lPbXwIDAQABAoIBAClcwW371edv7Tap
eEsPusocY9zHBBcp8s4KTBwnJHGciuu5KJivH2db49L3UDDbRGOGMBRdr9CMdELS
GR9x+meqvLSpICZ35tUpcnLLlBN+hzmCRcZ9o/irUofafBtqC6Cm8cfcpsCEM3cG
l2+gNawzXK3QBfJcAAknfEJfze2wYB7csuTaQrBgWe6rSv/r4YrfreY6hafO0xFH
lh1H5WC+okCxIUZra56omEQYSYLrZUafjq5PVKmAfPLoYyZ3IuegIsFlFO/0S+gk
2zNk3ip21H781IrPirT4sYmH5ELBZddP5bwOiTMkGj+2m0ZurMuLmx4/CfEpdF4K
ZVZfcoECgYEAz9FYnEYNoMHbrfj3JCrHGloR1ezat8qKCvJUPGkyXwC/oxGrArBx
+T12ewze832Cvp2rtdaKgptaoMUiSRoHOiVkUBxLIIv/Cix9c6+LrzCvaamcD5jq
XrBqw94J95bmhaBCm6bomLVdOjaOqtuKcYsduUDKyhYwSl/JXV52mJECgYEAw6TT
EuJvSbQO0Th3KZI07DSDUAccbWs9qVO33MhJWcQaZ3u2UOELF5ZIexBxi+p204ZQ
6qC194drvoUjgx02WQwEryCqCyZmU0kZmY2AlDFMg/78I5GV/uL0T8olWc8ngj+O
04erI+1iFgCfE3NQGCZJW2wwJcSAPqPL7TJ6rO8CgYATvu1vc8yJsMHBxv0ch6AB
Zft52xZxDiKNpbbRQqGRm02aHeykxcUejHN8f52TfyJ0ICEXlvn0LPSwf+qhDYMQ
SEs+vWF6BzNFcTK+Ujiwfay9GmuEo9/o/VQ3phpGVyUyHycVCQfisqDYDiUCIPgH
j9NULc7W4sLV3kIQyA/2sQKBgF9AzJORM2XLCLvMphfTW3j1SEmabLjJUcgzPn17
9lqCI+jmTqmqJ+BgBwIDy4S3rwrlhlf1zyKpCEhGQjf/7QoF0/IAEUpEc+Vw3cnp
HwUHy50odFJM+56RmSmP3geP2EiN8VgD4csoNG1J+ClcJ07atTSRxA9fUKEu1oax
4RB5AoGAKkoE3xFgCrKOmCC0jHYl53tOPsULr72h7pG5C9dmoZfulRXf5YPlJaxm
xw16xawNfktGmSENRguxzxkQzSC8xITvTkuC2ajPHEYcDa5AUs9kQ77GL+txwiue
+wB7yVl3rmnf9f8Kflux4DRiEeP/jnhyyUlf43Xj6RDmtjnQQjw=
-----END RSA PRIVATE KEY-----'''
}

# 初始化SDK
try:
    sdk_instance = WeWorkFinanceSDK(
        corpid=config['corpid'],
        chat_secret=config['chat_secret'],
        private_key=config['private_key']
    )
    print("企业微信会话存档SDK初始化成功")
except Exception as e:
    print(f"SDK初始化失败: {e}")
    sdk_instance = None

def ts2str(ts):
    """毫秒时间戳转北京时间字符串"""
    return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

def format_message_detailed(message):
    """格式化消息为中文可读文本"""
    lines = []

    # 消息类型
    msgtype = message.get('msgtype', '未知')
    lines.append(f"类型: {msgtype}")

    # 发送者
    from_user = message.get('from', '未知')
    lines.append(f"发送者: {from_user}")

    # 接收者
    to_list = message.get('tolist', [])
    lines.append(f"接收者: {to_list}")

    # 时间
    msgtime = message.get('msgtime', 0)
    dt = datetime.fromtimestamp(msgtime / 1000)
    lines.append(f"时间: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # 内容
    if msgtype == 'text':
        content = message.get('text', {}).get('content', '')
        lines.append(f"内容: {content}")
    elif msgtype == 'image':
        sdkfileid = message.get('image', {}).get('sdkfileid', '')
        lines.append(f"图片消息，sdkfileid: {sdkfileid}")
    elif msgtype == 'voice':
        duration = message.get('voice', {}).get('play_length', 0)
        lines.append(f"语音消息，时长: {duration}秒")
    elif msgtype == 'video':
        duration = message.get('video', {}).get('play_length', 0)
        lines.append(f"视频消息，时长: {duration}秒")
    elif msgtype == 'file':
        filename = message.get('file', {}).get('filename', '')
        lines.append(f"文件消息，文件名: {filename}")
    else:
        lines.append("内容: [不支持的消息类型]")

    return "\n".join(lines)

def format_message(message):
    """格式化消息为更友好的格式"""
    formatted = {
        "msgtype": message.get("msgtype"),
        "from": message.get("from"),
        "tolist": message.get("tolist", []),
        "msgtime": message.get("msgtime"),
        "msgid": message.get("msgid"),
        "action": message.get("action"),
        "roomid": message.get("roomid", "")
    }
    
    # 根据消息类型添加特定内容
    msgtype = message.get("msgtype")
    if msgtype == "text":
        formatted["content"] = message.get("text", {}).get("content", "")
    elif msgtype == "image":
        formatted["image"] = {
            "sdkfileid": message.get("image", {}).get("sdkfileid", ""),
            "md5sum": message.get("image", {}).get("md5sum", ""),
            "filesize": message.get("image", {}).get("filesize", 0)
        }
    elif msgtype == "voice":
        formatted["voice"] = {
            "sdkfileid": message.get("voice", {}).get("sdkfileid", ""),
            "voice_size": message.get("voice", {}).get("voice_size", 0),
            "play_length": message.get("voice", {}).get("play_length", 0)
        }
    elif msgtype == "video":
        formatted["video"] = {
            "sdkfileid": message.get("video", {}).get("sdkfileid", ""),
            "filesize": message.get("video", {}).get("filesize", 0),
            "play_length": message.get("video", {}).get("play_length", 0)
        }
    elif msgtype == "file":
        formatted["file"] = {
            "sdkfileid": message.get("file", {}).get("sdkfileid", ""),
            "filename": message.get("file", {}).get("filename", ""),
            "filesize": message.get("file", {}).get("filesize", 0)
        }
    
    # 添加时间格式化
    if formatted["msgtime"]:
        dt = datetime.fromtimestamp(formatted["msgtime"] / 1000)
        formatted["datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted


# 新增辅助方法：持续分页直到拿到足够的消息
def pull_until(sdk, start_time=None, end_time=None, userid=None, max_cnt=500):
    seq = 0
    collected = []
    while len(collected) < max_cnt:
        msgs, next_seq = sdk.get_chat_data(seq=seq, limit=100)
        if not msgs:
            break
        for m in msgs:
            ts = m.get('msgtime', 0) // 1000
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            if userid and userid not in [m.get('from')] + m.get('tolist', []):
                continue
            collected.append(m)
        seq = next_seq
    return collected

# API路由
@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    if sdk_instance:
        return jsonify({"status": "ok", "message": "SDK已初始化"})
    else:
        return jsonify({"status": "error", "message": "SDK初始化失败"}), 500

@app.route('/api/messages', methods=['GET'])
def get_messages():
    """获取消息接口"""
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500
    
    try:
        seq = request.args.get('seq', 0, type=int)
        limit = request.args.get('limit', 100, type=int)
        
        messages, next_seq = sdk_instance.get_chat_data(seq=seq, limit=limit)
        
        # 格式化消息
        formatted_messages = [format_message(msg) for msg in messages]
        
        return jsonify({
            "status": "success",
            "data": {
                "messages": formatted_messages,
                "next_seq": next_seq,
                "count": len(formatted_messages)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/messages/by_user', methods=['GET'])
def get_messages_by_user():
    """按用户ID获取消息"""
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500
    
    try:
        userid = request.args.get('userid')
        limit = request.args.get('limit', 100, type=int)
        
        if not userid:
            return jsonify({"error": "需要提供userid参数"}), 400
        
        messages = sdk_instance.get_chat_data_by_userid(userid, limit)
        
        # 格式化消息
        formatted_messages = [format_message(msg) for msg in messages]
        
        return jsonify({
            "status": "success",
            "data": {
                "messages": formatted_messages,
                "count": len(formatted_messages)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/image', methods=['GET'])
def download_image():
    """下载图片接口"""
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500
    
    try:
        sdkfileid = request.args.get('sdkfileid')
        if not sdkfileid:
            return jsonify({"error": "需要提供sdkfileid参数"}), 400
        
        # 下载图片
        image_path = sdk_instance.download_media(sdkfileid)
        
        # 返回图片文件
        return send_file(
            image_path,
            as_attachment=True,
            download_name=os.path.basename(image_path),
            mimetype='image/jpeg'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/messages/advanced', methods=['GET'])
def get_messages_advanced():
    """高级查询接口：支持多种条件组合"""
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500
    
    try:
        # 获取查询参数
        seq = request.args.get('seq', 0, type=int)
        limit = request.args.get('limit', 100, type=int)
        userid = request.args.get('userid')
        start_time = request.args.get('start_time', type=int)
        end_time = request.args.get('end_time', type=int)
        msg_type = request.args.get('msg_type')
        
        # 获取所有消息
        messages, next_seq = sdk_instance.get_chat_data(seq=seq, limit=limit)
        
        # 应用过滤条件
        filtered_messages = []
        for msg in messages:
            # 用户ID过滤
            if userid:
                from_user = msg.get('from')
                to_list = msg.get('tolist', [])
                if userid != from_user and userid not in to_list:
                    continue
            
            # 时间范围过滤
            if start_time and end_time:
                msg_time = msg.get('msgtime', 0) // 1000
                if not (start_time <= msg_time <= end_time):
                    continue
            
            # 消息类型过滤
            if msg_type and msg.get('msgtype') != msg_type:
                continue
            
            filtered_messages.append(msg)
        
        # 格式化消息
        formatted_messages = [format_message(msg) for msg in filtered_messages]
        
        return jsonify({
            "status": "success",
            "data": {
                "messages": formatted_messages,
                "next_seq": next_seq,
                "count": len(formatted_messages)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/messages/detailed', methods=['GET'])
def get_messages_detailed():
    """获取详细格式消息接口"""
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500
    
    try:
        seq = request.args.get('seq', 0, type=int)
        limit = request.args.get('limit', 100, type=int)
        
        messages, next_seq = sdk_instance.get_chat_data(seq=seq, limit=limit)
        
        # 构建详细格式的输出
        formatted_messages = []
        for i, msg in enumerate(messages):
            formatted_msg = {
                "index": i + 1,
                "formatted_text": format_message_detailed(msg),
                "raw": msg  # 可选：包含原始消息数据
            }
            formatted_messages.append(formatted_msg)
        
        return jsonify({
            "status": "success",
            "data": {
                "messages": formatted_messages,
                "next_seq": next_seq,
                "count": len(formatted_messages)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

@app.route('/api/messages/split', methods=['GET'])
def get_messages_split():
    """
    按用户 ID 拉消息，并自动拆成单聊 / 群聊
    参数：
        userid : 必须
        limit  : 默认 100（实际条数，不是 SDK 拉取条数）
    返回：
        {
          "status": "success",
          "data": {
            "single": [...],   # 单聊
            "group" : [...],   # 群聊
            "count_single": 2,
            "count_group": 8
          }
        }
    """
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500

    userid = request.args.get('userid')
    limit  = request.args.get('limit', 100, type=int)
    if not userid:
        return jsonify({"error": "缺少 userid"}), 400

    single, group = [], []
    seq, batch = 0, 100          # 每次拉 100 条原始消息
    max_rounds = 20            # 兜底 20 轮，防止死循环

    for _ in range(max_rounds):
        if len(single) + len(group) >= limit:
            break

        msgs, next_seq = sdk_instance.get_chat_data(seq=seq, limit=batch)
        if not msgs:
            break

        for m in msgs:
            # 只要跟当前用户有关
            from_u = m.get('from')
            to_lst = m.get('tolist', [])
            if userid not in [from_u] + to_lst:
                continue

            # 判断群聊还是单聊
            if m.get('roomid'):          # 有 roomid 就是群聊
                group.append(format_message(m))
            else:                        # 没有就是单聊
                single.append(format_message(m))

            if len(single) + len(group) >= limit:
                break

        seq = next_seq

    return jsonify({
        "status": "success",
        "data": {
            "single": single,
            "group":  group,
            "count_single": len(single),
            "count_group":  len(group)
        }
    })

@app.route('/api/messages/single', methods=['GET'])
def get_messages_single():
    """
    仅返回指定 userid 的【单聊】消息
    """
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500

    userid = request.args.get('userid')
    limit  = request.args.get('limit', 100, type=int)
    if not userid:
        return jsonify({"error": "缺少 userid"}), 400

    msgs = sdk_instance.get_chat_data_by_userid(userid=userid, limit=limit, room_flag=False)
    return jsonify({"status": "success", "count": len(msgs), "messages": msgs})

@app.route('/api/messages/group', methods=['GET'])
def get_messages_group():
    """
    仅返回指定 userid 的【群聊】消息
    """
    if not sdk_instance:
        return jsonify({"error": "SDK未初始化"}), 500

    userid = request.args.get('userid')
    limit  = request.args.get('limit', 100, type=int)
    if not userid:
        return jsonify({"error": "缺少 userid"}), 400

    msgs = sdk_instance.get_chat_data_by_userid(userid=userid, limit=limit, room_flag=True)
    return jsonify({"status": "success", "count": len(msgs), "messages": msgs})      

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)