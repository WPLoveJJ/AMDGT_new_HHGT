import ctypes
import json
import base64
from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA
import time
import os
import hashlib

class WeWorkFinanceSDK:
    def __init__(self, corpid, chat_secret, private_key):
        self.corpid = corpid
        self.chat_secret = chat_secret
        self.private_key = private_key
        
        # 设置DLL搜索路径
        self.dll_dir = r'D:\Desktop\小成功会话存档'
        self.download_dir = r'D:\Desktop\小成功会话存档\downloads'
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
                    
                    # # 如果是图片消息，尝试下载
                    # if decrypted_msg.get('msgtype') == 'image':
                    #     print(f"检测到图片消息 {i+1}")
                    #     image_data = decrypted_msg.get('image', {})
                    #     sdkfileid = image_data.get('sdkfileid')
                    #     if sdkfileid and self.has_download_function:
                    #         print("开始下载图片...")
                    #         image_path = self.download_media(sdk, sdkfileid)
                    #         if image_path:
                    #             decrypted_msg['image']['local_path'] = image_path
                    #             print(f"图片下载成功: {image_path}")
                    #         else:
                    #             print("图片下载失败")
                    #     elif sdkfileid:
                    #         print("当前DLL版本不支持下载功能")
                            
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
    
    def get_chat_data_by_time_range(self, start_time: int, end_time: int, limit=100):
        """
        按时间范围拉取消息（Unix时间戳，秒级）
        :param start_time: 起始时间戳
        :param end_time: 结束时间戳
        :param limit: 每次拉取条数，默认100
        :return: 解密后的消息列表
        """
        sdk = None
        slice_ptr = None
        messages = []

        try:
            sdk = self.lib.NewSdk()
            self.lib.Init(sdk, self.corpid.encode(), self.chat_secret.encode())

            slice_ptr = self.lib.NewSlice()
            ret = self.lib.GetChatData(sdk, 0, limit, b"", b"", 10, slice_ptr)
            if ret != 0:
                raise Exception(f"GetChatData failed: {ret}")

            content = ctypes.string_at(self.lib.GetContentFromSlice(slice_ptr), self.lib.GetSliceLen(slice_ptr))
            data = json.loads(content.decode())
            chatdata = data.get('chatdata', [])

            for item in chatdata:
                decrypted = self.decrypt_message(item)
                ts = decrypted.get('msgtime', 0) // 1000
                if start_time <= ts <= end_time:
                    messages.append(decrypted)

            return messages

        finally:
            if slice_ptr:
                self.lib.FreeSlice(slice_ptr)
            if sdk:
                self.lib.DestroySdk(sdk)

    def get_chat_data_by_userid(self, userid: str, limit=100):
        """
        按用户ID拉取消息（匹配 from 或 tolist）
        :param userid: 企业微信 userid
        :param limit: 每次拉取条数，默认100
        :return: 包含该用户的消息列表
        """
        sdk = None
        slice_ptr = None
        messages = []

        try:
            sdk = self.lib.NewSdk()
            self.lib.Init(sdk, self.corpid.encode(), self.chat_secret.encode())

            slice_ptr = self.lib.NewSlice()
            ret = self.lib.GetChatData(sdk, 0, limit, b"", b"", 10, slice_ptr)
            if ret != 0:
                raise Exception(f"GetChatData failed: {ret}")

            content = ctypes.string_at(self.lib.GetContentFromSlice(slice_ptr), self.lib.GetSliceLen(slice_ptr))
            data = json.loads(content.decode())
            chatdata = data.get('chatdata', [])

            for item in chatdata:
                decrypted = self.decrypt_message(item)
                from_user = decrypted.get('from')
                to_list = decrypted.get('tolist', [])
                if userid == from_user or userid in to_list:
                    messages.append(decrypted)

            return messages

        finally:
            if slice_ptr:
                self.lib.FreeSlice(slice_ptr)
            if sdk:
                self.lib.DestroySdk(sdk)        

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
    
    def download_media(self, sdk, sdkfileid, save_dir=None):
        """下载媒体文件 - 修复版本"""
        if not self.has_download_function:
            print("当前DLL版本不支持下载功能")
            return None
            
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
        
        # 分片下载媒体文件
        indexbuf = b""  # 首次下载为空
        proxy = b""
        passwd = b""
        timeout = 30
        
        all_data = b""
        download_complete = False
        retry_count = 0
        max_retries = 3
        
        while not download_complete and retry_count < max_retries:
            media_data = self.lib.NewMediaData()
            if not media_data:
                print("创建MediaData失败")
                return None
                
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
            print("下载的文件内容为空")
            return None
        
        # 保存文件
        try:
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
            print(f"保存文件失败: {e}")
            return None

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

   
            
   

# 使用示例
def main():
    # 配置参数
    config = {
        'corpid': 'ww7ef6355ab47b84bf',
        'chat_secret': 'K-QqC18EWV-79hQAuL1nAi6dKinw9_Aba72bZda9w40',
        'private_key': '''-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCe0kP0rgtyvQ4T
5hD8lKLa2cCplwVuqXTPilJKUq63uG7+zSD3DnXO2m5BFPxXw9fnrjhvE+vQqvBU
kZuXS3Edp6HrFs91K++jIabKZQPiwufSi8dBgZEacoTgk+NKbS1WQHixUpe+466T
I6yUbE+es//HEzHkT7Wu4RBJYA6Z7b6llEhKGkbVO/HzDdsyBL80MCCzt30UAorw
O9h4qvpaeu8BGhTd0CE8rlOY+RvNUhEe9a/aRjiiKYVwJlS/dtS/0xq9Kk0wlajC
k1J6EG0hJOO6A1gtgKDWjqF1kfolKi1xJx+RRgB0oWWC2HmRqF7FBvK/AtvwVr7D
ITDmU9tfAgMBAAECggEAKVzBbfvV52/tNql4Sw+6yhxj3McEFynyzgpMHCckcZyK
67komK8fZ1vj0vdQMNtEY4YwFF2v0Ix0QtIZH3H6Z6q8tKkgJnfm1SlycsuUE36H
OYJFxn2j+KtSh9p8G2oLoKbxx9ymwIQzdwaXb6A1rDNcrdAF8lwACSd8Ql/N7bBg
Htyy5NpCsGBZ7qtK/+vhit+t5jqFp87TEUeWHUflYL6iQLEhRmtrnqiYRBhJgutl
Rp+Ork9UqYB88uhjJnci56AiwWUU7/RL6CTbM2TeKnbUfvzUis+KtPixiYfkQsFl
10/lvA6JMyQaP7abRm6sy4ubHj8J8Sl0XgplVl9ygQKBgQDP0VicRg2gwdut+Pck
KscaWhHV7Nq3yooK8lQ8aTJfAL+jEasCsHH5PXZ7DN7zfYK+nau11oqCm1qgxSJJ
Ggc6JWRQHEsgi/8KLH1zr4uvMK9pqZwPmOpesGrD3gn3luaFoEKbpuiYtV06No6q
24pxix25QMrKFjBKX8ldXnaYkQKBgQDDpNMS4m9JtA7ROHcpkjTsNINQBxxtaz2p
U7fcyElZxBpne7ZQ4QsXlkh7EHGL6nbThlDqoLX3h2u+hSODHTZZDASvIKoLJmZT
SRmZjYCUMUyD/vwjkZX+4vRPyiVZzyeCP47Th6sj7WIWAJ8Tc1AYJklbbDAlxIA+
o8vtMnqs7wKBgBO+7W9zzImwwcHG/RyHoAFl+3nbFnEOIo2lttFCoZGbTZod7KTF
xR6Mc3x/nZN/InQgIReW+fQs9LB/6qENgxBISz69YXoHM0VxMr5SOLB9rL0aa4Sj
3+j9VDemGkZXJTIfJxUJB+KyoNgOJQIg+AeP01QtztbiwtXeQhDID/axAoGAX0DM
k5EzZcsIu8ymF9NbePVISZpsuMlRyDM+fXv2WoIj6OZOqaon4GAHAgPLhLevCuWG
V/XPIqkISEZCN//tCgXT8gARSkRz5XDdyekfBQfLnSh0Ukz7npGZKY/eB4/YSI3x
WAPhyyg0bUn4KVwnTtq1NJHED19QoS7WhrHhEHkCgYAqSgTfEWAKso6YILSMdiXn
e04+xQuvvaHukbkL12ahl+6VFd/lg+UlrGbHDXrFrA1+S0aZIQ1GC7HPGRDNILzE
hO9OS4LZqM8cRhwNrkBSz2RDvsYv63HCK577AHvJWXeuad/1/wp+W7HgNGIR4/+O
eHLJSV/jdePpEOa2OdBCPA==
-----END PRIVATE KEY-----'''
    }
    
    try:
        print("开始初始化企业微信会话内容存档 SDK...")
        
        # 创建 SDK 实例
        sdk = WeWorkFinanceSDK(
            corpid=config['corpid'],
            chat_secret=config['chat_secret'],
            private_key=config['private_key']
        )
        
   
        
        # 然后再获取聊天数据
        print("开始获取聊天数据...")
        seq = 0
        messages, next_seq = sdk.get_chat_data(seq=seq, limit=10)
        
        print(f"\n最终结果:")
        print(f"获取到的消息数量: {len(messages)}")
        print(f"下一个 seq: {next_seq}")
        
        # 打印消息内容
        for i, msg in enumerate(messages):
            print(f"\n--- 消息 {i+1} ---")
            print(f"类型: {msg.get('msgtype')}")
            print(f"发送者: {msg.get('from')}")
            print(f"接收者: {msg.get('tolist', [])}")
            print(f"时间: {msg.get('msgtime')}")
            
            if msg.get('msgtype') == 'text':
                print(f"内容: {msg.get('text', {}).get('content')}")
            elif msg.get('msgtype') == 'image':
                print(f"图片消息, sdkfileid: {msg.get('image', {}).get('sdkfileid')}")
                if 'local_path' in msg.get('image', {}):
                    print(f"本地路径: {msg['image']['local_path']}")
            
    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()