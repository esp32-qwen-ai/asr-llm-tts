from dotenv import load_dotenv
load_dotenv()

import socket
from lib.asr import ASR
from lib.llm import LLM
from lib.tts import TTS
import struct
import json

# 配置服务器地址和端口
HOST = '0.0.0.0'  # 所有可用的网络接口
PORT = 3000       # 任意端口（确保未被占用）

class Request:
    HEADER_SIZE = 8
    MAGIC = b'bee'  # 3字节魔数

    WAV_FORMAT = 1
    PCM_FORMAT = 2

    def __init__(self):
        self.magic = Request.MAGIC  # 3字节魔数
        self.type = 0               # 1字节类型
        self.eof = 0                # 1字节标识结束
        self.dummy = 0              # 1字节保留字段
        self.length = 0             # 2字节长度

        self.data = b''

    @classmethod
    def from_bytes(cls, data: bytes):
        req = cls()

        # 使用 struct.unpack_from 解析前 8 字节
        # 格式字符串：3s B B B H -> 3字节字符串、1字节无符号char、1字节、1字节、2字节短整型
        unpacked = struct.unpack_from('<3sBBBH', data)
        req.magic = unpacked[0]
        req.type = unpacked[1]
        req.eof = unpacked[2]
        req.dummy = unpacked[3]
        req.length = unpacked[4]

        return req  

class Response:
    HEADER_SIZE = 8
    MAGIC = b'bee'  # 3字节魔数

    ASR_BIT = 0x2
    LLM_BIT = 0x1
    TTS_BIT = 0

    PCM_DATA = 1
    EXIT_CHAT = 2
    TOKEN = 3

    def __init__(self):
        self.magic = Request.MAGIC  # 3字节魔数
        self.type = 0               # 1字节类型
        self.eof = 0                # 1字节标识结束
        self.is_local = 0           # 1字节表明asr/llm/tts用在线还是离线
        self.length = 0             # 2字节长度

        self.data = b''

    @classmethod
    def from_bytes(cls, data: bytes):
        resp = cls()

        # 使用 struct.unpack_from 解析前 8 字节
        # 格式字符串：3s B B B H -> 3字节字符串、1字节无符号char、1字节、1字节、2字节短整型
        unpacked = struct.unpack_from('<3sBBBH', data)
        resp.magic = unpacked[0]
        resp.type = unpacked[1]
        resp.eof = unpacked[2]
        resp.is_local = unpacked[3]
        resp.length = unpacked[4]

        return resp

    def to_bytes(self):
        # 使用 struct.pack 打包数据
        # 格式字符串：3s B B B H -> 3字节字符串、1字节无符号char、1字节、1字节、2字节短整型
        packed = struct.pack('<3sBBBH', self.magic, self.type, self.eof, self.is_local, self.length)
        return packed + self.data

class Connection:
    DECODE_HEADER = 0
    DECODE_PAYLOAD = 1

    def __init__(self, s, asr, llm, tts, pcm_chunk_size=9600*2):
        self.socket = s
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.payload = b''
        self.request = None
        self.decode_state = Connection.DECODE_HEADER
        self.pcm_chunk_size = pcm_chunk_size
        self.pending_pcm = b''

    def recv(self, size):
        return self.socket.recv(size)

    def send(self, data, eof=0):
        resp = Response()
        resp.eof = eof
        resp.length = len(data)
        resp.is_local |= self.asr.is_local() << Response.ASR_BIT
        resp.is_local |= self.llm.is_local() << Response.LLM_BIT
        resp.is_local |= self.tts.is_local() << Response.TTS_BIT
        if isinstance(data, bytes):
            resp.type = Response.PCM_DATA
            resp.data = data
        else:
            resp.data = data.encode("utf-8")
            resp.type = Response.TOKEN
            try:
                j = json.loads(data)
                if j["response"] == "EXIT":
                    resp.type = Response.EXIT_CHAT
            except:
                pass
        self.socket.sendall(resp.to_bytes())

    def process(self, data):
        self.payload += data
        while len(self.payload) > 0:
            if self.decode_state == Connection.DECODE_HEADER:
                if len(self.payload) < Request.HEADER_SIZE:
                    return None
                else:
                    self.request = Request.from_bytes(self.payload[:Request.HEADER_SIZE])
                    if self.request.magic != Request.MAGIC:
                        print(f"Invalid magic number {self.request.magic}")
                        raise ValueError(f"Invalid magic number {self.request.magic}")
                    self.decode_state = Connection.DECODE_PAYLOAD
            elif self.decode_state == Connection.DECODE_PAYLOAD:
                if len(self.payload) < Request.HEADER_SIZE + self.request.length:
                    return None
                else:
                    self.request.data = self.payload[Request.HEADER_SIZE:Request.HEADER_SIZE + self.request.length]
                    self.payload = self.payload[Request.HEADER_SIZE + self.request.length:]
                    self.process_request(self.request)
                    self.request = None
                    self.decode_state = Connection.DECODE_HEADER

    def process_request(self, req):
        if req.type == Request.WAV_FORMAT:
            print(f"Received WAV format data: {len(req.data)} bytes")
            raise ValueError("WAV format not supported")
        elif req.type == Request.PCM_FORMAT:
            self.pending_pcm += req.data
            if len(self.pending_pcm) >= self.pcm_chunk_size:
                self.process_pcm(req.eof)
        else:
            raise ValueError(f"Unknown request type: {req.type}")

        if req.eof:
            self.process_pcm(True)
            self.send(b'', True)

    def process_pcm(self, is_finish):
        while len(self.pending_pcm) > 0:
            self.asr.send_audio_frame(self.pending_pcm[0:self.pcm_chunk_size], is_finish)
            self.pending_pcm = self.pending_pcm[self.pcm_chunk_size:]

def main():
    # 创建 socket 对象 (IPv4, TCP)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 设置端口复用（便于快速重启）
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 绑定地址和端口
        server_socket.bind((HOST, PORT))
        print(f"Server is listening on {HOST}:{PORT}...")

        # 开始监听，最大等待连接数为 1
        server_socket.listen(1)

        while True:
            client_socket = None
            conn = None
            try:
                print("Waiting for a connection...")
                client_socket, addr = server_socket.accept()
                print(f"Connected by {addr}")

                # pcm(wav) -> asr(text) -> llm(text) -> tts(speech) -> socket
                tts = TTS(None)
                llm = LLM(tts)
                asr = ASR(llm, 16000)

                # for agent
                LLM.init_agent(asr, llm, tts)

                conn = Connection(client_socket, asr, llm, tts)
                tts.set_connection(conn)

                while True:
                    data = conn.recv(4096)
                    if not data:
                        print("Client disconnected.")
                        break
                    conn.process(data)
            except Exception as e:
                # raise
                if client_socket:
                    client_socket.close()
                    client_socket = None

if __name__ == '__main__':
    main()
