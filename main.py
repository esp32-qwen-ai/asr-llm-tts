from dotenv import load_dotenv
load_dotenv(override=True)

import socket
from lib.asr import ASR
from lib.llm import LLM
from lib.tts import TTS
import struct
import json
import webrtcvad
import yaml

class Vad:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, threshold=0.1):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.threshold = threshold
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # 0~3，3最严格

    def is_speech(self, pcm_data):
        frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 16位 = 2字节
        frames = [pcm_data[i:i+frame_size] for i in range(0, len(pcm_data), frame_size)]
        speech_frames = 0
        for frame in frames:
            if len(frame) < frame_size:
                continue
            if self.vad.is_speech(frame, sample_rate=self.sample_rate):
                speech_frames += 1
        return speech_frames / len(frames) >= self.threshold

class KWS:
    def __init__(self, asr, kw="hellohello"):
        self.asr = asr
        self.kw = kw.lower()
        self.wakeup = False
        self.vad = Vad()

    def is_wakeup(self, pcm):
        if self.wakeup:
            return True
        if not self.vad.is_speech(pcm):
            return False
        text = self.asr.convert_text(pcm)
        text = text.replace(" ", "").replace(",", "").replace("，", "").lower()
        if self.kw in text:
            print("--- WAKEUP ---")
            self.wakeup = True
            return True

    def exit(self):
        print("--- SLEEP ---")
        self.wakeup = False

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

    def __init__(self, s, asr, llm, tts, pcm_chunk_size=9600*2, config=None):
        self.socket = s
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.payload = b''
        self.request = None
        self.decode_state = Connection.DECODE_HEADER
        self.pcm_chunk_size = pcm_chunk_size
        self.pending_pcm = b''
        self.config = config
        self.kws = KWS(asr, kw=config["main"].get("kws", "hellohello"))

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
                    self.kws.exit()
                    return
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
        if not self.kws.is_wakeup(self.pending_pcm):
            self.pending_pcm = b''
            return
        while len(self.pending_pcm) > 0:
            self.asr.send_audio_frame(self.pending_pcm[0:self.pcm_chunk_size], is_finish and (len(self.pending_pcm) <= self.pcm_chunk_size))
            self.pending_pcm = self.pending_pcm[self.pcm_chunk_size:]

def load_config(fpath="config.yaml"):
    with open(fpath, 'r') as f:
        buf = f.read()
    config = yaml.safe_load(buf)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    return config

def main():
    config = load_config()
    host = config["main"]["host"]
    port = config["main"]["port"]
    # 创建 socket 对象 (IPv4, TCP)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # 设置端口复用（便于快速重启）
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 绑定地址和端口
        server_socket.bind((host, port))
        print(f"Server is listening on {host}:{port}...")

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
                tts = TTS(None, config)
                llm = LLM(tts, config)
                asr = ASR(llm, config)

                # for agent
                LLM.init_agent(asr, llm, tts)

                conn = Connection(client_socket, asr, llm, tts, config=config)
                tts.set_connection(conn)

                while True:
                    data = conn.recv(4096)
                    if not data:
                        print("Client disconnected.")
                        break
                    conn.process(data)
            except Exception as e:
                # raise
                print(e)
                if client_socket:
                    client_socket.close()
                    client_socket = None

if __name__ == '__main__':
    main()
