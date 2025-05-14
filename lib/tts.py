import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_tts import TTS as localTTS
from ali_tts import TTS as aliTTS
import numpy as np

def adjust_volume(pcm_data, volume=None):
    def calculate_safe_gain(data):
        max_val = np.max(np.abs(data))
        if max_val == 0: return 1  # 防止除以0
        return np.iinfo(np.int16).max / max_val

    data = np.frombuffer(pcm_data, dtype=np.int16)
    volume_factor = volume / 50.

    if volume_factor is None:
        volume_factor = calculate_safe_gain(data)  # 如果未指定增益因子，则自动计算一个安全的最大增益
    adjusted_pcm = data * volume_factor
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    adjusted_pcm_clipped = np.clip(adjusted_pcm, min_int16, max_int16)

    return adjusted_pcm_clipped.astype(np.int16).tobytes()

class TTS:
    def __init__(self, conn):
        self.conn = conn
        if self._detect_local():
            self._init_local()
        else:
            self._init_bailian()

    def set_connection(self, conn):
        self.conn = conn
        self.tts.set_connection(conn)

    def _detect_local(self):
        try:
            resp = requests.get(localTTS.LOCAL_TTS_API_PING, timeout=0.5)
            return resp.status_code == 200
        except:
            return False

    def _init_local(self):
        self.provider = "本地"
        self.tts = localTTS(self.conn)
        print("use local tts")

    def _init_bailian(self):
        self.provider = "百炼"
        self.tts = aliTTS(self.conn)
        print("use ali tts")

    def get_provider(self):
        return self.provider

    def set_provider(self, provider):
        if self.provider == provider:
            return
        if provider == "本地":
            self._init_local()
        elif provider == "百炼":
            self._init_bailian()
        else:
            raise ValueError(f"unsupported provider {provider}")

    def is_local(self):
        return self.provider == "本地"

    def get_spk_id_support(self):
        return self.tts.get_spk_id_support()

    def get_spk_id(self):
        return self.tts.get_spk_id()

    def set_spk_id(self, spk_id):
        self.tts.set_spk_id(spk_id)

    def get_volume(self):
        return self.tts.get_volume()

    def set_volume(self, volume):
        self.tts.set_volume(volume)

    def call(self, data):
        self.tts.call(data)

if __name__ == "__main__":
    tts = TTS(None)
    tts.call("你是谁？")
