import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_tts import TTS as localTTS
from ali_tts import TTS as aliTTS

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

    def call(self, data):
        self.tts.call(data)

if __name__ == "__main__":
    tts = TTS(None)
    tts.call("你是谁？")
