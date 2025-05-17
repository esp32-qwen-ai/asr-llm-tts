import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_asr import ASR as localASR
from ali_asr import ASR as aliASR

class ASR:
    def __init__(self, llm, config=None, sample_rate=16000, format_pcm='pcm'):
        self.llm = llm
        self.config = config
        self.sample_rate = sample_rate
        self.format_pcm = format_pcm
        provider = self.config["asr"].get("provider", "")
        if provider == "本地" and self._detect_local():
            self._init_local()
        else:
            self._init_bailian()
        self.asr_running = False

    def _detect_local(self):
        try:
            resp = requests.get(localASR.LOCAL_ASR_API_PING, timeout=0.5)
            return resp.status_code == 200
        except:
            return False

    def _init_local(self):
        self.provider = "本地"
        self.asr = localASR(self.llm, self.sample_rate)
        print("use local asr")

    def _init_bailian(self):
        self.provider = "百炼"
        self.asr = aliASR(self.llm, self.sample_rate, self.format_pcm)
        print("use ali asr")

    def get_provider(self):
        return self.provider

    def set_provider(self, provider):
        if self.provider == provider:
            return
        self.stop()
        if provider == "本地":
            if not self._detect_local():
                raise ValueError(f"local provider not ready")
            self._init_local()
        elif provider == "百炼":
            self._init_bailian()
        else:
            raise ValueError(f"unsupported provider {provider}")

        self.config["asr"]["provider"] = provider

    def is_local(self):
        return self.provider == "本地"

    def start(self):
        if self.asr_running:
            return
        self.asr_running = True
        self.asr.start()

    def stop(self):
        if not self.asr_running:
            return
        self.asr_running = False
        self.asr.stop()

    def send_audio_frame(self, data, is_finish=False):
        self.start()
        self.asr.send_audio_frame(data, is_finish)
        if is_finish:
            self.stop()

    def convert_text(self, data):
        return self.asr.convert_text(data)