import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_asr import ASR as localASR
from ali_asr import ASR as aliASR

class ASR:
    def __init__(self, llm, sample_rate=8000, format_pcm='pcm'):
        self.llm = llm
        self.sample_rate = sample_rate
        self.format_pcm = format_pcm
        if self._detect_local():
            self.provider = "本地"
            print("use local asr")
            self.asr = localASR(llm, sample_rate)
        else:
            self.provider = "百炼"
            print("use ali asr")
            self.asr = aliASR(llm, sample_rate, format_pcm)
        self.asr_running = False

    def _detect_local(self):
        try:
            resp = requests.get(localASR.LOCAL_ASR_API_PING, timeout=0.5)
            return resp.status_code == 200
        except:
            return False

    def get_provider(self):
        return self.provider

    def is_local(self):
        return self.provider == "本地"

    def start(self):
        self.asr.start()

    def stop(self):
        self.asr.stop()

    def send_audio_frame(self, data, is_finish=False):
        if not self.asr_running:
            self.asr_running = True
            self.asr.start()
        self.asr.send_audio_frame(data, is_finish)
        if is_finish:
            self.asr_running = False
            self.asr.stop()
