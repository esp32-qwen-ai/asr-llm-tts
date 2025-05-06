import requests
import base64

class ASR:
    LOCAL_ASR_API_PING = os.getenv("LOCAL_ASR_API_PING")
    LOCAL_ASR_API = os.getenv("LOCAL_ASR_API") 

    def __init__(self, llm, sample_rate):
        self.llm = llm
        self.sample_rate = sample_rate
        self.text = ""

    def start(self):
        pass

    def stop(self):
        pass

    def send_audio_frame(self, data, is_finish=False):
        resp = requests.post(ASR.LOCAL_ASR_API, json={
            "pcm": base64.b64encode(data).decode("utf-8"),
            "sample_rate": self.sample_rate,
            "is_finish": is_finish
        })
        if resp.text.strip() != "":
            print(resp.text)
        self.text += resp.text
        if is_finish:
            self.llm.call(self.text)
            self.text = ""
