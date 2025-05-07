import requests
import os

class TTS:
    LOCAL_TTS_API = os.getenv("LOCAL_TTS_API")
    LOCAL_TTS_API_PING = os.getenv("LOCAL_TTS_API") + "/ping"

    def __init__(self, conn):
        self.conn = conn
        self.spk_id = "chelsie"
        self.spk_id_support = requests.get(f"{TTS.LOCAL_TTS_API}/list_spk").json()

    def set_connection(self, conn):
        self.conn = conn

    def get_spk_id(self):
        return self.spk_id

    def set_spk_id(self, spk_id):
        if spk_id not in self.spk_id_support:
            raise ValueError(f"unsupport spk_id {spk_id}")
        self.spk_id = spk_id

    def get_spk_id_support(self):
        return self.spk_id_support

    def call(self, text):
        resp = requests.post(f"{TTS.LOCAL_TTS_API}/tts", json={
            "text": text,
            "spk_id": self.spk_id,
        }, stream=True)
        for chunk in resp.iter_content(chunk_size=4096):
            if not self.conn:
                print(len(chunk))
                continue
            self.conn.send(chunk, False)

if __name__ == "__main__":
    tts = TTS(None)
    tts.call("你是谁？")
