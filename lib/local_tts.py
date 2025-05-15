import requests
import os

class TTS:
    LOCAL_TTS_API = os.getenv("LOCAL_TTS_API")
    LOCAL_TTS_API_PING = os.getenv("LOCAL_TTS_API") + "/ping"

    def __init__(self, conn, config=None):
        self.conn = conn
        self.config = config
        self.spk_id = self.config["tts"].get("spk_id", "chelsie")
        self.spk_id_support = requests.get(f"{TTS.LOCAL_TTS_API}/list_spk").json()
        if self.spk_id not in self.spk_id_support:
            print(f"invalid spk_id {self.spk_id}, change to {self.spk_id_support[0]}")
            self.spk_id = self.spk_id_support[0]
        self.volume = self.config["tts"].get("volume", 30)

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

    def get_volume(self):
        return self.volume

    def set_volume(self, volume):
        self.volume = volume

    def call(self, text):
        from tts import adjust_volume
        resp = requests.post(f"{TTS.LOCAL_TTS_API}/tts", json={
            "text": text,
            "spk_id": self.spk_id,
        }, stream=True)
        for pcm_data in resp.iter_content(chunk_size=4096):
            if not self.conn:
                print(len(pcm_data))
                continue
            pcm_data = adjust_volume(pcm_data, self.volume)
            self.conn.send(pcm_data, False)

if __name__ == "__main__":
    tts = TTS(None)
    tts.call("你是谁？")
