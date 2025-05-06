# DashScope SDK 版本不低于 1.23.1
import os
import dashscope
import base64

class TTS:
    def __init__(self, conn, model="qwen-tts"):
        self.conn = conn
        self.model = model
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.spk_id = "Chelsie"
        self.spk_id_support = ["Chelsie", "Cherry"]

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
        if not text:
            return
        response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            model="qwen-tts",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            text=text,
            voice=self.spk_id,#"Cherry",
            stream=True
        )
        for chunk in response:
            if not self.conn:
                print(chunk)
                continue

            try:
                audio_string = chunk["output"]["audio"]["data"]
            except:
                continue
            wav_bytes = base64.b64decode(audio_string)
            if len(wav_bytes) == 0:
                continue
            # print(f"Audio bytes length: {len(wav_bytes)}", chunk["output"]["finish_reason"])
            self.conn.send(wav_bytes, False)

if __name__ == "__main__":
    tts = TTS(None)
    tts.call("你是谁？")