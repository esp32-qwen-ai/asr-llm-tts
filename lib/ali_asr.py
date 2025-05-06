import dashscope
from dashscope.audio.asr import *

# Real-time speech recognition callback
class Callback(RecognitionCallback):
    def __init__(self, llm):
        self.llm = llm
        self.text = ""

    def on_open(self) -> None:
        print('RecognitionCallback open')

    def on_close(self) -> None:
        print('RecognitionCallback close')

    def on_complete(self) -> None:
        print('RecognitionCallback completed.')  # recognition completed
        if self.llm:
            self.llm.call(self.text)
        self.text = ""

    def on_error(self, message) -> None:
        print('RecognitionCallback task_id: ', message.request_id)
        print('RecognitionCallback error: ', message.message)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        # print(result.get_request_id(), result.get_usage(sentence))
        if 'text' in sentence:
            print('RecognitionCallback text: ', sentence['text'])
            eof=RecognitionResult.is_sentence_end(sentence)
            if eof:
                print(
                    'RecognitionCallback sentence end, request_id:%s, usage:%s'
                    % (result.get_request_id(), result.get_usage(sentence)))
                self.text += sentence['text']

class ASR:
    def __init__(self, llm, sample_rate=8000, format_pcm='pcm'):
        self.llm = llm
        self.sample_rate = sample_rate
        self.format_pcm = format_pcm
        self.callback = Callback(llm)

        self.recognition = Recognition(
            model='paraformer-realtime-v2',
            # 'paraformer-realtime-v1'、'paraformer-realtime-8k-v1'
            format=format_pcm,
            # 'pcm'、'wav'、'opus'、'speex'、'aac'、'amr', you can check the supported formats in the document
            sample_rate=sample_rate,
            # support 8000, 16000
            semantic_punctuation_enabled=False,
            callback=self.callback)

    def start(self):
        self.recognition.start()
    
    def stop(self):
        self.recognition.stop()

    def send_audio_frame(self, data, is_finish=False):
        # send audio data to recognition service
        self.recognition.send_audio_frame(data)

if __name__ == '__main__':
    asr = ASR(None, format_pcm = 'pcm')
    asr.start()
    # Simulate sending audio data
    with open('test.wav', 'rb') as f:
        f.read(44)  # Skip the WAV header
        audio_data = f.read()
        chunks = [audio_data[i:i + 1024] for i in range(0, len(audio_data), 1024)]
        for chunk in chunks:
            asr.send_audio_frame(chunk)
    asr.stop()