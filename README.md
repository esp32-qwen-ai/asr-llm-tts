# 1. 简介
esp32-qwen-ai实现中枢，为前端esp32-io提供asr-llm-tts功能

# 2. 安装依赖
```
pip install -r requirements.txt
```

# 2. 配置环境变量
1. 本地asr-llm-tts api地址
```
# openai兼容的api地址，如ollama: http://127.0.0.1:11434/v1
LOCAL_LLM_API

# 本地部署语音识别，语音合成时需要指定
LOCAL_ASR_API
LOCAL_TTS_API
```

2. 非本地api地址
```
# 当前支持阿里百炼api_key
DASHSCOPE_API_KEY
```

3. 配置示例

支持 `.env` 方式配置

```
$ cat .env
LOCAL_LLM_API="http://win.lan:11434/v1"
LOCAL_ASR_API="http://win.lan:4000"
LOCAL_TTS_API="http://win.lan:5000"
DASHSCOPE_API_KEY="YOUR_KEY"
```

# 3. 运行
```
python main.py
```