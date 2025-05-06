# 1. 简介
本仓库是esp32-ai实现中枢，为前端esp32-io提供asr-llm-tts功能

# 2. 安装依赖
```
pip install -r requirements.txt
```

# 2. 配置环境变量
1. 本地asr-llm-tts api地址
```
# openai兼容的api地址，如ollama: http://127.0.0.1:11434/v1
LOCAL_LLM_API
# 本地服务探活地址，如: http://127.0.0.1:11434
LOCAL_LLM_API_PING

# 
LOCAL_ASR_API
LOCAL_ASR_API_PING
LOCAL_TTS_API
LOCAL_TTS_API_PING
```

2. 非本地api地址
```
# 当前支持阿里百炼api_key
DASHSCOPE_API_KEY
```
