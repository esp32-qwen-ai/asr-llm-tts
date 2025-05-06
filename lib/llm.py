from collections import deque
import json
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import ASSISTANT, FUNCTION
from qwen_agent.utils.output_beautify import typewriter_print
import requests
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import function_tool

TOOL_CALL_S = '[TOOL_CALL]'
TOOL_CALL_E = ''
TOOL_RESULT_S = '[TOOL_RESPONSE]'
TOOL_RESULT_E = ''
THOUGHT_S = '[THINK]'

class LLM:
    LOCAL_LLM_API = os.getenv("LOCAL_LLM_API")
    LOCAL_LLM_API_PING = os.getenv("LOCAL_LLM_API_PING")
    MIN_TEXT_TO_TTS = 60
    MAX_HISTORY = 10
    PROMPT = {"role": "system", "content": '''
# 角色设定
你是一个乐于助人的智能助手，具备自然语言理解能力，能够识别用户表达中的告别意图，并能调用各种功能工具来辅助完成任务。

## 核心功能与行为规则

### 🎯 主要任务：
1. **普通对话处理**：
   - 正常回应用户的提问或陈述。
   - 不需要添加任何额外的标记。

2. **包含告别语的对话处理**：
   - 当检测到用户的输入中包含特定的告别语（例如：“拜拜”、“再见”、“回头见”、“下次聊”等）时：
     - 正常回复用户内容；
     - **仅在**回复末尾追加固定格式：`{"response": "EXIT"}`;
     - **不要重复用户的问题或原始输入**。

3. **调用功能工具的处理**：
   - 在调用任何功能工具（例如 `function_tools`）之后：
     - 如果用户的输入中包含告别语，则在正常回复后追加 `{"response": "EXIT"}`；
     - 如果用户的输入中不包含告别语，则直接提供正常的回答，**不要**添加 `{"response": "..."}` 或其他任何形式的标记。

### 📌 注意事项：
- 回复应口语化、自然流畅，避免机械式回应；
- 确保只对明确包含告别语的输入添加 `{"response": "EXIT"}`；
- 对于非退出场景下的任何输入，直接提供正常的回答，**不要**添加 `{"response": "..."}` 或其他任何形式的标记；
- 在调用功能工具后，根据用户输入的内容决定是否追加 `{"response": "EXIT"}`。

## 示例交互

### ✅ 示例 1（包含告别语）
- 输入：“今天天气真好，我们改天再聊吧，拜拜！”
- 输出：很高兴和你聊天！祝你有美好的一天！{"response": "EXIT"}

### ❌ 示例 2（不应添加 EXIT 的普通对话）
- 输入：“今天是几月几号?”
- 输出：今天是2025年5月5日。

### ✅ 示例 3（调用功能工具）
- 输入：“请帮我查询一下今天的天气。”
- 功能调用：`function_tools.get_weather()`
- 输出：今天天气晴朗，最高气温28度，最低气温16度。

### ✅ 示例 4（调用功能工具后识别到告别语）
- 输入：“请帮我查询一下今天的天气，然后我们就结束了，拜拜！”
- 功能调用：`function_tools.get_weather()`
- 输出：今天天气晴朗，最高气温28度，最低气温16度。{"response": "EXIT"}

### ❌ 示例 5（调用功能工具后未识别到告别语）
- 输入：“请帮我查询一下今天的天气，谢谢！”
- 功能调用：`function_tools.get_weather()`
- 输出：今天天气晴朗，最高气温28度，最低气温16度。

## 结束语
现在，请根据上述指示开始工作。
'''}

    TOOLS = [
        {
            'mcpServers': {  # You can specify the MCP configuration file
                'time': {
                    'command': 'uvx',
                    'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
                },
                # 'fetch': {
                #     'command': 'uvx',
                #     'args': ['mcp-server-fetch']
                # },
                # "memory": {
                #     "command": "npx",
                #     "args": ["-y", "@modelcontextprotocol/server-memory"]
                # },
                # "filesystem": {
                #     "command": "npx",
                #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/caft/"]
                # },
            }
        },
        {
            'name': 'get_model_config',
            'args': {}
        },
        {
            'name': 'set_model_config',
            'args': {}
        },
        'code_interpreter',  # Built-in tools
    ]

    def __init__(self, tts, enable_thinking=False):
        self.tts = tts
        self.enable_thinking = enable_thinking
        self.history = deque(maxlen=LLM.MAX_HISTORY)
        self.asr = None
        self.tts_text = ""
        if self._detect_local():
            self.provider = "本地"
            self.model = "qwen3:8b"
            self.model_server = LLM.LOCAL_LLM_API
            print("use local llm")
        else:
            self.provider = "百炼"
            self.model = "qwen-plus"
            self.model_server = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            print("use ali llm")
        self.llm_cfg = {
            'model': self.model,
            'model_server': self.model_server,
            'api_key': os.getenv("DASHSCOPE_API_KEY"),
            'generate_cfg': {
                'temperature': 0.7,
                'extra_body': {
                    'enable_thinking': enable_thinking
                }
            },
        }
        self.bot = None

    def _detect_local(self):
        try:
            resp = requests.get(LLM.LOCAL_LLM_API_PING, timeout=0.5)
            return resp.status_code == 200
        except:
            return False

    @staticmethod
    def init_agent(asr, llm, tts):
        for item in LLM.TOOLS:
            if isinstance(item, dict) and item.get("name", None):
                if item["name"] == "get_model_config":
                    item["args"] = {
                        "asr": asr,
                        "llm": llm,
                        "tts": tts
                    }
                elif item["name"] == "set_model_config":
                    item["args"] = {
                        "asr": asr,
                        "llm": llm,
                        "tts": tts
                    }

    def get_provider(self):
        return self.provider

    def is_local(self):
        return self.provider == "本地"

    def get_model(self):
        return self.model

    def _process_history(self):
        while self.history[0]["role"] != "user":
            self.history.popleft()

    def _init_bot(self):
        self.bot = Assistant(llm=self.llm_cfg,
                        function_list=LLM.TOOLS,
                        name='Qwen3 Tool-calling Demo',
                        description="I'm a demo using the Qwen3 tool calling.")
    def call(self, text):
        if len(text.strip()) == 0:
            return
        self.tts_text = ""
        if not self.bot:
            self._init_bot()
        prefix = "" if self.enable_thinking else "/no_think "
        self.history.append({"role": "user", "content": prefix + text})
        self._process_history()
        messages = [LLM.PROMPT] + list(self.history)
        # print(messages)
        fulltext_tts_done = 0
        fulltext = ""
        response_plain_text = ""
        need_exit = False
        in_thinking = True
        for response in self.bot.run(messages=messages, delta_stream = True):
            response_plain_text = typewriter_print(response, response_plain_text)
            for msg in response:
                if msg['role'] == ASSISTANT:
                    if msg.get('reasoning_content'):
                        assert isinstance(msg['reasoning_content'], str), 'Now only supports text messages'
                        text = f'{THOUGHT_S}\n{msg["reasoning_content"]}'
                    if msg.get('content'):
                        text = msg["content"]
                        fulltext += text[len(fulltext):]
                    if msg.get('function_call'):
                        text = f'{TOOL_CALL_S} {msg["function_call"]["name"]}\n{msg["function_call"]["arguments"]}'
                        # 出现function_call会刷新assistant content输出
                        fulltext = ""
                        fulltext_tts_done = 0
                elif msg['role'] == FUNCTION:
                    text = f'{TOOL_RESULT_S} {msg["name"]}\n{msg["content"]}'

            if not in_thinking:
                in_thinking = fulltext.find("<think>") != -1

            if in_thinking:
                pos = fulltext.rfind("</think>")
                if pos == -1:
                    continue
                in_thinking = False

            # 识别到退出指令立即退出
            try:
                pos = fulltext.find('{"response":')
                if pos != -1:
                    if json.loads(fulltext[pos:])["response"] == "EXIT":
                        need_exit = True
                    fulltext = fulltext[:pos]
                    break
            except:
                pass

            # 为了生成语音连贯性，这里牺牲实时性
            delta = len(fulltext) - fulltext_tts_done
            if delta > LLM.MIN_TEXT_TO_TTS:
                if self._tts(fulltext[fulltext_tts_done:], False):
                    fulltext_tts_done = len(fulltext)

        print()
        self.history.extend(response)

        if fulltext_tts_done < len(fulltext):
            self._tts(fulltext[fulltext_tts_done:], True)
            fulltext_tts_done = len(fulltext)

        if need_exit and self.tts:
            self.tts.conn.send(json.dumps({"response": "EXIT"}), True)

    def _tts(self, text, force):
        pattern = r'<think>.*?</think>'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        if not cleaned_text:
            return True
        tts_text = cleaned_text[len(self.tts_text):]
        if not force and len(tts_text) < LLM.MIN_TEXT_TO_TTS:
            return False
        if self.tts:
            self.tts.call(tts_text)
        else:
            print(f"\n--> tts: >>>>{tts_text}<<<<")
        self.tts_text += tts_text
        return True

if __name__ == "__main__":
    import sys
    import os
    import time
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from asr import ASR
    from tts import TTS
    tts = TTS(None)
    llm = LLM(None)
    asr = ASR(llm, 16000)
    LLM.init_agent(asr, llm, tts)
    llm.call("随便播放一首中文歌曲")
    # time.sleep(3)
    llm.call("现在在播放哪一首歌曲")
    llm.call("停止播放歌曲")
    llm.call("计算下 1 + 2")
    llm.call("有哪些好听的音乐，推荐我看看")
    llm.call("播放第一首音乐")
    llm.call("当前模型配置是什么")
    llm.call("修改tts语音角色为luna")
    llm.call("南宋有哪些名人？")
    llm.call("哪些人是写诗的？")
    llm.call("哪些诗人是女的？")
    llm.call("南宋距离今天有多少年了？")
    llm.call("还有几个小时过年？")
    llm.call("今天下雨吗？明天会下雨吗？要不要穿外套")
    llm.call("当前使用的asr模型配置是什么")
    llm.call("你好，杭州今天下雨吗？回答完之后就退出吧")
    llm.call("没其他问题了，退下吧")
