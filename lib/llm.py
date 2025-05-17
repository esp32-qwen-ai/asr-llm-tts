from collections import deque
import json
import queue
import threading
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import ASSISTANT
from qwen_agent.utils.output_beautify import typewriter_print
import requests
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 为了加载Agent
import function_tool as _

TOOL_CALL_S = '[TOOL_CALL]'
TOOL_CALL_E = ''
TOOL_RESULT_S = '[TOOL_RESPONSE]'
TOOL_RESULT_E = ''
THOUGHT_S = '[THINK]'

class LLM:
    LOCAL_LLM_API = os.getenv("LOCAL_LLM_API")
    LOCAL_LLM_API_PING = os.getenv("LOCAL_LLM_API")
    MIN_TEXT_TO_TTS = 120
    MAX_HISTORY = 10
    PROMPT = {"role": "system", "content": '''
# 🧠 智能语音助手对话行为规范

## 🎯 角色设定
你是一个具备自然语言理解和意图识别能力的中文智能语音助手，你能准确判断用户是否表达结束对话的意图，并在**仅一次**确认意图后调用 `function_tools.exit_conversation()`。

---

## 🧩 行为规则

### 1. **告别意图识别**
- 当用户输入中包含如“再见”、“拜拜”、“结束了”、“下次聊”等表达时，视为**触发告别意图**。

### 2. **告别对话处理**
- 若识别到告别意图：
  - **仅调用一次** `function_tools.exit_conversation()`；
  - **仅一次**正常回应用户，保持自然、友好；
  - **不得重复调用任何 exit/end 类工具函数**。

### 3. **普通对话处理**
- 若未识别到告别意图：
  - 正常回应问题；
  - **不调用任何退出类工具函数**。

---

## ⚠️ 注意事项

- 必须用中文回答；
- 回复应自然、口语化，避免机械式表达；
- 回复内容里，**禁止**出现任何链接，任何表情符号，任何链接
- **无论用户输入是否包含告别语，只能调用一次退出函数**；
- 若之前的对话已调用过 `exit_conversation`，则不再重复调用；
- 不得在输出中添加任何额外标记或控制指令。

---

## ✅ 示例说明

### ✅ 正确示例
> 输入：今天天气不错，我们下次聊，再见！
> 动作：[TOOL_CALL] function_tools.exit_conversation()
> 输出：很高兴和你聊天，祝你一切顺利！

### ❌ 错误示例（重复调用）
> 输入：今天天气不错，我们下次聊，再见！
> 动作：[TOOL_CALL] function_tools.exit_conversation()
> 输出：很高兴和你聊天，祝你一切顺利！
> 动作：[TOOL_CALL] function_tools.exit_conversation() ✅（错误：重复调用）

---

## 🚀 现在，请根据上述设定开始工作。

'''}

    TOOLS = [
        {
            'mcpServers': {  # You can specify the MCP configuration file
                'time': {
                    'command': 'uvx',
                    'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
                },
                # 'windows': {
                #     'url': 'http://win.lan:8000/sse'
                # },
                "amap-amap-sse": {
                    "url": f"https://mcp.amap.com/sse?key={os.getenv('AMAP_TOKEN')}"
                }
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
        {
            'name': 'exit_conversation',
            'args': {}
        },
        {
            'name': 'tts_volume',
            'args': {}
        },
        {
            'name': 'mp3_online',
            'args': {}
        },
        'web_search',
        'web_extractor',
        'code_interpreter',  # Built-in tools
    ]

    def __init__(self, tts, config):
        self.tts = tts
        self.config = config
        self.enable_thinking = self.config["llm"].get("enable_thinking", False)
        self.history = deque(maxlen=LLM.MAX_HISTORY)
        self.asr = None

        provider = self.config["llm"].get("provider", "")
        if provider == "本地" and self._detect_local():
            self._init_local()
        else:
            self._init_bailian()

        self.bot = None
        self.need_exit_conversation = False
        self.tts_thread = None
        self.tts_queue = queue.Queue()

    def _detect_local(self):
        try:
            resp = requests.get(LLM.LOCAL_LLM_API_PING, timeout=0.5)
            return resp.status_code in [200, 404]
        except Exception as ex:
            print(ex)
            return False

    def _init_local(self):
        self.provider = "本地"
        self.model = self.config["llm"].get("model", "qwen3:8b")
        self.model_server = LLM.LOCAL_LLM_API
        print("use local llm")

    def _init_bailian(self):
        self.provider = "百炼"
        self.model = self.config["llm"].get("model", "qwen3-235b-a22b")
        self.model_server = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        print("use ali llm")

    @staticmethod
    def init_agent(asr, llm, tts):
        for item in LLM.TOOLS:
            if isinstance(item, dict) and item.get("name", None):
                if item["name"] in ("get_model_config", "set_model_config", "exit_conversation", "tts_volume", "mp3_online"):
                    item["args"] = {
                        "asr": asr,
                        "llm": llm,
                        "tts": tts
                    }

    def get_provider(self):
        return self.provider

    def set_provider(self, provider):
        if self.provider == provider:
            return
        self.bot = None
        if provider == "本地":
            if not self._detect_local():
                raise ValueError(f"local provider not ready")
            self._init_local()
        elif provider == "百炼":
            self._init_bailian()
        else:
            raise ValueError(f"unsupported provider {provider}")

        self.config["llm"]["provider"] = provider

    def is_local(self):
        return self.provider == "本地"

    def get_model(self):
        return self.model

    def _process_history(self):
        while self.history[0]["role"] != "user":
            self.history.popleft()

    def _init_bot(self):
        llm_cfg = {
            'model': self.model,
            'model_server': self.model_server,
            'api_key': os.getenv("DASHSCOPE_API_KEY"),
            'generate_cfg': {
                'temperature': 0.7,
                'extra_body': {
                    'enable_thinking': self.enable_thinking
                }
            },
        }
        self.bot = Assistant(llm=llm_cfg,
                        function_list=LLM.TOOLS,
                        name='Qwen3 Tool-calling Demo',
                        system_message=LLM.PROMPT['content'],
                        description="I'm a demo using the Qwen3 tool calling.")

    def exit_conversation(self):
        print(f"\n\n------------ exit_conversation --------------")
        self.need_exit_conversation = True

    def _start_tts_thread(self):
        if self.tts_thread:
            return
        def tts_process(q):
            try:
                while True:
                    text = q.get()
                    if not text:
                        break
                    if self.tts:
                        if text.strip():
                            self.tts.call(text)
                    else:
                        print(f"\n--> tts: >>>>{text}<<<<")
            except Exception as e:
                print(e)

        self.tts_thread = threading.Thread(target = tts_process, args=(self.tts_queue,))
        self.tts_thread.start()

    def _stop_tts_thread(self):
        if self.tts_thread:
            self.tts_queue.put(None)
            self.tts_thread.join()
            self.tts_thread = None

    def call(self, text):
        if len(text.strip()) == 0:
            return

        self._start_tts_thread()

        if not self.bot:
            self._init_bot()

        prefix = ""
        if not self.enable_thinking and self.is_local():
            # 实测ollama本地模型当前没法通过enable_thinking=False方式关闭think，这里hack下
            prefix = "/no_think "
        self.history.append({"role": "user", "content": prefix + text})
        self._process_history()
        messages = list(self.history)
        # print(messages)
        fulltext_tts_done = 0
        fulltext = ""
        response_plain_text = ""
        for response in self.bot.run(messages=messages):
            response_plain_text = typewriter_print(response, response_plain_text)
            msg = response[-1]
            if msg['role'] == ASSISTANT and msg.get('content'):
                text = msg["content"]
                if text.startswith("<think>") and "</think>" not in text:
                    # QwenAgent为了内部实现简单，stream是个复杂的状态机。
                    # 如果中间有工具调用时，这里需要刷新text
                    fulltext = ""
                    assert fulltext_tts_done == 0
                    continue

                pattern = r'<think>.*?</think>'
                cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
                if not cleaned_text:
                    continue

                # print(cleaned_text[len(fulltext):], end="", flush=True)
                fulltext += cleaned_text[len(fulltext):]

                # 为了生成语音连贯性，这里牺牲实时性
                delta = len(fulltext) - fulltext_tts_done
                if delta > LLM.MIN_TEXT_TO_TTS:
                    self.tts_queue.put(fulltext[fulltext_tts_done:])
                    fulltext_tts_done = len(fulltext)

        print()
        if fulltext_tts_done < len(fulltext):
            self.tts_queue.put(fulltext[fulltext_tts_done:])
            fulltext_tts_done = len(fulltext)

        self.history.extend(response)

        self._stop_tts_thread()

        if self.need_exit_conversation and self.tts:
            self.tts.conn.send(json.dumps({"response": "EXIT"}), True)
            self.need_exit_conversation = False

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from asr import ASR
    from tts import TTS
    import yaml
    with open("config.yaml", 'r') as f:
        buf = f.read()
    config = yaml.safe_load(buf)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    LLM.LOCAL_LLM_API = os.getenv("LOCAL_LLM_API")
    LLM.LOCAL_LLM_API_PING = os.getenv("LOCAL_LLM_API")
    tts = TTS(None, config)
    llm = LLM(None, config)
    asr = ASR(llm, config)
    LLM.init_agent(asr, llm, tts)
    llm.call("现在用的哪个provider")
    llm.call("把所有模型都切换成百炼")
    llm.call("用本地llm模型")
    llm.call("你好，杭州现在几点了，讲个100字冷笑话")
    llm.call("没事了，下次聊")
    llm.call("随便播放一首中文歌曲")
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
    llm.call("杭州今天下雨吗？明天会下雨吗？要不要穿外套")
    llm.call("当前使用的asr模型配置是什么")
    llm.call("停止播放歌曲")
    llm.call("你好，杭州今天下雨吗？回答完之后就退出吧")
    llm.call("没其他问题了，退下吧")
