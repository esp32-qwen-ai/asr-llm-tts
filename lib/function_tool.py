import json
import os
from queue import Empty, Queue
import random
import re
import threading
from typing import Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool
import requests
import urllib.parse
from bs4 import BeautifulSoup
import subprocess

@register_tool('get_model_config')
class GetModelConfig(BaseTool):
    description = '当你想查询当前运行的asr，llm，tts（包含语音列表）模型服务配置时非常有用。'
    parameters = []

    def __init__(self, config):
        super().__init__(config)

        self.asr = config["args"]["asr"]
        self.llm = config["args"]["llm"]
        self.tts = config["args"]["tts"]

    def call(self, *args, **kwargs) -> str:
        result = {
            "asr": {
                "provider": self.asr.get_provider(),
                "model": "paraformer"
            },
            "llm": {
                "provider": self.llm.get_provider(),
                "model": self.llm.get_model()
            },
            "tts": {
                "provider": self.tts.get_provider(),
                "model": "cosyvoice",
                "spk_id": self.tts.get_spk_id(),
                "supported_spk_ids": self.tts.get_spk_id_support()
            }
        }
        return f"{json.dumps(result, indent=2)}"

@register_tool('set_model_config')
class SetModelConfig(BaseTool):
    description = '当你想修改当前运行的asr，llm，tts（包含语音切换）模型服务配置时非常有用。'
    parameters = [{
        'name': 'config_service',
        'type': 'string',
        'description': '用户指定配置哪个服务，可选参数列表为：["asr", "llm", "tts"]',
        'required': True,
    },
    {
        'name': 'config_key',
        'type': 'string',
        'description': '用户指定配置项名称，可选参数列表为：["spk_id", "provider"]',
        'required': True,
    },
    {
        'name': 'config_value',
        'type': 'string',
        'description': '''* 用户指定要配置的值。
        - 如果要配置`spk_id`：值必须在`supported_spk_ids`列表里，且必须全为英文小写字母；
        - 如果要配置`provider`：值必须为`本地`或`百炼`，其他值为非法值''',
        'required': True,
    }
    ]

    def __init__(self, config):
        super().__init__(config)

        self.asr = config["args"]["asr"]
        self.llm = config["args"]["llm"]
        self.tts = config["args"]["tts"]

    def call(self, params: str, *args, **kwargs) -> str:
        p = json.loads(params)
        if p["config_service"] == "tts":
            if p["config_key"] == "spk_id":
                self.tts.set_spk_id(p["config_value"])
            elif p["config_key"] == "provider":
                self.tts.set_provider(p["config_value"])
            else:
                raise ValueError(f"unsupport config_key {p['config_key']}")
        elif p["config_service"] == "llm":
            if p["config_key"] == "provider":
                self.llm.set_provider(p["config_value"])
            else:
                raise ValueError(f"unsupport config_key {p['config_key']}")
        elif p["config_service"] == "asr":
            if p["config_key"] == "provider":
                self.asr.set_provider(p["config_value"])
            else:
                raise ValueError(f"unsupport config_key {p['config_key']}")
        else:
            raise ValueError(f"unsupport config_service {p['config_service']}")
        return f'config done: {p["config_service"]}, {p["config_key"]}, {p["config_value"]}'

# like 'amap_weather' but return 4 days
@register_tool('amap_weather_plus')
class AmapWeatherPlus(BaseTool):
    description = '获取对应城市的天气数据'
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '城市/区具体名称，默认值为`杭州市`；如`北京市海淀区`请描述为`海淀区`。**注意**如果是市名如`杭州`必须添加`市`后缀为`杭州市`',
        'required': True
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

        # remote call
        self.url = 'https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={key}&extensions=all'

        import pandas as pd
        self.city_df = pd.read_excel(
            'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/AMap_adcode_citycode.xlsx')

        self.token = self.cfg.get('token', os.environ.get('AMAP_TOKEN', ''))
        assert self.token != '', 'weather api token must be acquired through ' \
            'https://lbs.amap.com/api/webservice/guide/create-project/get-key and set by AMAP_TOKEN'

    def get_city_adcode(self, city_name):
        filtered_df = self.city_df[self.city_df['中文名'] == city_name]
        if len(filtered_df['adcode'].values) == 0:
            raise ValueError(f'location {city_name} not found, availables are {self.city_df["中文名"]}')
        else:
            return filtered_df['adcode'].values[0]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        location = params['location']
        response = requests.get(self.url.format(city=self.get_city_adcode(location), key=self.token))
        data = response.json()
        if data['status'] == '0':
            raise RuntimeError(data)
        else:
            return json.dumps(data, indent=2)

@register_tool('exit_conversation')
class ExitConversation(BaseTool):
    description = "当你识别到用户表达告别意图时主动且必须调用这个函数"
    parameters = []

    def __init__(self, config):
        super().__init__(config)

        self.asr = config["args"]["asr"]
        self.llm = config["args"]["llm"]
        self.tts = config["args"]["tts"]

    def call(self, params: str, *args, **kwargs) -> str:
        self.llm.exit_conversation()

@register_tool('tts_volume')
class TTSVolume(BaseTool):
    description = "当需要查询或调整音量大小时这个函数非常有用"
    parameters = [{
        'name': 'operate',
        'type': 'string',
        'description': '操作类型，合法值为`get`或`set`',
        'required': True
    },
    {
        'name': 'volume',
        'type': 'int',
        'description': '当调整音量时需要传递这个参数，合法值在0到100之间的整数。建议：先获取当前值，再根据需要调整',
        'required': False
    }]

    def __init__(self, config):
        super().__init__(config)

        self.asr = config["args"]["asr"]
        self.llm = config["args"]["llm"]
        self.tts = config["args"]["tts"]

    def call(self, params: str, *args, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        if params["operate"] == "get":
            return self.tts.get_volume()
        elif params["operate"] == "set":
            self.tts.set_volume(params["volume"])

@register_tool('mp3_online')
class MP3Online(BaseTool):
    description = "在线搜索和播放音乐。`注意`：禁止将歌曲链接展示给用户"
    parameters = [{
        'name': 'recommend',
        'type': 'int',
        'description': '当要推荐歌曲时传入这个参数，合法值：1',
        'required': False
    },
    {
        'name': 'query',
        'type': 'string',
        'description': '要搜索的歌曲名或歌手名。',
        'required': False
    },
    {
        'name': 'play',
        'type': 'string',
        'description': '要播放音乐时传入这个参数，这个参数值是一个url必须来自`query`方法返回的结果，根据输入的url播放对应歌曲',
        'required': False
    }]

    def __init__(self, config):
        super().__init__(config)

        self.asr = config["args"]["asr"]
        self.llm = config["args"]["llm"]
        self.tts = config["args"]["tts"]

    def _recommend(self):
        API = "https://www.yymp3.com/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        resp = requests.get(API, headers=headers)
        with open('test.html', 'w') as f: f.write(resp.text)
        assert resp.status_code == 200, f"{resp.status_code}, {resp.text}"
        soup = BeautifulSoup(resp.text, 'html.parser')
        singer_links = soup.find_all('a', href=lambda x: x and x.startswith('/Play/') or x.startswith('Play/'), target='_yymp3')
        result = []
        for link in singer_links:
            name = link.get_text(strip=True)
            url = link['href']
            result.append({
                "song_name": name,
                "url": urllib.parse.urljoin("https://www.yymp3.com", url)
            })
        random.shuffle(result)
        return result

    def _search(self, query):
        API = "https://www.yymp3.com/search/"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        }
        data = {
            "tp": "1",
            "key": query
        }
        resp = requests.post(API, data=data, headers=headers)
        assert resp.status_code == 200, f"{resp.status_code}, {resp.text}"
        soup = BeautifulSoup(resp.text, 'html.parser')
        # <a href="/Singer/81.htm" target=_blank>周传雄</a>
        singer_links = soup.find_all('a', href=lambda x: x and x.startswith('/Play/') or x.startswith('Play/'), target='_yymp3')
        result = []
        for link in singer_links:
            name = link.get_text(strip=True)
            url = link['href']
            if name == "试听":
                continue
            result.append({
                "song_name": name,
                "url": urllib.parse.urljoin("https://www.yymp3.com", url)
            })
        return result

    def _download(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers)
        assert resp.status_code == 200, f"{resp.status_code}, {resp.text}"
        ret = re.search('song_data\[0\]="(.*?)";', resp.text)
        # 121347|青花|81|周传雄|new9/zhouchuanxiong10/8.wma|9540||
        ret = ret.group(1).split("|")
        for item in ret:
            if item.endswith(".wma"):
                item = item.replace(".wma", ".mp3")
                break
        else:
            raise ValueError(f"mp3 not found in {url}")
        uri = f"https://ting8.yymp3.com/{item}"
        resp = requests.get(uri)
        assert resp.status_code == 200, f"{resp.status_code}, {resp.text}"
        return resp.content

    def _convert_mp3_binary_to_pcm(self, mp3_binary_data):
        """
        将 MP3 二进制数据转换为 PCM 流。

        :param mp3_binary_data: MP3 的二进制数据（bytes）
        :yield: 每次读取的 PCM 二进制数据块
        """
        command = [
            'ffmpeg',
            '-v', 'error',
            '-f', 'mp3',
            '-i', 'pipe:0',
            '-ar', '24000',
            '-sample_fmt', 's16',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-f', 's16le',
            'pipe:1'
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1024 * 64
            )

            output_queue = Queue()
            stop_flag = threading.Event()

            def output_reader():
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    output_queue.put(chunk)
                stop_flag.set()

            # 启动线程读取输出
            thread = threading.Thread(target=output_reader)
            thread.start()

            process.stdin.write(mp3_binary_data)
            process.stdin.close()

            # 从队列中读取所有输出块
            while not stop_flag.is_set() or not output_queue.empty():
                try:
                    chunk = output_queue.get_nowait()
                    yield chunk
                except Empty:
                    continue

            # 等待输出线程结束
            thread.join()

            # 等待子进程结束
            process.wait()

            if process.returncode != 0:
                print("FFmpeg 执行失败！")
                print("错误信息：")
                print(process.stderr.read().decode('utf-8'))

        except Exception as e:
            print(f"发生异常：{e}")
            if 'process' in locals():
                process.kill()
                process.wait()

    def call(self, params: str, *args, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        if params.get("recommend", 0):
            result = self._recommend()
            if not result:
                raise ValueError(f"fail to recommend")
            return result
        if params.get("query", ""):
            result = self._search(params["query"])
            if not result:
                raise ValueError(f"《{params['query']}》 not found")
            return result
        if params.get("play", ""):
            content = self._download(params["play"])
            from tts import adjust_volume
            for pcm_data in self._convert_mp3_binary_to_pcm(content):
                self.tts.conn.send(adjust_volume(pcm_data, self.tts.get_volume()), False)
            return "已播放完成"