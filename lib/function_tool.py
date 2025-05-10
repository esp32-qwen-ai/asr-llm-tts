import json
import os
from typing import Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool
import requests

@register_tool('get_model_config')
class GetModelConfig(BaseTool):
    description = '当你想查询当前运行的asr，llm，tts模型服务配置时非常有用。'
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
    description = '当你想修改当前运行的asr，llm，tts模型服务配置时非常有用。'
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
        'description': '用户指定要配置的值',
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
        'description': '城市/区具体名称，如`北京市海淀区`请描述为`海淀区`',
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