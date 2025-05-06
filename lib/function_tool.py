import json
import requests

from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('get_model_config')
class GetModelConfig(BaseTool):
    description = '当你想查询当前运行的asr，llm，tts模型服务配置时非常有用。'
    parameters = []

    def __init__(self, config):
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
