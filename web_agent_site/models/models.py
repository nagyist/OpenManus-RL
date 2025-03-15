"""
Model implementations. The model interface should be suitable for both
the ``site env'' and the ``text env''.
"""
import json
import random
import re

import requests

random.seed(4)


class BasePolicy:
    def __init__(self):
        pass

    def forward(self, observation, available_actions):
        """
        Args:
            observation (`str`):
                HTML string

            available_actions ():
                ...
        Returns:
            action (`str`): 
                Return string of the format ``action_name[action_arg]''.
                Examples:
                    - search[white shoes]
                    - click[button=Reviews]
                    - click[button=Buy Now]
        """
        raise NotImplementedError


class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        action = input('> ')
        return action


class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        if available_actions['has_search_bar']:
            action = 'search[shoes]'
        else:
            action_arg = random.choice(available_actions['clickables'])
            action = f'click[{action_arg}]'
        return action


class GPTPolicy(BasePolicy):
    prompt: str = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instruction.
Every round I will give you an observation and a list of available actions, \
you have to respond a action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
  - search[keywords]
  - click[value]
If action not valid, perform nothing.
Keywords in search is up to you, but value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:
{"thought": "HERE IS YOUR THOUGHT", "action": "HERE IS THE ACTION STRING"}
    """

    def __init__(self, gpt=3.5):
        super().__init__()
        self.url = "http://40.74.217.35:10001/api/openai/chat-completion"
        if gpt == 4:
            self.url = "http://40.74.217.35:10010/api/openai/chat-completion"
            gpt = "4"
            print("GPT-4！")
        else:
            gpt = "3.5-turbo"
            print("using model: ", gpt)
        self.gpt = gpt
        self.history = [{"role": "system", "content": self.prompt}]

    def forward(self, observation, available_actions):
        self.history.append({"role": "system", "content": f"Observation:\n{observation}\n\n"
                                                          f"Available Actions:\n{available_actions}"})
        while True:
            try:
                r = requests.post(self.url, json={
                    "model": f"gpt-{self.gpt}",
                    "messages": self.history,
                    # "temperature": 0,
                }, timeout=60)
                if r.status_code == 500 and "token" in r.text:
                    # token limit exceeded
                    print(r.text)
                    break
                assert r.status_code == 200, "HTTP Error: " + str(r.status_code) + r.text
                r = r.json()
                content = r["choices"][0]["message"]["content"].strip()
                self.history.append({"role": "assistant", "content": content})
                break
            except Exception as e:
                print(e)

        try:
            print(content)
            result = json.loads(content)["action"]
            return result
        except Exception as e:
            print(e)
            return None


class DavinciPolicy(BasePolicy):
    prompt: str = GPTPolicy.prompt

    def __init__(self, model="text-davinci-003"):
        super().__init__()
        self.url = "http://40.74.217.35:10001/api/openai/completion"
        self.history = f"Q:\n{self.prompt}\n"
        print(f"using model: {model}")
        self.model = model

    def forward(self, observation, available_actions):
        self.history += f"Q:\nObservation:\n{observation}\n\nAvailable Actions:\n{available_actions}\nA:\n"
        while True:
            try:
                r = requests.post(self.url, json={
                    "model": self.model,
                    "prompt": self.history,
                    "max_tokens": 256,
                    # "temperature": 0,
                }, timeout=60)
                if r.status_code == 500 and "token" in r.text:
                    # token limit exceeded
                    print(r.text)
                    break
                assert r.status_code == 200, "HTTP Error: " + str(r.status_code) + r.text
                r = r.json()
                content = r["choices"][0]["text"].strip()
                self.history += content + "\n"
                break
            except Exception as e:
                print(e)

        try:
            print(content)
            result = json.loads(content)["action"]
            return result
        except Exception as e:
            print(e)
            return None


class GLMPolicy(BasePolicy):
    prompt: str = GPTPolicy.prompt

    def __init__(self):
        super().__init__()
        self.url = "http://40.74.217.35:10001/api/openai/chat-completion"
        self.history = []

    def forward(self, observation, available_actions):
        for i in range(3):
            try:
                r = requests.post(self.url, json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": self.prompt},
                        *self.history,
                        {"role": "system", "content": f"Observation:\n{observation}\n\n"
                                                      f"Available Actions:\n{available_actions}"}
                    ],
                    # "temperature": 0,
                }, timeout=60)
                assert r.status_code == 200, "HTTP Error: " + str(r.status_code)
                r = r.json()
                content = r["choices"][0]["message"]["content"].strip()
                self.history.append({"role": "assistant", "content": content})
                break
            except Exception as e:
                print(e)

        return content


def chatglm_completion_call(history, query):
    MODEL_REQUEST_URL = "http://10.254.134.125:8001/completion/stream"
    headers = {
        "CHARSET": "utf-8",
        "Authorization": "85577713-4c52-48bd-bf65-57112ce337d2",
        "Content-Type": "application/json"
    }

    prompt = ''
    chat_round = 1
    for ix, dialogue in enumerate(history):
        if ix % 2 == 0:
            prompt += f'##第 {chat_round} 轮##\n问：{dialogue}\n\n'
        else:
            prompt += f'答：{dialogue}\n\n'
            chat_round += 1

    prompt += f'##第 {chat_round} 轮##\n问：{query}\n\n答：'
    args = {
        "top_p": 0.7,
        "temperature": 0.5,
        "prompt": prompt,
        "seed": 42,
    }
    response = requests.request("POST", MODEL_REQUEST_URL, headers=headers, json=args)

    ans = []
    for line in response.iter_lines():
        ans.append(line.decode("utf-8"))
    finish_idx, meta_idx = -1, -1
    for i in range(len(ans) - 1, -1, -1):
        if ans[i] == "event: finish":
            finish_idx = i
            break
    for i in range(finish_idx + 1, len(ans)):
        if ans[i].startswith("meta: {\"model_version\":\"qa-glm-v"):
            meta_idx = i
    res = []
    for i in range(finish_idx + 2, meta_idx):
        res.append(ans[i][6:])
    return "\n".join(res)


class PublicChatGLMPolicy(BasePolicy):
    prompt: str = GPTPolicy.prompt

    def __init__(self):
        super().__init__()
        self.api_key = "7e486765822f4100bf5feee78a6a37d0"
        self.public_key = "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBALOpXL/AzqJkVdmg8rVtlUWSAKvcy/yleZ+YVXFbxZZ/uMOXBg3e0" \
                          "+AZtPmBHqZjTKHLMuC9SHG3HwIkJxEsXC0CAwEAAQ=="
        print("using public glm model")
        cheat = '\nA valid example should be like:\n' \
                '{"thought": "I want to search for shoes", "action": "search[shoes]"}\nor:\n' \
                '{"thought": "I want to check detailed information", "action": "click[detail]"}\n'
        self.history = [self.prompt + cheat, ""]

    def forward(self, observation, available_actions):
        from wudao.api_request import executeEngine, getToken

        observation = observation.replace(" ", " ")
        observation = observation.replace("[SEP]", "||")
        invalids = []
        for c in observation:
            if ord(c) > 127:
                invalids.append(c)
        for c in invalids:
            print("REPLACING", c)
            observation = observation.replace(c, "*")

        # 接口 API KEY
        API_KEY = self.api_key
        # 公钥
        PUBLIC_KEY = self.public_key

        # 能力类型
        ability_type = "chatGLM"
        # 引擎类型
        engine_type = "chatGLM"

        self.history.append(f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}")

        # 请求参数样例
        data = {
            "top_p": 0.7,
            "temperature": 0.9,
            "prompt": self.history[-1],
            "requestTaskNo": str(random.randint(0, 1000000000)),
            "history": self.history[:-1]
        }
        print("History length:", len(data["history"]))
        try:
            token_result = getToken(API_KEY, PUBLIC_KEY)
            if token_result and token_result["code"] == 200:
                token = token_result["data"]
                resp = executeEngine(ability_type, engine_type, token, data)
                assert resp["code"] == 200
                print(resp)
                print("resp |", resp["data"]["outputText"])
                self.history.append(resp["data"]["outputText"])
                result = json.loads(resp["data"]["outputText"])["action"]
                return result
            else:
                raise Exception("Invalid API_KEY or PUBLIC_KEY")
        except Exception as e:
            print(e)
            return None


class SmallModelPolicy(BasePolicy):
    prompt: str = GPTPolicy.prompt
    cheat = '\nA valid example should be like:\n' \
            'All of your comments or thoughts should be in JSON, do not comment outside of JSON.\n' \
            'Your response must be valid JSON format.\n' \
            '{"thought": "I want to search for shoes", "action": "search[shoes]"}\nor:\n' \
            '{"thought": "I want to check detailed information", "action": "click[detail]"}\n'

    def __init__(self, model="chatglm-6b_v1.1"):
        super().__init__()
        self.url = f"http://166.111.5.162:39999/api/v1/{model}/call"
        self.key = "key-kT6PrwKetv2Xsd9Y"
        self.headers = {"Authorization": self.key}
        self.history = [{"role": "user", "content": self.prompt + self.cheat}]
        requests.post(f"http://166.111.5.162:39999/api/v1/{model}/activate", headers=self.headers, json={})
        print("using", model)

    def forward(self, observation, available_actions):
        try:
            self.history.append({"role": "user", "content": f"Observation:\n{observation}\n\n"
                                                            f"Available Actions:\n{available_actions}"})
            resp = requests.post(self.url, headers=self.headers, json={
                "messages": self.history
            })
            resp = resp.json()
            content = resp["result"]
            print(content)
            self.history.append({"role": "assistant", "content": content})
            result = json.loads(content)["action"]
            return result
        except Exception as e:
            print(e)
            return None


class ClaudePolicy(BasePolicy):
    prompt: str = GPTPolicy.prompt

    def __init__(self, model="claude-v1.3"):
        super().__init__()
        self.url = "http://40.74.217.35:10007/api/claude/completion"
        self.key = "key-kT6PrwKetv2Xsd9Y"
        self.headers = {"Authorization": self.key}
        self.history = [{"role": "user", "content": self.prompt}]
        print("using", model)

    def forward(self, observation, available_actions):
        try:
            self.history.append({"role": "user", "content": f"Observation:\n{observation}\n\n"
                                                            f"Available Actions:\n{available_actions}"})
            resp = requests.post(self.url, headers=self.headers, json={
                "messages": self.history
            })
            resp = resp.json()
            content = resp["completion"]
            print(content)
            self.history.append({"role": "assistant", "content": content})
            result = json.loads(content)["action"]
            return result
        except Exception as e:
            print(e)
            return None


class PalmPolicy(BasePolicy):
    prompt: str = GPTPolicy.prompt

    def __init__(self, model="models/text-bison-001"):
        super().__init__()
        self.url = "http://40.74.217.35:10009/completion"
        self.history = "Q:\n" + self.prompt + "\n"
        self.model = model
        print("using:", model)

    def forward(self, observation, available_actions):
        try:
            self.history += "\nQ:\n" + f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}\n\nA:"
            resp = requests.post(self.url, json={
                "model": self.model,
                "prompt": self.history
            })
            content = resp.json()
            print(content)
            self.history += content + "\n"
            result = json.loads(content)["action"]
            return result
        except Exception as e:
            print(e)
            return None


class NewGLMPolicy(BasePolicy):
    def __init__(self):
        self.url = "http://36.103.203.43:9989/api/call"
        self.history = [GPTPolicy.prompt, "ok"]
        print("using new glm")

    def forward(self, observation, available_actions):
        self.history.append(f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}")
        headers = {"Authorization": "TEMPORARY_AUTHORIZATION_QWDOIAOAHOI2471"}

        response = requests.request("POST", self.url, headers=headers, json={
            "prompt": self.history[-1],
            "history": self.history[:-1]
        })
        try:
            ans = re.search("event: finish\n.+?\ndata: (.+?)\n", response.text).group(1)
        except Exception as e:
            ans = ""
            print(response.text)
            print(e)
        self.history.append(ans)
        try:
            print(ans)
            return json.loads(ans)["action"]
        except Exception as e:
            print(e)
            return None


from fastchat.model.model_adapter import get_conversation_template


class FastChatAgent(BasePolicy):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(self) -> None:
        self.controller_address = "http://166.111.5.162:49990"
        self.worker_address = None
        self.model_name = "vicuna-33b-v1.3"
        self.temperature = 0.7
        self.max_new_tokens = 256
        self.history = [{"role": "user", "content": GPTPolicy.prompt},
                        {"role": "agent", "content": "ok"}
                        ]

    def forward(self, observation, available_actions) -> str:
        self.history.append({"role": "user", "content": f"Observation:\n{observation}\n\n"
                                                        f"Available Actions:\n{available_actions}"})
        if self.worker_address:
            worker_addr = self.worker_address
        else:
            controller_addr = self.controller_address
            ret = requests.post(controller_addr + "/refresh_all_workers")
            ret = requests.post(controller_addr + "/list_models")
            # print(ret.json())
            models = ret.json()["models"]
            models.sort()
            # print(f"Models: {models}")
            ret = requests.post(
                controller_addr + "/get_worker_address", json={"model": self.model_name}
            )
            worker_addr = ret.json()["address"]
            # print(f"worker_addr: {worker_addr}")
        if worker_addr == "":
            # print(f"No available workers for {self.model_name}")
            return
        conv = get_conversation_template(self.model_name)
        for history_item in self.history:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "agent":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        headers = {"User-Agent": "FastChat Client"}
        gen_params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        response = requests.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=gen_params,
            stream=True,
        )
        ans = response.json()["text"]
        self.history.append({"role": "agent", "content": ans})
        try:
            print(ans)
            return json.loads(ans)["action"]
        except Exception as e:
            print(e)
            return None


if __name__ == '__main__':
    p = SmallModelPolicy()
    ans = p.forward("test", "test")
    print(ans)
