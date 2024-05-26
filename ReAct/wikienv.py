import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup

# 定义一个辅助函数来清理字符串
def clean_str(p):
    # 这个函数将字符串进行编码和解码，以确保它能够正确显示
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

class textSpace(gym.spaces.Space):
    # 定义一个简单的文本空间类，继承自gym的Space类
    def contains(self, x) -> bool:
        # 这个方法检查给定的x是否是有效的字符串类型
        return isinstance(x, str)

class WikiEnv(gym.Env):
    # WikiEnv类是gym库中Env类的一个子类，用于模拟与Wikipedia的交互

    def __init__(self):
        # 构造函数，初始化环境的各个组件
        super().__init__()
        self.page = None  # 当前的Wikipedia页面
        self.obs = None  # 当前的观察（可能是页面内容或错误信息）
        self.lookup_keyword = None  # 当前要查找的关键词
        self.lookup_list = None  # 包含当前关键词的所有段落列表
        self.lookup_cnt = None  # 当前查找的索引
        self.steps = 0  # 当前步骤数
        self.answer = None  # 代理当前给出的答案
        self.observation_space = self.action_space = textSpace()  # 定义观察和动作空间
        self.search_time = 0  # 搜索总时间
        self.num_searches = 0  # 搜索次数

    def _get_obs(self):
        # 辅助函数，返回当前的观察值
        return self.obs

    def _get_info(self):
        # 辅助函数，返回当前的步骤和答案信息
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
        # 重置环境到初始状态
        # 如果提供seed，可以用于随机数生成器的种子
        # return_info决定是否返回额外的信息，如步骤数和答案
        # options可以提供额外的重置选项
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        # 重置环境状态
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        # 返回观察和信息
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
        # 根据给定的关键词构建一个包含该关键词的所有段落的列表
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    @staticmethod
    def get_page_obs(page):
        # 从给定的页面内容中提取观察值
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        # 返回前五个句子的文本
        return ' '.join(sentences[:5])

    def search_step(self, entity):
        # 执行搜索步骤，尝试找到给定实体的Wikipedia页面
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        old_time = time.time()
        response_text = requests.get(search_url).text
        self.search_time += time.time() - old_time
        self.num_searches += 1
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # 如果搜索结果中有标题，说明找到了相关页面
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action):
        # 执行给定的动作，动作可以是搜索、查找或完成回答
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:  # 如果已经完成回答，则结束
            done = True
            return self.obs, reward, done, self._get_info()

        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            self.search_step(entity)
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:  # 如果关键词改变，重置查找
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = "Invalid action: {}".format(action)

        self.steps += 1
        return self.obs, reward, done, self._get_info()

    def get_time_info(self):
        # 返回搜索速度信息
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }