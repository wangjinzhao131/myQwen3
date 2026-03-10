"""
分词器实现 - 从零实现
包含 BPE (Byte Pair Encoding) 和简单字符级分词器
"""

import json
import re
from collections import defaultdict


class BPETokenizer:
    """
    BPE (Byte Pair Encoding) 分词器

    BPE 算法原理：
        1. 初始化：每个字符是一个 token
        2. 迭代：找到最频繁的相邻 token 对，合并为新 token
        3. 重复直到达到目标词表大小

    举例：
        语料: "aa ab"
        初始: ['a', 'a', 'a', 'b']  # 每个字符一个 token

        第1轮: 'a' + 'a' 最频繁（出现2次）
        合并后: ['aa', 'a', 'b']  # 新增 token 'aa'

        第2轮: 'aa' + 'a' 最频繁
        合并后: ['aaa', 'b']  # 新增 token 'aaa'

    优点：
        - 自动学习常见子词
        - 处理未登录词（OOV）能力强
        - 词表大小可控
    """

    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size  # 目标词表大小
        self.vocab = {}  # 词表
        self.merges = []  # 合并规则列表（按顺序）
        self.token_to_id = {}  # token -> ID 映射
        self.id_to_token = {}  # ID -> token 映射

    def get_stats(self, word_freqs):
        """
        统计相邻 token 对的频率

        参数：
            word_freqs: 字典 {单词: 频率}
                       单词格式: "h e l l o </w>"（空格分隔的字符）

        返回：
            pairs: 字典 {(token1, token2): 频率}

        举例：
            word_freqs = {"h e l l o </w>": 2, "w o r l d </w>": 1}

            统计结果:
            ('h', 'e'): 2
            ('e', 'l'): 2
            ('l', 'l'): 2
            ('l', 'o'): 2
            ('o', '</w>'): 2
            ('w', 'o'): 1
            ...
        """
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            # 按空格分割得到 token 列表
            symbols = word.split()
            # 统计相邻 token 对
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, word_freqs, pair):
        """
        合并指定的 token 对

        参数：
            word_freqs: 字典 {单词: 频率}
            pair: 要合并的 token 对，如 ('l', 'l')

        返回：
            new_word_freqs: 合并后的字典

        举例：
            pair = ('l', 'l')
            "h e l l o </w>" -> "h e ll o </w>"
        """
        new_word_freqs = {}
        bigram = " ".join(pair)  # "l l"
        replacement = "".join(pair)  # "ll"

        for word, freq in word_freqs.items():
            # 将 "l l" 替换为 "ll"
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus, num_merges=None):
        """
        训练 BPE 分词器

        参数：
            corpus: 训练语料（字符串）
            num_merges: 合并次数，默认为 vocab_size - 256

        训练过程：
            1. 预处理：将语料分割为单词
            2. 初始化：每个字符是一个 token，加上 </w> 标记词尾
            3. 迭代合并：
               a. 统计相邻 token 对频率
               b. 找到最频繁的 pair
               c. 合并该 pair
               d. 记录合并规则
            4. 建立词表
        """
        if num_merges is None:
            num_merges = self.vocab_size - 256

        # 步骤1: 分词
        # \w+ 匹配单词字符（字母、数字、下划线）
        # [^\w\s] 匹配非单词非空白字符（标点等）
        words = re.findall(r"\w+|[^\w\s]", corpus.lower())

        # 步骤2: 统计词频，并初始化为字符序列
        # 格式: "h e l l o </w>"
        # </w> 表示词尾，用于区分词边界
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[" ".join(list(word)) + " </w>"] += 1

        # 步骤3: 初始化词表（ASCII 字符）
        # 前 256 个 ID 分配给 ASCII 字符
        for i in range(256):
            self.token_to_id[chr(i)] = i
            self.id_to_token[i] = chr(i)

        next_id = 256  # 新 token 从 ID 256 开始

        # 步骤4: 迭代合并
        for _ in range(num_merges):
            # 统计相邻 token 对频率
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break

            # 找到最频繁的 pair
            best_pair = max(pairs, key=pairs.get)

            # 合并该 pair
            word_freqs = self.merge_vocab(word_freqs, best_pair)

            # 记录合并规则
            self.merges.append(best_pair)

            # 添加新 token 到词表
            new_token = "".join(best_pair)
            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token
            next_id += 1

            if next_id >= self.vocab_size:
                break

    def tokenize(self, text):
        """
        分词：将文本分割为 token 序列

        参数：
            text: 输入文本

        返回：
            tokens: token 列表

        分词过程：
            1. 将文本分割为单词
            2. 对每个单词：
               a. 初始化为字符序列
               b. 按训练时的合并规则顺序应用合并
               c. 选择优先级最高（rank 最小）的 pair 合并
            3. 返回 token 列表
        """
        text = text.lower()
        words = re.findall(r"\w+|[^\w\s]", text)

        tokens = []
        for word in words:
            # 初始化为字符序列 + </w>
            word_tokens = list(word) + ["</w>"]

            # 迭代应用合并规则
            while len(word_tokens) > 1:
                # 找到所有相邻 pair
                pairs = [
                    (word_tokens[i], word_tokens[i + 1])
                    for i in range(len(word_tokens) - 1)
                ]

                # 建立 pair -> rank 映射
                # rank 越小，优先级越高（越早被合并）
                pair_ranks = {pair: i for i, pair in enumerate(self.merges)}

                # 找到 rank 最小的 pair（优先级最高）
                min_pair = None
                min_rank = float("inf")
                for pair in pairs:
                    if pair in pair_ranks and pair_ranks[pair] < min_rank:
                        min_rank = pair_ranks[pair]
                        min_pair = pair

                # 如果没有可合并的 pair，结束
                if min_pair is None:
                    break

                # 应用合并
                new_word_tokens = []
                i = 0
                while i < len(word_tokens):
                    # 如果当前位置是目标 pair，合并并跳过下一个
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == min_pair
                    ):
                        new_word_tokens.append("".join(min_pair))
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens

            tokens.extend(word_tokens)

        return tokens

    def encode(self, text):
        """
        编码：将文本转换为 token ID 序列

        参数：
            text: 输入文本

        返回：
            ids: token ID 列表
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # 未登录词：拆分为字符
                for char in token:
                    ids.append(self.token_to_id.get(char, 0))
        return ids

    def decode(self, ids):
        """
        解码：将 token ID 序列转换回文本

        参数：
            ids: token ID 列表

        返回：
            text: 解码后的文本
        """
        tokens = [self.id_to_token.get(id, "<unk>") for id in ids]
        text = "".join(tokens)
        # 将 </w> 替换为空格
        text = text.replace("</w>", " ")
        return text.strip()

    def save(self, path):
        """保存分词器到文件"""
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """从文件加载分词器"""
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}


class SimpleTokenizer:
    """
    简单字符级分词器

    原理：每个字符是一个 token

    优点：
        - 实现简单
        - 不会出现未登录词

    缺点：
        - 序列很长（每个字符一个 token）
        - 无法学习子词模式

    用于快速原型和学习
    """

    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.char_to_id = {}  # 字符 -> ID
        self.id_to_char = {}  # ID -> 字符

    def train(self, corpus):
        """
        训练：建立字符词表

        参数：
            corpus: 训练语料
        """
        # 获取所有唯一字符（排序保证一致性）
        chars = sorted(set(corpus.lower()))

        # 建立映射
        for i, char in enumerate(chars):
            self.char_to_id[char] = i
            self.id_to_char[i] = char

        # 添加特殊 token
        self.char_to_id["<pad>"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "<pad>"

        self.char_to_id["<unk>"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "<unk>"

    def encode(self, text):
        """
        编码：文本 -> ID 序列

        参数：
            text: 输入文本

        返回：
            ids: 字符 ID 列表
        """
        return [self.char_to_id.get(c, self.char_to_id["<unk>"]) for c in text.lower()]

    def decode(self, ids):
        """
        解码：ID 序列 -> 文本

        参数：
            ids: 字符 ID 列表

        返回：
            text: 解码后的文本
        """
        return "".join([self.id_to_char.get(i, "<unk>") for i in ids])

    @property
    def vocab_size_actual(self):
        """实际词表大小"""
        return len(self.char_to_id)


if __name__ == "__main__":
    corpus = "Hello world! This is a test corpus for training our tokenizer."

    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(corpus, num_merges=100)

    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"Decoded: {decoded}")
