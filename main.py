import time

import numpy as np
import json
from typing import Union, Optional, Callable, Literal
from config import Config, no_grad, test_mode
import optimizers
from variable import Variable
from optimizers import SGD
from models import MLP, Model, Sequential
from layers import Linear, Parameter, Layer, Embedding
from functions import mean_squared_error, sigmoid, matmul, Function, cat, softmax, dropout, stack
from datasets import Dataset
from dataloaders import DataLoader
from dataclasses import dataclass

# 中文故事 https://github.com/chenyangMl/llama2.c-zh
# 中文医疗 https://huggingface.co/datasets/shibing624/medical


import os


# 仅设置一块可见
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Tokenizer:
    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.vocab = model["tokens"]
        self.scores = model["scores"]
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.n_words = len(self.vocab)
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def str_lookup(self, token: str) -> int:
        try:
            index = self.vocab.index(token)
            return index
        except ValueError as err:
            return -1

    def encode(
            self,
            text: str,
            add_bos: bool = True,
            add_eos: bool = False,
            add_prefix: bool = True,
            add_new_bos: bool = False,
    ) -> list[int]:
        tokens = []
        for pos, char in enumerate(text):
            id = self.str_lookup(char)
            if id >= 0:
                tokens.append(id)
            else:
                tokens = tokens + [(i + 3) for i in char.encode()]
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                # Check if we can merge the pair (tokens[i], tokens[i+1])
                string = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                id = self.str_lookup(string)
                if id != -1 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                break

            # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id
            # Delete token at position best_idx+1, shift the entire sequence back 1
            tokens = tokens[0: best_idx + 1] + tokens[best_idx + 2:]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        if add_prefix:
            tokens.insert(0, self.special_tokens['sop'])
            tokens.insert(0, self.special_tokens['[gMASK]'])
        if add_new_bos:
            tokens.append(self.bos_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        res = []
        for i in ids:
            token = self.vocab[i]
            res.append(token)
        text = "".join(res)
        text = text.strip("<s>").strip("</s>")
        return text


class SelfAttention(Model):
    def __init__(self,
                 args: 'LLaMaArgs',
                 rope_apply: Callable):
        super(SelfAttention, self).__init__()

        assert args.num_heads * args.head_dim == args.hidden_size
        assert args.num_heads % args.num_key_value_heads == 0
        assert args.head_dim % 2 == 0

        self.max_len = args.max_len
        self.max_batch_size = args.max_batch_size
        self.enable_kv_cache = args.enable_kv_cache
        self.use_gpu = args.use_gpu

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.attention_bias = args.attention_bias
        self.dropout_ratio = args.dropout_ratio

        self.dropout_on = args.dropout_ratio != 0
        self.kv_repeat_num = self.num_heads // self.num_key_value_heads

        self.rope_apply = rope_apply

        self.q_proj = Linear(in_size=self.hidden_size, out_size=self.num_heads * self.head_dim,
                             nobias=~self.attention_bias)

        self.k_proj = Linear(in_size=self.hidden_size, out_size=self.num_key_value_heads * self.head_dim,
                             nobias=~self.attention_bias)

        self.v_proj = Linear(in_size=self.hidden_size, out_size=self.num_key_value_heads * self.head_dim,
                             nobias=~self.attention_bias)

        self.o_proj = Linear(in_size=self.hidden_size, out_size=self.hidden_size, nobias=~self.attention_bias)

        if self.enable_kv_cache:
            self.k_cache = Variable(np.zeros([self.max_batch_size, self.num_key_value_heads, 0, self.head_dim]))
            self.v_cache = Variable(np.zeros([self.max_batch_size, self.num_key_value_heads, 0, self.head_dim]))
            if self.use_gpu:
                self.k_cache.to_gpu()
                self.v_cache.to_gpu()

    def forward(self, x, cos_pos, sin_pos):
        batch_size = x.shape[0]
        length = x.shape[1]
        # embed_dim = x.shape[2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [batch_size, length, hidden_size]

        q = q.reshape(batch_size, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, length, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, length, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        # [batch_size, length, num_heads, head_dim]
        # [batch_size, num_heads, length, head_dim]

        # q,k rope finish
        # q = apply_RoPE(q, cos_pos, sin_pos)
        # k = apply_RoPE(k, cos_pos, sin_pos)
        q = self.rope_apply(q, cos_pos, sin_pos)
        k = self.rope_apply(k, cos_pos, sin_pos)

        if self.enable_kv_cache:
            start_pos = self.k_cache.shape[2]
        else:
            start_pos = 0

        if self.enable_kv_cache:
            self.k_cache = cat((self.k_cache, k), axis=2)
            self.v_cache = cat((self.v_cache, v), axis=2)
            k = self.k_cache
            v = self.v_cache

        # print(k[0, 0])
        # print(v[0, 0])

        # 相乘之前若是kv头数不一样还需要重复 num_heads % num_key_value_heads
        if self.num_heads != self.num_key_value_heads:
            k = k[:, np.arange(self.num_key_value_heads).repeat(self.kv_repeat_num), :, :]
            v = v[:, np.arange(self.num_key_value_heads).repeat(self.kv_repeat_num), :, :]

        attention_weight = matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        mask = np.full((length, length), -np.inf)
        mask = np.triu(mask, k=1)
        mask = np.concatenate((np.zeros((length, start_pos)), mask), axis=1)

        if self.use_gpu:
            from cuda import as_cupy
            mask = as_cupy(mask)

        attention_weight = attention_weight + mask

        attention_weight = softmax(attention_weight, axis=-1)

        if self.dropout_on:
            attention_weight = dropout(attention_weight, self.dropout_ratio)

        output = matmul(attention_weight, v)  # (bzs, num_heads, length, head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, length, self.hidden_size)
        # (bzs, length, embed_dim)
        output = self.o_proj(output)

        return output


class SiLU(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        self.sigmoid = 1 / (1 + np.exp(-x))
        y = x * self.sigmoid
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        y = self.outputs[0]()
        gx = gy * (y + self.sigmoid * (1 - y))
        # y'=(xs)'=s+xs(1-s)=s+xs-xss=xs+s(1-xs)=y+s(1-y)
        return gx


def silu(x):
    return SiLU()(x)


class SwiGLUFeedForwardNetwork(Model):
    def __init__(self, hidden_size: int, intermediate_size: int, use_bias: bool = False):
        super(SwiGLUFeedForwardNetwork, self).__init__()
        self.fc_gate = Linear(in_size=hidden_size, out_size=intermediate_size, nobias=~use_bias)
        self.fc_up = Linear(in_size=hidden_size, out_size=intermediate_size, nobias=~use_bias)
        self.fc_down = Linear(in_size=intermediate_size, out_size=hidden_size, nobias=~use_bias)

    def forward(self, x):
        x1 = self.fc_up(x)
        x = silu(self.fc_gate(x))
        x = x * x1
        x = self.fc_down(x)
        return x


# class RMSNorm(Layer):
#     def __init__(self, hidden_size: int, eps: float = 1e-6):
#         super(RMSNorm, self).__init__()
#         self.weight = Parameter(np.ones(hidden_size), 'weight')
#         self.epsilon = eps
#
#     def forward(self, x):
#         x_shape = x.shape
#         x = x * ((x ** 2).sum(axis=x.ndim - 1) / x_shape[-1] + self.epsilon).reshape(*(x_shape[:-1] + (1,))) ** (-1 / 2)
#         x = self.weight * x
#         return x

class RoPELlama:
    def __init__(self,
                 max_len: int,
                 output_dim: int,
                 rope_theta: float = 10000.0):
        self.max_len = max_len
        self.output_dim = output_dim
        self.rope_theta = rope_theta

        def apply(q: Variable, cos_pos: np.ndarray, sin_pos: np.ndarray):
            q2 = stack((-q[..., 1::2], q[..., ::2]), axis=-1)
            q2 = q2.reshape(q.shape)
            q = q * cos_pos + q2 * sin_pos
            return q

        self.apply = apply

    def get_cos_sin(self):
        position = np.arange(0, self.max_len, dtype=np.float32)[..., np.newaxis]
        ids = np.arange(0, self.output_dim // 2, dtype=np.float32)
        theta = self.rope_theta ** (-2 * ids / self.output_dim)
        embeddings = position * theta
        # (max_len, output_dim//2, 2)
        embeddings = np.stack([np.sin(embeddings), np.cos(embeddings)], axis=-1)
        # (bs, head, max_len, output_dim//2, 2)
        embeddings = np.tile(embeddings,
                             (1, 1, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复
        # (bs, head, max_len, output_dim)
        # reshape后就是：偶数sin, 奇数cos了
        embeddings = np.reshape(embeddings, (1, 1, self.max_len, self.output_dim))
        cos_pos = embeddings[..., 1::2].repeat(2, axis=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
        sin_pos = embeddings[..., ::2].repeat(2, axis=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制
        return cos_pos, sin_pos


class RoPEHF:
    def __init__(self,
                 max_len: int,
                 output_dim: int,
                 rope_theta: float = 500000.0):
        self.max_len = max_len
        self.output_dim = output_dim
        self.rope_theta = rope_theta

        def apply(q: Variable, cos_pos: np.ndarray, sin_pos: np.ndarray):
            q2 = cat((-q[..., q.shape[-1] // 2:], q[..., : q.shape[-1] // 2]), axis=-1)
            q = q * cos_pos + q2 * sin_pos
            return q

        self.apply = apply

    def get_cos_sin(self):
        # HF
        position = np.arange(0, self.max_len, dtype=np.float32)[..., np.newaxis]
        ids = np.arange(0, self.output_dim // 2, dtype=np.float32)
        theta = self.rope_theta ** (-2 * ids / self.output_dim)
        embeddings = position * theta
        embeddings = np.concatenate((embeddings, embeddings), axis=-1)[np.newaxis, np.newaxis, :, :]
        cos_pos = np.cos(embeddings)
        sin_pos = np.sin(embeddings)
        return cos_pos, sin_pos


# def sinusoidal_position_embedding(batch_size: int,
#                                   nums_head: int,
#                                   max_len: int,
#                                   output_dim: int,
#                                   rope_theta: float = 10000.0):
#     # (max_len, 1)
#     position = np.arange(0, max_len, dtype=np.float32)[..., np.newaxis]
#     # (output_dim//2)
#     ids = np.arange(0, output_dim // 2, dtype=np.float32)  # 即公式里的i, i的范围是 [0,d/2]
#     theta = rope_theta ** (-2 * ids / output_dim)
#
#     # (max_len, output_dim//2)
#     embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))
#
#     # (max_len, output_dim//2, 2)
#     embeddings = np.stack([np.sin(embeddings), np.cos(embeddings)], axis=-1)
#
#     # (bs, head, max_len, output_dim//2, 2)
#
#     embeddings = np.tile(embeddings, (batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复
#
#     # (bs, head, max_len, output_dim)
#     # reshape后就是：偶数sin, 奇数cos了
#
#     embeddings = np.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
#     return embeddings
#
#
# def apply_RoPE(q: Variable, cos_pos: np.ndarray, sin_pos: np.ndarray):
#     q2 = stack((-q[..., 1::2], q[..., ::2]), axis=-1)
#     q2 = q2.reshape(q.shape)  # reshape后就是正负交替了
#     q = q * cos_pos + q2 * sin_pos
#     return q


class TransformerDecoderBlock(Model):
    def __init__(self,
                 args: 'LLaMaArgs',
                 rope_apply: Callable):
        super(TransformerDecoderBlock, self).__init__()

        self.max_len = args.max_len
        self.max_batch_size = args.max_batch_size
        self.enable_kv_cache = args.enable_kv_cache

        self.hidden_size = args.hidden_size

        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.attention_bias = args.attention_bias
        self.dropout_ratio = args.dropout_ratio

        self.ffn_intermediate_size = args.ffn_intermediate_size
        self.ffn_bias = args.ffn_bias

        self.rms_eps = args.rms_eps

        self.multi_head_self_attention = SelfAttention(args=args, rope_apply=rope_apply)
        self.ffn = SwiGLUFeedForwardNetwork(hidden_size=self.hidden_size, intermediate_size=self.ffn_intermediate_size,
                                            use_bias=self.ffn_bias)

        self.rms_norm_1 = RMSNorm(hidden_size=self.hidden_size, eps=self.rms_eps)
        self.rms_norm_2 = RMSNorm(hidden_size=self.hidden_size, eps=self.rms_eps)

    def forward(self, x, cos_pos, sin_pos):
        x = self.multi_head_self_attention(self.rms_norm_1(x), cos_pos, sin_pos) + x
        x = self.ffn(self.rms_norm_2(x)) + x
        return x


class LLaMa(Model):
    def __init__(self,
                 args: 'LLaMaArgs'):
        super(LLaMa, self).__init__()

        self.max_len = args.max_len
        self.max_batch_size = args.max_batch_size
        self.enable_kv_cache = args.enable_kv_cache
        self.use_gpu = args.use_gpu

        self.vocab_size = args.vocab_size
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size

        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.num_key_value_heads = args.num_key_value_heads
        self.attention_bias = args.attention_bias
        self.rope_theta = args.rope_theta
        self.dropout_ratio = args.dropout_ratio

        self.ffn_intermediate_size = args.ffn_intermediate_size
        self.ffn_bias = args.ffn_bias

        self.rms_eps = args.rms_eps

        self.embedding = Embedding(in_size=self.vocab_size, out_size=self.hidden_size)

        self.rope_type = args.rope_type
        if self.rope_type == 'Llama':
            self.rope = RoPELlama(max_len=self.max_len,
                                  output_dim=self.head_dim,
                                  rope_theta=self.rope_theta)
        else:
            self.rope = RoPEHF(max_len=self.max_len,
                               output_dim=self.head_dim,
                               rope_theta=self.rope_theta)

        self.transformers = Sequential(*[
            TransformerDecoderBlock(args=args, rope_apply=self.rope.apply) for _ in range(self.num_layers)])

        self.last_rms = RMSNorm(hidden_size=self.hidden_size, eps=self.rms_eps)
        self.linear = Linear(in_size=self.hidden_size, out_size=self.vocab_size, nobias=True)

        self.weight_share = args.weight_share

        if self.weight_share:
            self.linear.W = self.embedding.W.T

        self.cos_pos, self.sin_pos = self.rope.get_cos_sin()

        if self.use_gpu:
            from cuda import as_cupy
            self.cos_pos = as_cupy(self.cos_pos)
            self.sin_pos = as_cupy(self.sin_pos)

    def forward(self, x):
        if self.enable_kv_cache:
            start_pos = self.transformers.layers[0].multi_head_self_attention.k_cache.shape[2]
        else:
            start_pos = 0
        now_len = x.shape[1]
        if start_pos + now_len >= self.max_len:
            raise 'kv cache is full'
        x = self.embedding(x)
        for layer in self.transformers.layers:
            x = layer(x, self.cos_pos[:, :, start_pos:(start_pos + now_len), :],
                      self.sin_pos[:, :, start_pos:(start_pos + now_len), :])
        x = self.last_rms(x)
        x = self.linear(x[:, -1, :])
        # return softmax(x, 2)
        return x

    def clean_kv_cache(self):
        if self.enable_kv_cache:
            for i in self.transformers.layers:
                if self.use_gpu:
                    import cupy as cp
                    i.multi_head_self_attention.k_cache = Variable(
                        cp.zeros((self.max_batch_size, self.num_key_value_heads, 0, self.head_dim)))
                    i.multi_head_self_attention.v_cache = Variable(
                        cp.zeros((self.max_batch_size, self.num_key_value_heads, 0, self.head_dim)))
                else:
                    i.multi_head_self_attention.k_cache = Variable(
                        np.zeros([self.max_batch_size, self.num_key_value_heads, 0, self.head_dim]))
                    i.multi_head_self_attention.v_cache = Variable(
                        np.zeros([self.max_batch_size, self.num_key_value_heads, 0, self.head_dim]))
            print('kv cache cleaned')
        else:
            print('kv cache is not enabled')

    def generate(self, token: np.ndarray, max_gen: int, temperature: float, top_k: int, eos_id: int = 2):
        token_batch, token_len = token.shape
        assert token_batch == 1
        if token_len > self.max_len:
            token = token[:, (token_len - self.max_len):]
            token_len = self.max_len

        new_char = 0
        for i in range(max_gen):
            if self.enable_kv_cache:
                if i == 0:
                    r = self(token)
                else:
                    r = self(np.array([[new_char]]))
            else:
                r = self(token)
            r.to_cpu()
            if temperature == 0:
                new_char = np.argmax(r.data)
                new_char = int(new_char)
            else:
                new_r = r.data / temperature
                r_top_k = np.argsort(-new_r)[:, top_k]
                new_r[new_r < new_r[:, r_top_k]] = -np.inf
                probs = softmax(new_r).data.astype(np.float64)
                probs = probs / probs.sum()
                new_char = np.argmax(np.random.multinomial(n=1, pvals=probs[0]))
                new_char = int(new_char)

            token = np.concatenate((token, np.array([[new_char]])), axis=1)
            # print(tokenizer.decode([new_char]), end='')
            yield new_char
            if new_char == eos_id:
                break
        return token

    def chat(self, promote: str, tokenizer: Tokenizer, max_gen: int = 500, temperature: float = 1.0, top_k: int = 100,
             bos_id: int = 2):
        tokens = tokenizer.encode(promote, add_eos=False, add_new_bos=True, add_bos=False, add_prefix=False)
        # tokens = tokenizer.encode(promote, add_eos=False, add_new_bos=False, add_bos=True, add_prefix=False)
        tokens = np.array(tokens)[np.newaxis, ...]
        gen = ''
        for i in self.generate(tokens, max_gen, temperature, top_k, bos_id):
            if i == tokenizer.eos_id:
                print('<eos>')
            new_char = tokenizer.decode([i])
            gen += new_char
            print(new_char, end='')
        return gen


class RMSNormFunction(Function):
    def __init__(self, eps: float = 1e-6):
        self.epsilon = eps

    def forward(self, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray]:
        self.rms_inv = ((x ** 2).sum(axis=x.ndim - 1, keepdims=True) / x.shape[-1] + self.epsilon) ** (-1 / 2)
        self.rms_x = x * self.rms_inv
        y = self.rms_x * w
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        x, w = self.inputs
        gw = (gy * self.rms_x).sum(axis=tuple([i for i in range(x.ndim - 1)]))
        gx = gy * w * self.rms_inv - x * (self.rms_inv ** 3) * (
                (gy * w * x).sum(axis=x.ndim - 1, keepdims=True) / x.shape[-1])
        return gx, gw


def rms_norm(x, w, eps=1e-6):
    return RMSNormFunction(eps=eps)(x, w)


class RMSNorm(Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.weight = Parameter(np.ones(hidden_size), 'weight')
        self.epsilon = eps

    def forward(self, x):
        return rms_norm(x, self.weight, eps=self.epsilon)


@dataclass
class LLaMaArgs:
    vocab_size: int = 64783
    num_layers: int = 12
    hidden_size: int = 1024
    num_heads: int = 8
    head_dim: int = 128
    num_key_value_heads: int = 8
    attention_bias: bool = False
    weight_share: bool = True
    rope_type: Literal['Llama', 'HF'] = 'Llama'
    rope_theta: float = 10000.0
    enable_kv_cache: bool = True
    ffn_intermediate_size: int = 2752
    ffn_bias: bool = False
    max_len: int = 1024
    rms_eps: float = 1e-5
    dropout_ratio: float = 0.0
    max_batch_size: int = 1
    use_gpu: bool = True


baby_llama_zh = LLaMaArgs(
    vocab_size=64783,
    num_layers=12,
    hidden_size=1024,
    num_heads=8,
    head_dim=128,
    num_key_value_heads=8,
    attention_bias=False,
    weight_share=True,
    rope_type='Llama',
    rope_theta=10000.0,
    enable_kv_cache=True,
    ffn_intermediate_size=2752,
    ffn_bias=False,
    max_len=1024,
    rms_eps=1e-5,
    dropout_ratio=0.0,
    max_batch_size=1,
    use_gpu=True,
)

atom_7b = LLaMaArgs(
    vocab_size=65000,
    num_layers=32,
    hidden_size=4096,
    num_heads=32,
    head_dim=128,
    num_key_value_heads=32,
    attention_bias=False,
    weight_share=False,
    rope_type='HF',
    rope_theta=500000.0,
    enable_kv_cache=True,
    ffn_intermediate_size=11008,
    ffn_bias=False,
    max_len=4096,
    rms_eps=1e-5,
    dropout_ratio=0.0,
    max_batch_size=1,
    use_gpu=True,
)


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.s = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.e = time.time()
        print(f'{self.name} cost {self.e - self.s} seconds')


if __name__ == '__main__':
    np.random.seed(114514)

    model_dict_atom_7b = {
        'args': atom_7b,
        'weights_path': 'Atom7b.npz',
        'tokenizer_path': 'tokenizer_atom7b.model.np',
    }
    model_dict_baby_llama_zh = {
        'args': baby_llama_zh,
        'weights_path': 'WEIGHTS.npz',
        'tokenizer_path': 'tokenizer_chatglm2.model.np',
    }

    model_dict = model_dict_baby_llama_zh

    with no_grad(), test_mode():
        tokenizer = Tokenizer(model_path=model_dict['tokenizer_path'])

        with Timer('model init'):  # 771
            m = LLaMa(args=model_dict['args'])

        with Timer('weights load'):  # 235
            m.load_weights(model_dict['weights_path'])

        if model_dict['args'].use_gpu:
            with Timer('to gpu'):  # 6
                m.to_gpu()

        # i = np.array([[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]])
        # print(m(i))

        # m1 = LLaMa(args=model_args)
        # m1.load_weights('WEIGHTS.npz')
        # m1.to_gpu()
        # m2 = LLaMa(args=model_args)
        # m2.load_weights('WEIGHTS.npz')
        # m2.to_gpu()
        # the_str = '写一篇700字以上的有关大语言模型的议论文'
        # print(the_str)
        # for i in range(100):
        #     r1 = m1.chat(the_str, tokenizer)
        #     r2 = m1.chat(r1, tokenizer)
        #     the_str = r2

        # https://github.com/AI-Study-Han/Mini-Llama2-Chinese/tree/main

        # the_str = '什么是大语言模型'
        # print(the_str)
        # m.chat(the_str, tokenizer)

        for _ in range(100):
            test_str = input()
            if test_str == '\\clean':
                m.clean_kv_cache()
                continue
            if test_str == '\\stop':
                break
            m.chat(test_str, tokenizer)
