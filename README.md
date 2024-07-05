# LLaMa3-Numpy-trainable
用Numpy复现可训练的LLaMa3
# 大作业 llama3-numpy

## 简介

### 作业要求

1.在 AI Studio 上做一个公开项目，命名为“山东大学威海数据科学实验班_Llama3_Numpy 复现_小组成员名字”。在该公开项目里根据下面的链接复现 Llama3。需要做到：

（1）用中文将每个步骤详细解说；

（2）在项目最开头注明这个项目是大三暑假大作业的一部分，写明小组成员姓大作业 llama3-numpy​
￼
白锦帆
6 月 12 日创建
简介​
作业要求​
1.在 AI Studio 上做一个公开项目，命名为“山东大学威海数据科学实验班_Llama3_Numpy 复现_小组成员名字”。在该公开项目里根据下面的链接复现 Llama3。需要做到：​
（1）用中文将每个步骤详细解说；​
（2）在项目最开头注明这个项目是大三暑假大作业的一部分，写明小组成员姓名和年级，并明确标出下面的参考来源。

[https://github.com/naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)

如果不能复现 Llama3，可以改成复现 picoGPT:

[https://github.com/jaymody/picoGPT](https://github.com/jaymody/picoGPT)

录制一个视频，讲解复现过程，上传至 B 站（注明 AI Studio 公开项目链接），并在 QQ 实验班公告群发布。截止日期：2024 年 7 月 5 日晚。

### 我们做了什么

- **用 Numpy 复现了 LLaMa 架构的 LLM**
- 可以开启 KV cache
- 并且这个 LLaMa 是**可以训练**的,这是因为我们用 Numpy 搭了一个**深度学习框架**
- 也可以**在 GPU 上跑**的(使用 Cupy)
- 在这个架构上,我们分别实现了一个 0.2b 的 baby llama 模型,和一个 7b 的 Atom7b 中文 LLaMa 模型

## llama3 架构

### Decoder Only Transformer Block

#### 结构

#### 层归一化 RMSnorm

##### 公式表示

$$
\bar{a}_i = \frac{a_i}{\mathbf{RMS}(\mathbf{a})} g_i, where \mathbf{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2}
$$

实际操作中为了防止分母变成零,加入一个$$\epsilon > 0$$,通常取 1e-6:

$$
\mathbf{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}a_i^2 + \epsilon}
$$

##### 原理

与普通的 LayerNorm 相比,RMSnorm 没有了去中心化的操作,可以提升运行效率,同时也表现更好.

##### Transformers 包实现代码

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

#### SwiGLU Feed Forward Network

SwiGLU 即为 Swish Gated Linear Unit 但是在实际实现中通常使用 SiLU 代替 Swish,即为$\beta=1$时.$\beta\to+\infty$时为 ReLU.

$$
\mathbf{Swish}_\beta(x)=x\sigma(\beta x),\mathbf{SiLU}(x)=x\sigma(x)
$$

FC 即为全连接层.

为什么 SwiGLU 有效?对此原论文这样说:

> We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to **divine benevolence**.

#### 多头注意力

##### 原理

###### QKV 的含义

Q,K,V 是 Self Attention 的基础.

- Q(Query):Q 可以被视为寻求信息的元素.对于输入序列中的每个单词,都会计算一个 Q 向量.这些 Q 表示要注意序列中的内容.
- K(Key):K 就像路标.它们有助于识别和定位序列中的重要元素.与 Q 一样,为每个单词计算 K 向量.
- V(Value):V 携带信息.同样,对于每个单词,都会计算一个 V 向量.这些 V 向量包含我们在确定序列中单词的重要性时要考虑的内容.

这三者都是通过简单的 Linear 层实现的.

###### Attention Scores

公式如下:

$$
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}\end{equation}
$$

Attention Scores 是一个矩阵,他的第 i 行,就是根据第 i 个词关于自身和之前词的注意力的大小,按其比例混合 V 后得到的结果.

为了计算 Attention Scores 就必须得到 Attention Weights.

先从最简单的情况开始,假设第 i 个词的 q,k 个向量为$q_i,k_i$,第 j 个词的 q,k 个向量为$q_j,k_j$,那么第 i 个词关于第 j 个词的注意力权重(未 scale)就为$q_{i}k_{j}^{T}$,这也就是两者的点积$<q_{i},k_{j}>$,加上 scale 就是$\frac{q_{i}k_{j}^{T}}{\sqrt{d_k}}$,这个缩放因子$\sqrt{d_k}$是起到调节作用使得内积不至于太大的.

那么为了更优雅的计算 Attention Weights 矩阵,需要使用矩阵乘法:$\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}$,这样第 i 个词关于第 j 个词的注意力权重就在这个矩阵的第 i 行第 j 列.

由于注意力混合只能是某一个词关于自身和之前词的注意力,这是因为模型的目的是预测下一个词.也就是说对于第 i 个词只能看它关于第 j(j<=i)个词的注意力.因此就需要一个上三角形的 Mask,来遮住.这个 Mask 在上三角部分为负无穷,因为只有负无穷在 softmax 下才为 0.

| 
$$
\frac{q_{1}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 
$$
\frac{q_{2}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{2}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{3}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{3}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{3}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{4}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{4}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{5}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{5}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{5}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{5}k_{4}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{5}k_{5}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{6}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{6}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{6}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{6}k_{4}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{6}k_{5}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{6}k_{6}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{7}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{7}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{7}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{7}k_{4}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{7}k_{5}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{7}k_{6}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{7}k_{7}^{T}}{\sqrt{d_k}}
$$

 |

为了根据比例混合 V,我们需要把每一个词的原始的 attention weights 向量转化成一个全为非负元素且和 1 的向量,也就是一个分布,这个转化方式就是 softmax 函数.这样就得到了 Attention Weights 矩阵.

然后在与 V 相乘,就得到了 Attention Scores.

###### 从单头到多头

即是将多个单头结果拼接,然后经过一个 o_proj 的 Linear 层.

##### KV cache

KV cache 是一种"以空间换时间"的通过储存 k,v 加速方法.

为了理解,我们先看三个词的情况.这是此时未经过 softmax 的 Attention Weights:

| 
$$
\frac{q_{1}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| --------------------------------- | --------------------------------- | --------------------------------- |
| 
$$
\frac{q_{2}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{2}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{3}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{3}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{3}k_{3}^{T}}{\sqrt{d_k}}
$$

 |

然后 softmax 后与 V(如下)相乘,之后经过一系列操作得到下一个词.

| 
$$
v_1
$$

 |
| --- |
| 
$$
v_2
$$

 |
| 
$$
v_3
$$

 |

为了下下一个词,我们将原输入(1,2,3)和这次的输出(4)拼接得到(1,2,3,4)再输入模型,这回计算就变成了:

| 
$$
\frac{q_{1}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 
$$
\frac{q_{2}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{2}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{3}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{3}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{3}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
-\infty
$$

                           |
| 
$$
\frac{q_{4}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{4}^{T}}{\sqrt{d_k}}
$$

 |

然后 softmax 后与 V(如下)相乘,之后经过一系列操作得到下一个词.

| 
$$
v_1
$$

 |
| --- |
| 
$$
v_2
$$

 |
| 
$$
v_3
$$

 |
| 
$$
v_4
$$

 |

我们发现,真正影响下一个词的,是最后一个词的 Attention Weights 向量,即在预测下一个词时,有用的是:

| 
$$
\frac{q_{4}k_{1}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{2}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{3}^{T}}{\sqrt{d_k}}
$$

 | 
$$
\frac{q_{4}k_{4}^{T}}{\sqrt{d_k}}
$$

 |
| --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |

它 softmax 后与 V:

| 
$$
v_1
$$

 |
| --- |
| 
$$
v_2
$$

 |
| 
$$
v_3
$$

 |
| 
$$
v_4
$$

 |

可以看到,这样我们已经省去了$q_1,q_2,q_3$的计算.

更进一步,$k_1,k_2,k_3$以及$v_1,v_2,v_3$都是之前计算过了的,可以通过预先储存的方式省取.

这样需要计算的就只有$q_4,k_4,v_4$以及上面的相乘就可以推出第五个词了,省了很多运算.

##### 旋转位置嵌入 RoPE

###### 原理

RoPE 的目标是"通过绝对位置编码的方式实现相对位置编码",假设通过下述运算来给 q,k 添加绝对位置信息:

$$
\begin{equation}\tilde{\boldsymbol{q}}_m = \boldsymbol{f}(\boldsymbol{q}, m), \quad\tilde{\boldsymbol{k}}_n = \boldsymbol{f}(\boldsymbol{k}, n)\end{equation}
$$

这样,经过该操作后,q,k 就带有了位置 m,n 的绝对位置信息.

而 Attention 的核心运算是内积,所以要想让内积的结果带有相对位置信息,就需要以下恒等关系:

$$
\begin{equation}\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n)\end{equation}
$$

可以通过复数运算与角的关系启发,想到如下的对二维形式的编码方法:

$$
\begin{equation} 
\boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\ q_1\end{pmatrix}\end{equation}
$$

由于内积满足线性叠加性,因此任意偶数维的 RoPE,我们都可以表示为二维情形的拼接,即:

$$
\begin{equation}\scriptsize{\underbrace{\begin{pmatrix} 
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ 
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ 
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}\end{pmatrix}}\end{equation}
$$

也就是说,给位置为 m 的向量 q 乘上矩阵$R_m$,位置为 n 的向量 k 乘上矩阵$R_n$,用变换后的 Q,K 序列做 Attention,那么 Attention 就自动包含相对位置信息了,因为成立恒等式:

$$
\begin{equation}(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) =  \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\end{equation}
$$

值得指出的是,$R_m$是一个正交矩阵,它不会改变向量的模长,因此通常来说它不会改变原模型的稳定性.

由于$R_m$的稀疏性,所以直接用矩阵乘法来实现会很浪费算力,原作者推荐通过下述方式来实现 RoPE:

$$
\begin{equation}\begin{pmatrix}q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} 
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} 
\end{pmatrix} + \begin{pmatrix}-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2} 
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} 
\end{pmatrix}\end{equation}
$$

在$\theta_i$的选择上,RoPE 同样沿用了 Sinusoidal 位置编码的方案,即$\theta_i = 10000^{-2i/d}$,它可以带来一定的远程衰减性.

###### 两种实现方法

在世界上存在两种 RoPE 的实现方法,它们都是对的,但这差异导致这两种不同实现方法训练出来的模型彼此不兼容.

HuggingFace 的 transformers 包中的 RoPE,以及 GPT-NeoX 的 RoPE 是按照以下实现的:

```python
import torch


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None        
        self.sin_cached = None
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1    )  # dim=-1 triggers a bug in torch < 1.8.0

@torch.jit.scriptdef apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Llama2 官方源码,Mesh Transformer JAX (JAX)的 RoPE 则是按照以下,这种实现更符合原作:

```python
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat


def fixed_pos_embedding(x, seq_dim=0):
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

    sinusoid_inp = np.einsum("i , j -> i j", np.arange(x.shape[seq_dim]), inv_freq)

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2)[:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)
```

此事在 transformers 包的代码中亦有记载,下面的函数的功能是完成对 k 的 linear 和 q 的 linear 的权重的一个转换，以此适配 transformers 版本的 RoPE:

```python
# permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

##### 分组查询注意力 GQA

###### 介绍

GQA(Grouped Query Attention)是在 MQA(Multi-Query Attention)的基础上改善而来的.

MQA 是原版多头注意力 MHA 的一种变体.比较极端,只有一个 KV Head,多个 Q Heads 共享相同的 KV Head.这相当于不同 Head 的 Attention 差异,全部都放在了 Query 上,需要模型仅从不同的 Q Heads 上就能够关注到输入 hidden states 不同方面的信息.这极大地降低了 KV Cache 的需求,但是会导致模型效果有所下降.

GQA 则是对 MQA 的极端的一种折中.GQA 把 Q Heads 进行分组，每组 Q Heads 对应一个 KV Head.如图,把 8 个 Q Heads 分成 4 组,每个 Grouped Query Head 包含 2 个 Q Heads,一个 Grouped Query Head 对应一个 KV Head,此时总共有 4 个 KV Heads.这样的设计使得 GQA 可以在减少计算量和 KV Cache 同时确保模型效果不受到大的影响.

###### 代码

GQA 在各个模型代码的实现中都不约而同的采用了如下的重复 kv 的方法:

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

### 分词器 Tokenizer

#### 作用

Tokenizer 将自然语言转化为 Token 序列,每一个 Token 都是一个整数.

#### BPE 与 BBPE 原理

BPE(Byte-Pair Encoding)的核心思想是事先给定一个最大分词数量,针对语料文本中的每个字符 token,逐步合并出现频率最高的连续的两个字符组合,形成一个新词 token,直到达到目标分词数量.BPE 的计算流程图如下:

而 BBPE(Byte-level BPE),则是在字节层面进行 BPE,这样对于未知语言和生僻字,就都可以编码成相应的 UTF-8 编码.但这样也会造成一个中文生僻字等于多个 token 的情况.

### Embedding 层和 Output Linear

#### Embedding 层

Tokenizer 输出的自然语言的对应 Token 序列,即一串正整数,经过 Embedding 层成为一个词向量序列.因此本质上 Embedding 层就是一个矩阵的查表,这个矩阵的第$i$行为 Token 编号为$i$的词所对应的词向量.

#### Output Linear 层

也就是最后的输出.输出的是一个长度为词表大小向量.

#### 权值共享（可选）

预训练刚兴起时,在语言模型的输出端 Output Linear 层重用 Embedding 层的权重是很常见的操作,比如 BERT,第一版的 T5,早期的 GPT,都使用了这个操作,这是因为当模型主干部分不大且词表很大时,Embedding 层的参数量很可观,如果输出端再新增一个独立的同样大小的权重矩阵的话,会导致显存消耗的激增.

权值共享最直接的后果可能是预训练的初始损失非常大.因此现在随着模型参数规模的增大,Embedding 层的占比相对变小了,这个方法也变得不常用.

### 后处理

在 Output Linear 输出后,得到一个长度为词表长度的向量,这个向量会首先除以一个参数 Temperature,即温度,这个温度控制模型输出的随机性,温度越高,随机性越强,反之则确定性越强,温度为 1 时则为正常,而温度为 0 时代表模型只会输出最大值所代表的词.

在这之后有一个 top_k 参数,它控制模型只关注最大的 k 个输出,以保证模型输出不会太离谱.凡是小于这个向量的第 k 大的值的值都会变成负无穷.

这个向量之后经过 Softmax,变成一个概率分布,之后在这个概率分布中抽样,来选取下一个词.当然如果温度为 0 就直接选择最大的那个.

## 自制深度学习框架

这里空间太小,讲不完,所以只写了构建 LLaMa 架构的必要部分.建议去看大作业提交视频和参考文献中的书『ゼロから 作 る Deep Learning ❸』斎藤 康毅.

![](static/Q8ojbaIqio3XwDxRkplcyLHWnog.png)

### Embedding 层

就是一个查表操作,这涉及到 GetItem 函数的正向和反向传播:

![](static/PLiYbXfoLoiHFKxPs2YcHoR1nsc.png)

### RMSnorm

这里为了加速运算,我们将若干个算子融合成一个 RMSnorm 的函数,实现其正向和反向传播:

```python
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
```

数学原理如下:

RMSnorm 计算如下:

$\hat{x} = \frac x{\sqrt {\frac 1n\sum _{i= 1}^{n}x_{i}^{2}+ \epsilon }}$, $RMS= \sqrt {\frac 1n\sum _{i= 1}^{2}x_{i}^{2}+ \epsilon }$

$$
y=g\odot \hat{x}
$$

为了方便,记:

$$
\sigma^2=\frac 1n\sum _{i= 1}^{n}x_{i}^{2}
$$

那么:

$$
\frac{\partial L}{\partial g_{i}}=\frac{\partial L}{\partial y_{i}}\hat{x}_{i}
$$

$$
\frac{\partial L}{\partial x_{i}}=\sum_{j=1}^{n}\frac{\partial L}{\partial{y}_{j}}\frac{\partial y_{j}}{\partial \hat{x}_{j}}\frac{\partial \hat{x}_{j}}{\partial x_{j}}=\sum_{j=1}^{n}\frac{\partial L}{\partial y_{j}}\cdot g_{j}\cdot\frac{\partial \hat{x}_{j}}{\partial x_{j}}
$$

由乘法求导法则得:

$$
\frac{\partial\hat{x}_{i}}{\partial x_{i}}=\delta_{ij}(\sigma^{2}+\epsilon)^{-\frac{1}{2}}-\frac{1}{n}x_{i}x_{j}(\sigma^{2}+\epsilon)^{-\frac{3}{2}}
$$

代入有:

$$
\frac{\partial L}{\partial x_{i}}=\sum_{j=1}^{n}\frac{\partial L}{\partial y_{j}}g_{j}\delta_{ij}(\sigma^{2}+\epsilon)^{-\frac{1}{2}}-\sum_{j=1}^{n}\frac{\partial L}{\partial y_{j}}g_{j}\frac{1}{n}x_{i}x_{j}(\sigma^2+\epsilon)^{-\frac{3}{2}}
$$

前一项由于$\delta_{ij}$只剩一个,后一项提出无关求和的:

$$
=\frac{\partial L}{\partial y_{i}}g_{i}(\sigma^{2}+\epsilon)^{-\frac{1}{2}}-\frac{1}{n}x_{i}(\sigma^2+\epsilon)^{-\frac{3}{2}}\sum_{j=1}^{n}\frac{\partial{L}}{\partial y_{j}}g_{j}x_{j}
$$

### Attention Block

这里的算子可就多了.

#### Linear 层和矩阵乘法

若$\mathbf{y}=\mathbf{x}\mathbf{W},L=f(\mathbf{y})$,则:

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^\mathrm{T}$,

$$
\frac{\partial L}{\partial W} = x^\mathrm{T}\frac{\partial L}{\partial y}
$$

#### Reshape 和 Transpose

![](static/SJobbnnYSoSQ2qx1WDxc2tmCnff.png)

Transpose 同理.

### SwiGLU FFN

除了上面的 Linear 层,这里还有:

#### 乘法

即为矩阵按元素相乘,这就很基础.

#### SiLU

```python
class SiLU(Function):
    def forward(self, x: np.ndarray) -> tuple[np.ndarray]:
        self.sigmoid = 1 / (1 + np.exp(-x))
        y = x * self.sigmoid
        return y

    def backward(self, gy: np.ndarray) -> Union[tuple[np.ndarray, ...], np.ndarray]:
        y = self.outputs[0]()
        gx = gy * (y + self.sigmoid * (1 - y))
        return gx


def silu(x):
    return SiLU()(x)
```

$\mathbf{SiLU}(x)=x\sigma(x)$自然反向也好求.

$\frac{\partial \sigma(x)}{\partial{x}}=\sigma(x)(1-\sigma(x))$,

$$
\frac{\partial \mathbf{SiLU}(x)}{\partial{x}}=(\sigma(x)+x\sigma(x)(1-\sigma(x)))
$$

## 参考文献

作业原本提供:

[https://github.com/naklecha/llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)

llama3(不可训练)numpy 实现:

[https://github.com/likejazz/llama3.np](https://github.com/likejazz/llama3.np)

Baby llama:

[https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)

Atom7b:

[https://github.com/LlamaFamily/Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese)

RMSnorm 论文:

[https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)

llama3 的 ffn SwiGLU 论文:

[https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)

注意力论文:

[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

注意力介绍博文:

[https://spaces.ac.cn/archives/4765](https://spaces.ac.cn/archives/4765)

注意力介绍:

[https://armanasq.github.io/nlp/self-attention/](https://armanasq.github.io/nlp/self-attention/)

RoPE 论文 RoFormer:

[https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)

RoPE 原作者亲自讲解:

[https://spaces.ac.cn/archives/8265](https://spaces.ac.cn/archives/8265)

RoPE 介绍与实现:

https://blog.eleuther.ai/rotary-embeddings/

RoPE 两种实现的引发的不同:

[https://github.com/huggingface/transformers/issues/25199](https://github.com/huggingface/transformers/issues/25199)

关于权重共享:

[https://spaces.ac.cn/archives/9698](https://spaces.ac.cn/archives/9698)

『ゼロから 作 る Deep Learning ❸』(O'Reilly Japan, 2020):

[https://github.com/oreilly-japan/deep-learning-from-scratch-3](https://github.com/oreilly-japan/deep-learning-from-scratch-3)

上面那本书的读者实现的 stack:

[https://github.com/laksjdjf/dezero-diffusion/blob/main/modules/unet.py](https://github.com/laksjdjf/dezero-diffusion/blob/main/modules/unet.py)
