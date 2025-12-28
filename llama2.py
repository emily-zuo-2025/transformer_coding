from transformers import PretrainedConfig
import torch
import torch.nn as nn

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
    self,
    dim: int = 768, # 模型维度
    n_layers: int = 12, # Transformer的层数
    n_heads: int = 16, # 注意⼒机制的头数
    n_kv_heads: int = 8, # 键值头的数量
    vocab_size: int = 6144, # 词汇表⼤⼩
    hidden_dim: int = None, # 隐藏层维度
    multiple_of: int = 64,
    norm_eps: float = 1e-5, # 归⼀化层的eps
    max_seq_len: int = 512, # 最⼤序列⻓度
    dropout: float = 0.0, # dropout概率
    flash_attn: bool = True, # 是否使⽤Flash Attention
    **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps是为了防⽌除以0的情况
        self.eps = eps
        # weight是⼀个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # 计算RMSNorm的核⼼部分
        # x.pow(2).mean(-1, keepdim=True)计算了输⼊x的平⽅的均值
        # torch.rsqrt是平⽅根的倒数，这样就得到了RMSNorm的分⺟部分，再加上eps防⽌分⺟为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # forward函数是模型的前向传播
        # ⾸先将输⼊x转为float类型，然后进⾏RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的⼀个可学习的缩放因⼦
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
# Test case
args = ModelConfig()
norm = RMSNorm(args.dim, args.norm_eps)
x = torch.randn(1, 50, args.dim)
output = norm(x)
print(output.shape)
