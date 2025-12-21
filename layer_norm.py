class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
	super().__init__()
    # 线性矩阵做映射
	self.a_2 = nn.Parameter(torch.ones(features))
	self.b_2 = nn.Parameter(torch.zeros(features))
	self.eps = eps
	
    def forward(self, x):
		# 在统计每个样本所有维度的值，求均值和方差
		mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
		std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
		# 注意这里也在最后一个维度发生了广播
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2