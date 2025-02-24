import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ex_Transformer(nn.Module):
    def __init__(self, configs):
        super(ex_Transformer, self).__init__()
        
        # Transformer解码器层
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=configs.d_model, nhead=8)
            for _ in range(configs.d_layers)
        ])
        
        # 输入线性层
        self.input_linear = nn.Linear(configs.c_in + configs.ex_dim, configs.d_model)  # 1是时间序列的维度，dim是辅助变量的维度
        
        # 输出线性层
        self.output_linear = nn.Linear(configs.d_model, configs.c_in)  # 输出维度为1
        self.proj = nn.Linear(1, configs.ex_c_out)
        self.initialize_to_zeros(self.decoder_layers)
        self.initialize_to_zeros(self.input_linear)
        self.initialize_to_zeros(self.output_linear)
        self.initialize_to_zeros(self.proj)
    
    def initialize_to_zeros(self,model):
        for param in model.parameters():
            if param.requires_grad:  # 确保只对可训练参数进行初始化
                nn.init.zeros_(param)
    

    def forward(self, X, X_ex):
        # 假设 X 的形状是 [Batch, seq, dim1, dim2]
        # 将 X 进行处理，假设我们对 dim2 进行平均或其他操作
        X_processed = X.mean(dim=-1)  # 现在的形状是 [Batch, seq, dim1]
        
        # 将处理后的预测值和辅助变量拼接
        combined_input = torch.cat([X_processed, X_ex], dim=-1)  # [Batch, seq, dim1 + dim]
        
        # 通过线性层映射到Transformer的输入维度
        combined_input = self.input_linear(combined_input)  # [Batch, seq, d_model]
        
        # 转换为Transformer的输入格式
        combined_input = combined_input.permute(1, 0, 2)  # [seq, Batch, d_model]
        
        # 初始化解码器的输入
        tgt = combined_input  # 这里我们直接使用combined_input作为目标序列
        
        # 通过每一层Transformer解码器
        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt)  # 自注意力机制
        
        # 转换回原来的维度
        tgt = tgt.permute(1, 0, 2)  # [Batch, seq, d_model]
        
        # 通过输出线性层得到最终预测
        output = self.output_linear(tgt)  # [Batch, seq, c_in]
        
        return self.proj(output.unsqueeze(-1))+X  # [Batch, seq, ex_c_out]
