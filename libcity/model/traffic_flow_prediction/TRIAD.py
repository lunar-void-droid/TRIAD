
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from functools import partial
from einops import rearrange, repeat
from logging import getLogger

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from libcity.model.tlast_layers.DIFFormer import DIFFormerConv
from libcity.model.tlast_layers.ProbAttention import ProbAttention



# Attention Layers (注意力层模块)
class MultiHeadsAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, scale=None, attention_dropout=0.1, returnA=False):
        super(MultiHeadsAttention, self).__init__()
        self.scale = scale
        self.returnA = returnA
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.returnA:
            return V.contiguous(), A.contiguous()
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, d_keys=None, d_values=None, mix=True, 
                 dropout=0.1, returnA=False, att_type='full', prob_factor=5):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (hid_dim//n_heads)
        d_values = d_values or (hid_dim//n_heads)

        if att_type == 'full' or att_type == 'proxy':
            self.inner_attention = MultiHeadsAttention(
                scale=None, attention_dropout=dropout, returnA=returnA)
        elif att_type == 'prob':
            self.inner_attention = ProbAttention(
                False, prob_factor, attention_dropout=dropout, output_attention=returnA)
        
        self.query_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(hid_dim, d_keys * n_heads)
        self.value_projection = nn.Linear(hid_dim, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hid_dim)
        self.n_heads = n_heads
        self.returnA = returnA
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, A = self.inner_attention(queries, keys, values)
        
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        
        if self.returnA:
            return out, A
        else:
            return out, None



# Encoder Layers (编码器层模块)
class DIFFormer_EncoderLayer(nn.Module):
    """DIFFormer编码器层"""
    def __init__(self, hid_dim, n_heads, norm_flag, query_cross_time=True,
                 d_ff=None, dropout=0.1, return_att=False, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att
        self.att_layer = DIFFormerConv(hid_dim, hid_dim, num_heads=n_heads, output_attn=return_att)
        self.dropout = nn.Dropout(dropout)
        self.norm_flag = norm_flag
        assert norm_flag in ['none', 'first', 'after']
        self.query_cross_time = query_cross_time
        
        if norm_flag == 'first' and query_cross_time: 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)
        
        assert activation in ['gelu','relu','GLU']
        activation_func_dict = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'GLU': nn.GLU()
        }
        if activation == 'GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP1 = nn.Sequential(
            nn.Linear(hid_dim, d_ff1),
            activation_func_dict[activation],
            nn.Linear(d_ff, hid_dim)
        )

    def forward(self, data):
        batch = data.shape[0]
        T = data.shape[1]
        
        if self.query_cross_time:
            now = data[:, -1, :, :]
            query = repeat(now, 'b n c -> (b t) n c', t=T)
            data = rearrange(data, 'b t n c-> (b t) n c')
            
            if self.norm_flag == 'first': 
                query = self.norm0(query)     
                x = self.norm1(data)
            else:
                x = data
            x, A = self.att_layer(query, x)
        else:
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag == 'first': 
                x = self.norm1(data)
            else: 
                x = data
            x, A = self.att_layer(x, x)
        
        x = data + self.dropout(x)
        if self.norm_flag == 'first': 
            x = self.norm2(x)
        elif self.norm_flag == 'after': 
            x = self.norm1(x)
        
        x = x + self.dropout(self.MLP1(x))
        if self.norm_flag == 'after': 
            x = self.norm2(x)

        final_out = rearrange(x, '(b T) N d -> b T N d', b=batch)
        if self.return_att:
            A = rearrange(A, '(b t) h l s -> b t h l s', b=batch)
            return final_out, A
        return final_out, None


class Full_Prob_EncoderLayer(nn.Module):
    """Full/Prob Attention编码器层"""
    def __init__(self, hid_dim, n_heads, att_type, norm_flag='first', d_ff=None, 
                 dropout=0.1, att_dropout=0.1, return_att=False, activation='gelu', 
                 query_cross_time=True, prob_factor=None):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.return_att = return_att
        self.query_cross_time = query_cross_time
        self.att_layer = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type=att_type, 
            returnA=return_att, prob_factor=prob_factor
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_flag = norm_flag
        
        assert norm_flag in ['none', 'first', 'after']
        if norm_flag == 'first' and query_cross_time: 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)

        assert activation in ['gelu','relu','GLU']
        activation_func_dict = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'GLU': nn.GLU()
        }
        if activation == 'GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP1 = nn.Sequential(
            nn.Linear(hid_dim, d_ff1),
            activation_func_dict[activation],
            nn.Linear(d_ff, hid_dim)
        )

    def forward(self, data):
        batch = data.shape[0]
        T = data.shape[1]
        
        if self.query_cross_time:
            now = data[:, -1, :, :]
            query = repeat(now, 'b n c -> (b t) n c', t=T)               
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag == 'first': 
                query = self.norm0(query)     
                x = self.norm1(data)
            else: 
                x = data
            x, A = self.att_layer(query, x, x)
        else:
            data = rearrange(data, 'b t n c-> (b t) n c')
            if self.norm_flag == 'first': 
                x = self.norm1(data)
            else: 
                x = data
            x, A = self.att_layer(x, x, x)
        
        x = data + self.dropout(x)
        if self.norm_flag == 'first': 
            x = self.norm2(x)
        elif self.norm_flag == 'after': 
            x = self.norm1(x)
        x = x + self.dropout(self.MLP1(x))
        if self.norm_flag == 'after': 
            x = self.norm2(x)

        final_out = rearrange(x, '(b T) N d -> b T N d', b=batch)
        if self.return_att:
            A = rearrange(A, '(b t) h l s -> b t h l s', b=batch)
            return final_out, A
        return final_out, None


class ProxyEncoderLayer(nn.Module):
    """Proxy Attention编码器层"""
    def __init__(self, factor, hid_dim, n_heads, num_nodes=None, d_ff=None, 
                 dropout=0.1, att_dropout=0.1, activation='gelu',
                 query_cross_time=True, norm_flag=1, dynamic_proxy=True, return_att=False):
        super().__init__()
        d_ff = d_ff or 4*hid_dim
        self.query_cross_time = query_cross_time
        self.norm_flag = norm_flag
        self.return_att = return_att
        self.dynamic_proxy = dynamic_proxy
        assert num_nodes is not None
        
        if dynamic_proxy:
            self.readout_fc = nn.Linear(num_nodes, factor)
        else:
            self.proxy_tensor = nn.Parameter(torch.zeros(1, factor, hid_dim))
            nn.init.xavier_uniform_(self.proxy_tensor)
            
        self.node2proxy = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att
        )
        self.proxy2node = AttentionLayer(
            hid_dim, n_heads, dropout=att_dropout, att_type='proxy', returnA=return_att
        )
        self.dropout = nn.Dropout(dropout)

        if norm_flag == 'first': 
            self.norm0 = nn.LayerNorm(hid_dim)
        if norm_flag != 'none':
            self.norm1 = nn.LayerNorm(hid_dim)
            self.norm2 = nn.LayerNorm(hid_dim)

        assert activation in ['gelu','relu','GLU']
        activation_func_dict = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'GLU': nn.GLU()
        }
        if activation == 'GLU':
            d_ff1 = d_ff*2
        else:
            d_ff1 = d_ff
        self.MLP2 = nn.Sequential(
            nn.Linear(hid_dim, d_ff1),
            activation_func_dict[activation],
            nn.Linear(d_ff, hid_dim)
        )

    def forward(self, data):
        now = data[:, -1, :, :]
        batch = data.shape[0]
        T = data.shape[1]

        if self.dynamic_proxy:
            if self.query_cross_time:
                now = data[:, -1, :, :]
                temp = rearrange(now, 'b n c -> b c n')
                z_proxy = self.readout_fc(temp)
                z_proxy = repeat(z_proxy, 'b c k -> (b repeat) c k', repeat=T)
                z_proxy = rearrange(z_proxy, 'bt c K -> bt K c')
            else:
                temp = rearrange(data, 'b t n c -> b t c n')
                z_proxy = self.readout_fc(temp)
                z_proxy = rearrange(z_proxy, 'b t c K -> (b t) K c')     
        else:
            z_proxy = repeat(self.proxy_tensor, '1 m d ->(1 repeat) m d', repeat=T*batch)

        data = rearrange(data, 'b t n c-> (b t) n c')

        if self.norm_flag == 'first':
            z_proxy = self.norm0(z_proxy)
            data = self.norm1(data)
            proxy_feature, A1 = self.node2proxy(z_proxy, data, data)      
            node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
            enc_feature = data + self.dropout(node_feature)
            enc_feature = self.norm2(enc_feature)
            enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))
        elif self.norm_flag == 'after':
            proxy_feature, A1 = self.node2proxy(z_proxy, data, data)
            node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
            enc_feature = data + self.dropout(node_feature)
            enc_feature = self.norm1(enc_feature)
            enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))
            enc_feature = self.norm2(enc_feature)
        elif self.norm_flag == 'none':
            proxy_feature, A1 = self.node2proxy(z_proxy, data, data)
            node_feature, A2 = self.proxy2node(data, proxy_feature, proxy_feature)
            enc_feature = data + self.dropout(node_feature)
            enc_feature = enc_feature + self.dropout(self.MLP2(enc_feature))

        final_out = rearrange(enc_feature, '(b T) N hid_dim -> b T N hid_dim', b=batch)
        if self.return_att:
            A1 = rearrange(A1, '(b t) h l s -> b t h l s', b=batch)
            A2 = rearrange(A2, '(b t) h l s -> b t h l s', b=batch)
            return final_out, [A1, A2]
        else:
            return final_out, None



# Embedding Layers (嵌入层模块)
class getTimeEmbedding(nn.Module):
    """时间嵌入层"""
    def __init__(self, hid_dim, slice_size_per_day, emb_dropout=0.1):
        super().__init__()
        self.time_in_day_embedding = nn.Embedding(slice_size_per_day, hid_dim)
        self.day_in_week_embedding = nn.Embedding(7, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, time_data):
        t_hour = time_data[..., 0:1].squeeze(-1).long()
        t_day = time_data[..., 1:2].squeeze(-1).long()
        time_in_day_emb = self.time_in_day_embedding(t_hour)
        day_in_week_emb = self.day_in_week_embedding(t_day)
        time_emb = time_in_day_emb + day_in_week_emb
        time_emb = self.dropout(time_emb)    
        return time_emb


class getTimeLagEmbedding(nn.Module):
    """多视图时间滞后嵌入"""
    def __init__(self, in_dim, out_dim, emb_dropout=0.1, num_experts: int = 4,
                 use_adaptive_lag: bool = True, max_lag: int = 6, lag_tau: float = 1.0,
                 ablate_mlp: bool = False, ablate_fft: bool = False, ablate_moe: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = max(1, int(num_experts))
        
        # 自适应滞后参数
        self.use_adaptive_lag = use_adaptive_lag
        self.max_lag = max_lag
        self.lag_tau = lag_tau

        self.ablate_mlp = ablate_mlp
        self.ablate_fft = ablate_fft
        self.ablate_moe = ablate_moe
        

        self._debug_lag_weights_once = True

        self.time_lag_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

        self.moe_gate = nn.Linear(in_dim, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, in_dim)
            ) for _ in range(self.num_experts)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        

        if self.use_adaptive_lag:
            self.lag_selector = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, max_lag + 1)
            )

    def forward(self, time_emb, ref_emb=None):
        """
            time_emb: (B, T, H) 时间嵌入
            ref_emb: (B, 1, H)
            out: (B, T, H) 时间滞后嵌入
        """
        if self.use_adaptive_lag:
            # 自适应多滞后模式
            diff = self._compute_adaptive_lag_diff(time_emb)
        else:

            if ref_emb is None:
                ref_emb = time_emb[:, -1:]
            diff = time_emb - ref_emb
        
        # 三视图融合
        view_time = self.time_lag_mlp(diff)
        view_freq = torch.fft.fft(diff, dim=-1).real
        gate_logits = self.moe_gate(diff)
        gates = torch.softmax(gate_logits, dim=-1)
        expert_outs = []
        for exp in self.experts:
            expert_outs.append(exp(diff))
        expert_stack = torch.stack(expert_outs, dim=-1)
        view_sem = torch.sum(expert_stack * gates.unsqueeze(-2), dim=-1)
        

        if self.ablate_mlp:
            view_time = torch.zeros_like(view_time)
        if self.ablate_fft:
            view_freq = torch.zeros_like(view_freq)
        if self.ablate_moe:
            view_sem = torch.zeros_like(view_sem)
        
        fused = torch.cat([view_time, view_freq, view_sem], dim=-1)
        out = self.fusion(fused)
        out = self.dropout(out)
        return out
    
    def _compute_adaptive_lag_diff(self, time_emb):
        """
        计算自适应多滞后加权差分
            time_emb: (B, T, H)
            diff: (B, T, H) 加权融合后的差分
        """
        B, T, H = time_emb.shape
        K = self.max_lag + 1
        

        diffs_list = []
        for k in range(K):
            if k == 0:
                ref_k = time_emb  # (B, T, H)
            else:

                padding = time_emb[:, 0:1, :].expand(B, k, H)  # (B, k, H)
                ref_k = torch.cat([padding, time_emb[:, :-k, :]], dim=1)  # (B, T, H)
            diff_k = time_emb - ref_k  # (B, T, H)
            diffs_list.append(diff_k)
        
        # diffs: (B, T, H, K)
        diffs = torch.stack(diffs_list, dim=-1)
        

        logits = self.lag_selector(time_emb)  # (B, T, K)
        tau = max(self.lag_tau, 1e-6)
        weights = F.softmax(logits / tau, dim=-1)
        

        if self._debug_lag_weights_once:
            with torch.no_grad():
                w_mean = weights.mean(dim=(0, 1))
                w_sample = weights[0, :3, :]
                print(f"[AdaptiveLag] 权重均值 (K={K}): {w_mean.cpu().numpy().round(3)}")
                print(f"[AdaptiveLag] 样本权重 (t=0,1,2):\n{w_sample.cpu().numpy().round(3)}")
            self._debug_lag_weights_once = False
        

        weights = weights.unsqueeze(2)  # (B, T, 1, K)
        diff = torch.sum(diffs * weights, dim=-1)  # (B, T, H)
        
        return diff


class getSpatialEmbedding(nn.Module):
    """空间嵌入层"""
    def __init__(self, hid_dim, num_nodes, emb_dropout=0.1):
        super().__init__()
        self.spatial_embedding = nn.Embedding(num_nodes, hid_dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, spatial_indexs=None):
        if spatial_indexs is None:
            batch, _, num_nodes, _ = x.shape
            spatial_indexs = torch.LongTensor(torch.arange(num_nodes)).to(x.device)
        spatial_emb = self.spatial_embedding(spatial_indexs).unsqueeze(0).unsqueeze(1)
        spatial_emb = self.dropout(spatial_emb)
        return spatial_emb


class DataEncoding(nn.Module):
    """数据编码层"""
    def __init__(self, in_dim, hid_dim, hasCross=True, activation='relu'):
        super().__init__()
        assert activation in ['gelu', 'relu']
        in_units = in_dim*2 if hasCross else in_dim
        self.hasCross = hasCross
        self.linear1 = nn.Linear(in_units, hid_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, latestX):
        if self.hasCross:
            data = torch.cat([x, latestX], dim=-1)
        else:
            data = x
        data = self.linear1(data)
        data = self.activation(data)
        data = self.linear2(data)
        return data



# Time-series Decomposition (时序分解模块)
class TimeSeriesDecomposition(nn.Module):
    """时序分解模块（原始 AvgPool 版本）"""
    def __init__(self, hid_dim, kernel_size=3):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1
        self.avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, 
                                     padding=kernel_size//2, count_include_pad=False)

    def forward(self, x):
        B, T, N, C = x.shape
        x_bnct = rearrange(x, 'b t n c -> (b n) c t')
        trend_bnct = self.avgpool(x_bnct)
        trend = rearrange(trend_bnct, '(b n) c t -> b t n c', b=B, n=N)
        seasonal = x - trend
        return trend, seasonal


class LearnableMultiScaleDecomp(nn.Module):
    """
    可学习多尺度时序分解模块
    x: (B, T, N, C)
    return trend, seasonal: (B, T, N, C)
    """
    def __init__(self, hid_dim, scales=(3, 7, 13), temp=1.0, dropout=0.0, norm="ln"):
        super().__init__()
        self.scales = list(scales)
        self.temp = float(temp)
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        

        self._debug_scale_weights_once = True


        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))


        self.trend_extractors = nn.ModuleList([
            nn.Conv1d(
                in_channels=hid_dim,
                out_channels=hid_dim,
                kernel_size=k,
                padding=k // 2,
                groups=hid_dim,   # depthwise
                bias=False
            )
            for k in self.scales
        ])

        if norm == "ln":
            self.norm = nn.LayerNorm(hid_dim)
        else:
            self.norm = nn.Identity()


        self.seasonal_gate = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        B, T, N, C = x.shape
        # (B, T, N, C) -> (B*N, C, T)
        x_bnct = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)

        trends = []
        for conv in self.trend_extractors:
            t = conv(x_bnct)  # (B*N, C, T)
            trends.append(t)


        weights = F.softmax(self.scale_logits / max(self.temp, 1e-6), dim=0)
        

        if self._debug_scale_weights_once:
            with torch.no_grad():
                print(f"[MS-Decomp] 尺度权重 scales={self.scales}: {weights.cpu().numpy().round(3)}")
            self._debug_scale_weights_once = False

        trend_bnct = 0.0
        for w, t in zip(weights, trends):
            trend_bnct = trend_bnct + w * t


        trend = trend_bnct.view(B, N, C, T).permute(0, 3, 1, 2).contiguous()
        trend = self.norm(trend)

        seasonal = x - trend
        seasonal = self.dropout(seasonal)

        # gated seasonal (stabilize)
        gate = self.seasonal_gate(seasonal)
        seasonal = seasonal * gate

        return trend, seasonal



# Predictor (预测器模块)
class TimeWisePredictor(nn.Module):

    def __init__(self, num_nodes, input_length, predict_length, in_dim, pre_dim,
                 num_of_filters=128, activation='gelu'):
        super(TimeWisePredictor, self).__init__()
        self.num_nodes = num_nodes
        self.input_length = input_length
        self.in_dim = in_dim
        self.pre_dim = pre_dim
        self.predict_length = predict_length
        assert activation in ['gelu', 'relu']
        
        self.predict_unit = nn.Sequential(
            nn.Linear(input_length*in_dim, num_of_filters),
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )
        
        self.predict_unit_list = nn.ModuleList([
            nn.Linear(num_of_filters, pre_dim) for _ in range(predict_length)
        ])

    def forward(self, data):
        data = rearrange(data, 'b t n c -> b n (t c)')
        data = self.predict_unit(data)
        need_concat = []
        B, N, _ = data.shape
       
        for j in range(self.predict_length):
            unit_out = self.predict_unit_list[j](data)
            unit_out = rearrange(unit_out, 'b n c -> b 1 n c')
            need_concat.append(unit_out)
        
        final_out = torch.cat(need_concat, dim=1)
        return final_out




class TRIADModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_length = args.input_length
        self.predict_length = args.predict_length
        self.in_dim = args.in_dim
        self.pre_dim = args.pre_dim
        self.num_nodes = args.num_nodes
        self.tau = args.tau
        self.hid_dim = args.hid_dim
        self.n_heads = args.n_heads
        self.M = args.M
        self.d_out = args.d_out
        self.query_cross_time = args.query_cross_time
        self.return_att = bool(args.return_att)
        self.num_layers = args.num_layers
        self.norm_flag = args.norm_flag
        self.att_type = args.att_type
        self.activation_data = args.activation_data
        self.activation_enc = args.activation_enc
        self.activation_dec = args.activation_dec
        self.emb_dropout = args.emb_dropout
        self.te_emb_dropout = args.te_emb_dropout
        self.se_emb_dropout = args.se_emb_dropout
        self.tle_emb_dropout = args.tle_emb_dropout
        self.enc_dropout = args.enc_dropout
        self.att_dropout = args.att_dropout
        self.prob_factor = args.prob_factor
        self.slice_size_per_day = args.slice_size_per_day
        self.use_adaptive_lag = args.use_adaptive_lag
        self.max_lag = args.max_lag
        self.lag_tau = args.lag_tau
        
        # MS-Decomp 参数
        self.use_ms_decomp = args.use_ms_decomp
        self.ms_decomp_scales = args.ms_decomp_scales
        self.ms_decomp_temp = args.ms_decomp_temp
        self.ms_decomp_dropout = args.ms_decomp_dropout
        self.ms_decomp_norm = args.ms_decomp_norm
        
        # TLE消融参数
        self.tle_ablate_mlp = getattr(args, 'tle_ablate_mlp', False)
        self.tle_ablate_fft = getattr(args, 'tle_ablate_fft', False)
        self.tle_ablate_moe = getattr(args, 'tle_ablate_moe', False)

        assert self.att_type in ['proxy', 'prob', 'full', 'difformer']
        assert self.norm_flag in ['none', 'first', 'after']

        # 数据编码
        self.data_encoding = DataEncoding(
            self.in_dim, self.hid_dim, True, activation=self.activation_data
        )
        self.x_dropout = nn.Dropout(p=self.emb_dropout)
        
        # 嵌入层
        self.getTemb = getTimeEmbedding(
            self.hid_dim, self.slice_size_per_day, self.te_emb_dropout
        )
        self.get_xTimeLagEmb = getTimeLagEmbedding(
            self.hid_dim, self.hid_dim, self.tle_emb_dropout,
            use_adaptive_lag=self.use_adaptive_lag,
            max_lag=self.max_lag,
            lag_tau=self.lag_tau,
            ablate_mlp=self.tle_ablate_mlp,
            ablate_fft=self.tle_ablate_fft,
            ablate_moe=self.tle_ablate_moe
        )
        self.getSemb = getSpatialEmbedding(
            self.hid_dim, self.num_nodes, self.se_emb_dropout
        )
        
        # 时序分解模块
        if self.use_ms_decomp:
            self.decomp = LearnableMultiScaleDecomp(
                hid_dim=self.hid_dim,
                scales=self.ms_decomp_scales,
                temp=self.ms_decomp_temp,
                dropout=self.ms_decomp_dropout,
                norm=self.ms_decomp_norm
            )
        else:
            self.decomp = TimeSeriesDecomposition(self.hid_dim, kernel_size=3)

        # 编码器层
        if self.att_type in ['full', 'prob']:
            self.spatial_agg_list = nn.ModuleList([
                Full_Prob_EncoderLayer(
                    self.hid_dim, n_heads=self.n_heads, dropout=self.enc_dropout, 
                    att_type=self.att_type, norm_flag=self.norm_flag,
                    query_cross_time=self.query_cross_time, att_dropout=self.att_dropout,
                    return_att=self.return_att, prob_factor=self.prob_factor,
                    activation=self.activation_enc
                ) for _ in range(self.num_layers)
            ])
        elif self.att_type == 'difformer':
            self.spatial_agg_list = nn.ModuleList([
                DIFFormer_EncoderLayer(
                    self.hid_dim, n_heads=self.n_heads, norm_flag=self.norm_flag,
                    query_cross_time=self.query_cross_time, dropout=self.enc_dropout, 
                    return_att=self.return_att, activation=self.activation_enc
                ) for _ in range(self.num_layers)
            ])
        elif self.att_type == 'proxy':
            self.spatial_agg_list = nn.ModuleList([
                ProxyEncoderLayer(
                    self.M, self.hid_dim, n_heads=self.n_heads, num_nodes=self.num_nodes, 
                    norm_flag=self.norm_flag, query_cross_time=self.query_cross_time,
                    dropout=self.enc_dropout, dynamic_proxy=True, att_dropout=self.att_dropout,
                    return_att=self.return_att, activation=self.activation_enc
                ) for _ in range(self.num_layers)
            ])

        # 预测器
        self.time_wise_predictor = TimeWisePredictor(
            self.num_nodes, self.input_length, self.predict_length, 
            self.hid_dim, pre_dim=self.pre_dim, num_of_filters=self.d_out, 
            activation=self.activation_dec
        )

    def forward(self, x, x_time):
        B, T, N, _ = x.shape
        
        assert x.shape[-1] == self.in_dim
        assert T == self.input_length
        assert x_time.shape == (B, T, 2)

        # 1. 数据编码
        latestX = x[:, -1:, :, :].repeat([1, self.input_length, 1, 1])
        data = self.data_encoding(x, latestX)
        data = self.x_dropout(data)

        # 2. 时间嵌入
        x_time_emb = self.getTemb(x_time)
        x_time_emb_expanded = x_time_emb.unsqueeze(2).expand(B, T, N, -1)
        data = data + x_time_emb_expanded

        # 3. 时间滞后嵌入
        x_time_lag_emb = self.get_xTimeLagEmb(
            x_time_emb, x_time_emb[:, -1:]
        )
        x_time_lag_emb_expanded = x_time_lag_emb.unsqueeze(2).expand(B, T, N, -1)
        data = data + x_time_lag_emb_expanded

        # 4. 空间嵌入
        x_spa_emb = self.getSemb(data)
        data = data + x_spa_emb
        
        # 5. 时序分解
        trend, seasonal = self.decomp(data)

        # 6. 多层空间注意力编码器
        A_list = []
        x_trend = trend
        x_seasonal = seasonal
        for i in range(self.num_layers):
            x_trend, A_t = self.spatial_agg_list[i](x_trend)
            x_seasonal, A_s = self.spatial_agg_list[i](x_seasonal)
            A_list.append((A_t, A_s))
        data = x_trend + x_seasonal

        # 7. 时间维度预测
        main_output = self.time_wise_predictor(data)

        # 8. 残差连接
        if self.input_length == self.predict_length:
            main_output = main_output + latestX[:, :, :, :self.pre_dim]
        else:
            main_output = main_output + latestX[:, 0:self.predict_length, :, :self.pre_dim]

        if self.return_att:
            return main_output, A_list
        return main_output, None




class TRIAD(AbstractTrafficStateModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        self._logger = getLogger(__name__)
        self._scaler = self.data_feature.get('scaler')
        

        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.output_dim = config.get('output_dim', 1)

        self.input_dim = config.get('input_dim', self.output_dim)

        self.x_feature_dim = config.get('x_feature_dim', None)

        self.feature_dim = self.input_dim

        self._debug_dataflow_once = True
        
        # 时间参数
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        
        # 模型参数
        self.hid_dim = config.get('hid_dim', 64)
        self.d_out = config.get('d_out', 1024)
        self.n_heads = config.get('n_heads', 2)
        self.M = config.get('M', 8)
        self.num_layers = config.get('num_layers', 1)
        self.tau = config.get('tau', 3)
        
        # 注意力参数
        self.att_type = config.get('att_type', 'proxy')
        self.query_cross_time = config.get('query_cross_time', 1)
        self.return_att = config.get('return_att', 0)
        self.prob_factor = config.get('prob_factor', 5)
        
        # 正则化参数
        self.norm_flag = config.get('norm_flag', 'first')
        self.enc_dropout = config.get('enc_dropout', 0.1)
        self.att_dropout = config.get('att_dropout', 0.1)
        self.emb_dropout = config.get('emb_dropout', 0.1)
        self.te_emb_dropout = config.get('te_emb_dropout', 0.1)
        self.se_emb_dropout = config.get('se_emb_dropout', 0.1)
        self.tle_emb_dropout = config.get('tle_emb_dropout', 0.1)
        
        # 激活函数参数
        self.activation_data = config.get('activation_data', 'relu')
        self.activation_enc = config.get('activation_enc', 'gelu')
        self.activation_dec = config.get('activation_dec', 'gelu')
        
        # 时间参数
        self.add_time_in_day = config.get("add_time_in_day", True)
        self.add_day_in_week = config.get("add_day_in_week", True)
        self.slice_size_per_day = config.get('slice_size_per_day', 288)
        
        # 自适应滞后参数
        self.use_adaptive_lag = config.get("use_adaptive_lag", True)
        self.max_lag = config.get("max_lag", 6)
        self.lag_tau = config.get("lag_tau", 1.0)
        
        # MS-Decomp 参数
        self.use_ms_decomp = config.get("use_ms_decomp", True)
        self.ms_decomp_scales = config.get("ms_decomp_scales", [3, 7, 13])
        self.ms_decomp_temp = config.get("ms_decomp_temp", 1.0)
        self.ms_decomp_dropout = config.get("ms_decomp_dropout", 0.0)
        self.ms_decomp_norm = config.get("ms_decomp_norm", "ln")
        
        # TLE消融参数
        self.tle_ablate_mlp = config.get("tle_ablate_mlp", False)
        self.tle_ablate_fft = config.get("tle_ablate_fft", False)
        self.tle_ablate_moe = config.get("tle_ablate_moe", False)
        
        # 损失函数参数
        self.huber_delta = config.get('huber_delta', 2)
        self.set_loss = config.get('set_loss', 'masked_mae')
        self.device = config.get('device', torch.device('cpu'))
        self.time_intervals = config.get('time_intervals', 300)
        

        self._build_model()
        
        self._logger.info('✨ TRIAD模型初始化完成')
        self._logger.info(f'   - 节点数: {self.num_nodes}')
        self._logger.info(f'   - 输入窗口: {self.input_window}, 输出窗口: {self.output_window}')
        self._logger.info(f'   - 隐藏维度: {self.hid_dim}, 注意力头数: {self.n_heads}')
        self._logger.info(f'   - 注意力类型: {self.att_type}')
        self._logger.info(f'   - 自适应滞后: use_adaptive_lag={self.use_adaptive_lag}, max_lag={self.max_lag}, lag_tau={self.lag_tau}')
        if self.use_ms_decomp:
            self._logger.info(f'   - MS-Decomp: scales={self.ms_decomp_scales}, temp={self.ms_decomp_temp}')
        else:
            self._logger.info(f'   - 时序分解: AvgPool (kernel_size=3)')
        
    def _build_model(self):

        class Args:
            pass
        
        args = Args()
        args.input_length = self.input_window
        args.predict_length = self.output_window
        args.in_dim = self.feature_dim
        args.pre_dim = self.output_dim
        args.num_nodes = self.num_nodes
        args.hid_dim = self.hid_dim
        args.d_out = self.d_out
        args.n_heads = self.n_heads
        args.M = self.M
        args.num_layers = self.num_layers
        args.tau = self.tau
        args.query_cross_time = self.query_cross_time
        args.return_att = self.return_att
        args.norm_flag = self.norm_flag
        args.att_type = self.att_type
        args.activation_data = self.activation_data
        args.activation_enc = self.activation_enc
        args.activation_dec = self.activation_dec
        args.emb_dropout = self.emb_dropout
        args.te_emb_dropout = self.te_emb_dropout
        args.se_emb_dropout = self.se_emb_dropout
        args.tle_emb_dropout = self.tle_emb_dropout
        args.enc_dropout = self.enc_dropout
        args.att_dropout = self.att_dropout
        args.prob_factor = self.prob_factor
        args.slice_size_per_day = self.slice_size_per_day
        args.use_adaptive_lag = self.use_adaptive_lag
        args.max_lag = self.max_lag
        args.lag_tau = self.lag_tau
        args.use_ms_decomp = self.use_ms_decomp
        args.ms_decomp_scales = self.ms_decomp_scales
        args.ms_decomp_temp = self.ms_decomp_temp
        args.ms_decomp_dropout = self.ms_decomp_dropout
        args.ms_decomp_norm = self.ms_decomp_norm
        args.tle_ablate_mlp = self.tle_ablate_mlp
        args.tle_ablate_fft = self.tle_ablate_fft
        args.tle_ablate_moe = self.tle_ablate_moe
        
        self.model = TRIADModel(args)
        
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch, batch_lap_pos_enc=None):
        x = batch['X']
        # Handle both grid (5D) and point (4D) data formats
        if len(x.shape) == 5:
            # Grid data: (B, T, H, W, F) -> (B, T, N, F)
            B, T, H, W, F = x.shape
            N = H * W
            x = x.reshape(B, T, N, F)
        else:
            # Point data: (B, T, N, F)
            B, T, N, F = x.shape
        x_time = self._extract_time_features(batch, B, T)
        x_traffic = x[..., 0:self.input_dim]

        if getattr(self, "_debug_dataflow_once", False):
            self._logger.info("============================================================")
            self._logger.info(" (第1个batch)")
            self._logger.info(f"   X 原始 shape: {x.shape} (B,T,N,F_total)")
            self._logger.info(f"   self.input_dim: {self.input_dim}")
            self._logger.info(f"   self.x_feature_dim: {self.x_feature_dim}")
            self._logger.info(f"   x_traffic shape: {x_traffic.shape} (B,T,N,input_dim)")
            self._logger.info(f"   x_time shape: {x_time.shape} (B,T,2)")
            self._logger.info(
                f"   x_time[:,:,0] (time_in_day) min/max: {int(x_time[:,:,0].min())}/{int(x_time[:,:,0].max())} "
                f"(应在[0,{self.slice_size_per_day-1}])"
            )
            self._logger.info(
                f"   x_time[:,:,1] (day_in_week) min/max: {int(x_time[:,:,1].min())}/{int(x_time[:,:,1].max())} "
                f"(应在[0,6])"
            )
            self._logger.info("============================================================")
            self._debug_dataflow_once = False

        output, _ = self.model(x_traffic, x_time)
        return output
    
    def _extract_time_features(self, batch, B, T):
        x = batch['X']
        N = x.shape[2]

        def _to_bt(tensor, expect_nodes):
            if tensor.dim() == 2:
                return tensor
            if tensor.dim() == 3:
                if tensor.shape[-1] == 1:
                    return tensor.squeeze(-1)
                if tensor.shape[-1] == expect_nodes:
                    return tensor[:, :, 0]
                if tensor.shape[-1] == 7:
                    return tensor.argmax(dim=-1).long()
            if tensor.dim() == 4:
                if tensor.shape[-1] == 7:
                    return tensor[:, :, 0, :].argmax(dim=-1).long()
                if tensor.shape[-1] == 1:
                    return tensor.squeeze(-1)[:, :, 0]
            # Handle 5D grid data: (B, T, H, W, F) -> (B, T)
            if tensor.dim() == 5:
                # Take the first grid cell's value (all cells should have same time)
                result = tensor[:, :, 0, 0]  # 直接不取最后一维
                if result.dim() > 2:
                    result = result.squeeze(-1)
                return result
            out = tensor
            while out.dim() > 2:
                out = out.squeeze(-1)
            return out

        # 检查 batch 中是否有独立的时间特征字段
        # 注意：Batch 类没有 __contains__，需要用 hasattr 或检查 data 字典
        batch_data = batch.data if hasattr(batch, 'data') else batch
        has_time_feats = ('time_in_day_feat' in batch_data and 'day_in_week_feat' in batch_data)
        
        if has_time_feats:
            time_in_day = batch['time_in_day_feat']
            day_in_week = batch['day_in_week_feat']

            time_in_day_bt = _to_bt(time_in_day, N)
            day_in_week_bt = _to_bt(day_in_week, N)

            if time_in_day_bt.dtype.is_floating_point:
                time_idx = torch.floor(time_in_day_bt * self.slice_size_per_day).long()
            else:
                time_idx = time_in_day_bt.long()
            time_idx = torch.clamp(time_idx, 0, self.slice_size_per_day - 1)

            if day_in_week_bt.dtype.is_floating_point:
                day_idx = torch.round(day_in_week_bt).long()
            else:
                day_idx = day_in_week_bt.long()
            day_idx = torch.clamp(day_idx, 0, 6)

            x_time = torch.stack([time_idx.to(x.device), day_idx.to(x.device)], dim=-1)
        else:
            offset = self.x_feature_dim if self.x_feature_dim is not None else self.input_dim
            if x.shape[-1] > offset:
                if x.shape[-1] >= offset + 1:
                    time_feat = x[..., offset]
                    base = time_feat[:, :, 0]
                    # 兼容两种存储方式：[0,1) 比例值 或 [0,287] 整数索引
                    if base.dtype.is_floating_point and base.max() <= 1.0 + 1e-6:
                        time_idx = torch.floor(base * self.slice_size_per_day).long()
                    else:
                        time_idx = base.long()
                else:
                    time_idx = torch.zeros(B, T, dtype=torch.long, device=x.device)

                if x.shape[-1] >= offset + 8:
                    day_onehot = x[..., offset+1:offset+8]
                    day_idx = day_onehot[:, :, 0, :].argmax(dim=-1).long()
                else:
                    day_idx = torch.zeros(B, T, dtype=torch.long, device=x.device)

                time_idx = torch.clamp(time_idx, 0, self.slice_size_per_day - 1)
                day_idx = torch.clamp(day_idx, 0, 6)
                time_idx = torch.clamp(time_idx, 0, self.slice_size_per_day - 1)
                day_idx = torch.clamp(day_idx, 0, 6)

                # Ensure both are (B, T) before stacking
                if time_idx.dim() > 2:
                    time_idx = time_idx[:, :, 0] if time_idx.shape[2] > 1 else time_idx.squeeze(2)
                if day_idx.dim() > 2:
                    day_idx = day_idx[:, :, 0] if day_idx.shape[2] > 1 else day_idx.squeeze(2)

                x_time = torch.stack([time_idx, day_idx], dim=-1)

                x_time = torch.stack([time_idx, day_idx], dim=-1)
            else:
                x_time = torch.zeros(B, T, 2, dtype=torch.long, device=x.device)
        
        return x_time
    
    def get_loss_func(self, set_loss='huber'):
        if set_loss == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=np.nan)
        elif set_loss == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=np.nan)
        elif set_loss == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=np.nan)
        elif set_loss == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=np.nan)
        else:
            self._logger.warning(f'Unrecognized loss function {set_loss}, using huber loss')
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        return lf
    
    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='huber'):
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return lf(y_predicted, y_true)
    
    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.forward(batch)
        # Handle Grid data: reshape (B, T, H, W, F) -> (B, T, N, F)
        if len(y_true.shape) == 5:
            B, T, H, W, F = y_true.shape
            y_true = y_true.reshape(B, T, H * W, F)

        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen, set_loss=self.set_loss)
    
    def predict(self, batch, lap_mx=None):
        return self.forward(batch)
