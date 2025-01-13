import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *
import pdb
from clip_modules.model_loader import load
from model.HSIC import hsic_normalized, HSIC
from model.contrastive_loss import _CONTRASTIVE_LOSS


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        # down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
    
def init_tfts(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta

def apply_tfts(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y

def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp


def compose(phase, amp):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
    x = x / math.sqrt(x.shape[2] * x.shape[3])
    x = torch.view_as_complex(x)
    return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')


class AdapterAO(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
         
        self.adapter_scalar = adapter_scalar

        if adapter_scalar == "learnable_scalar":
            # self.scale = nn.Parameter(torch.ones(1))
            self.scale_att = nn.Linear(self.n_embd, 1)
            self.scale_obj = nn.Linear(self.n_embd, 1)
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.scale_att.weight, a=math.sqrt(5))
                nn.init.zeros_(self.scale_att.bias)
                nn.init.kaiming_uniform_(self.scale_obj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.scale_obj.bias)
        else:
            self.scale_att = float(adapter_scalar)
            self.scale_obj = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_att = nn.Linear(self.down_size, self.n_embd)

        
        self.dropout = dropout
        self.init_option = init_option

        ##
        self.up_proj_obj = nn.Linear(self.down_size, self.n_embd)
        self.tfts_gamma_att, self.tfts_beta_att = init_tfts(1024)
        self.tfts_gamma_obj, self.tfts_beta_obj = init_tfts(1024)
        self.non_linear_func_2 = nn.GELU()
        
        self.proj_out_att = nn.Identity()#nn.Linear(self.n_embd,768)
        self.proj_out_obj = nn.Identity()#nn.Linear(self.n_embd,768)
         
        ################
        ##
        self.weight = nn.Parameter(torch.zeros(2))
        self.T = 0.1
        ##
        ################
        self.dap_downsample_att = nn.Linear(257, 1)
        self.dap_downsample_obj = nn.Linear(257, 1)
        nn.init.zeros_(self.dap_downsample_att.weight)
        nn.init.zeros_(self.dap_downsample_att.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm_att = LayerNorm(self.n_embd, eps=1e-6)
        self.dap_norm_obj = LayerNorm(self.n_embd, eps=1e-6)
        #################
        self._reset_parameters()


    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj_att.weight)
                nn.init.zeros_(self.up_proj_att.bias)

                nn.init.zeros_(self.up_proj_obj.weight)
                nn.init.zeros_(self.up_proj_obj.bias)
                

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        # down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up_att = self.up_proj_att(down)

        up_obj = self.up_proj_obj(down)
        
        if self.adapter_scalar == "learnable_scalar":
            scale_att = torch.sigmoid(self.scale_att(x))
            scale_obj = torch.sigmoid(self.scale_obj(x))
            up = scale_att * up_att + scale_obj * up_obj
        else:
            up = up_att * self.scale_att + up_obj * self.scale_obj
             

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        # ##################
        
        # weight = F.softmax(self.weight / self.T, dim=0) # Weights of the original and adapted features
        # HW1, B, C = residual.shape
        # HW = int(np.sqrt(HW1-1))
        # if HW1 != 257:
        #     x_hw = residual[3:,:,:].reshape(HW,HW,B,C).permute(2,3,0,1)
        #     x_hw_adapt_att = up_att[3:,:,:].reshape(HW,HW,B,C).permute(2,3,0,1)
        #     x_hw_adapt_obj = up_obj[3:,:,:].reshape(HW,HW,B,C).permute(2,3,0,1)
        # else:
        #     x_hw = residual[1:,:,:].reshape(HW,HW,B,C).permute(2,3,0,1)
        #     x_hw_adapt_att = up_att[1:,:,:].reshape(HW,HW,B,C).permute(2,3,0,1)
        #     x_hw_adapt_obj = up_obj[1:,:,:].reshape(HW,HW,B,C).permute(2,3,0,1)
        # # x_adapt_hw = 
        # phase_ori, amp_ori = decompose(x_hw, 'all') # Decompose the original feature into phase and amplitude
         
       
        # phase_adapt_att, amp_adapt_att = decompose(x_hw_adapt_att.to(torch.complex64), 'all') # Decompose the adapted feature into phase and amplitude
        # phase_adapt_obj, amp_adapt_obj = decompose(x_hw_adapt_obj.to(torch.complex64), 'all')
        
        # amp = amp_adapt_att * weight[0] + amp_ori * weight[1] # Combine the amplitude of the original and adapted features
        # phase = phase_adapt_obj * weight[0] + phase_ori * weight[1] 
        
        # x_att = compose(phase_ori, amp) # Combine the phase and amplitude to get the final feature
        # x_obj = compose(phase, amp_ori)

        # x_att_reshape = x_att.reshape(B,C,-1).permute(2,0,1).half()
        # x_obj_reshape = x_obj.reshape(B,C,-1).permute(2,0,1).half()
        ##################
        # adapter_prompt_att = self.non_linear_func_2(up_att).mean(dim=0, keepdim=True)
        # adapter_prompt_att = apply_tfts(adapter_prompt_att, self.tfts_gamma_att, self.tfts_beta_att)

        # adapter_prompt_obj = self.non_linear_func_2(up_obj).mean(dim=0, keepdim=True)
        # adapter_prompt_obj = apply_tfts(adapter_prompt_obj, self.tfts_gamma_obj, self.tfts_beta_obj)


   
        ###################
        ###########to generate prompt
        if up_att.shape[0] == 257: 
            adapter_trans_att = self.dap_norm_att(up_att).permute(1,2,0)
            adapter_trans_obj = self.dap_norm_obj(up_obj).permute(1,2,0)
            
        else:
            up_att = torch.cat((up_att[:1,:,:],up_att[3:,:,:]),dim=0)
            adapter_trans_att = self.dap_norm_att(up_att).permute(1,2,0)

            up_obj = torch.cat((up_obj[:1,:,:],up_obj[3:,:,:]),dim=0)
            adapter_trans_obj = self.dap_norm_obj(up_obj).permute(1,2,0)
            # adapter_prompt = self.dap_downsample(adapter_trans)
            # adapter_prompt  = adapter_prompt.permute(2,0,1)

        adapter_prompt = self.dap_downsample_att(adapter_trans_att)
        adapter_prompt_att  = adapter_prompt.permute(2,0,1)
        adapter_prompt = self.dap_downsample_obj(adapter_trans_obj)
        adapter_prompt_obj  = adapter_prompt.permute(2,0,1)

        adapter_prompt = torch.cat([adapter_prompt_att,adapter_prompt_obj],dim=0)

        return output, adapter_prompt, self.proj_out_att(adapter_prompt_att.squeeze()), self.proj_out_obj(adapter_prompt_obj.squeeze())
    

class Adapter_Generator(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option
          
        self._reset_parameters()

        ######## generate prompt
        self.non_linear_func_generator = nn.GELU()
        self.tfts_gamma, self.tfts_beta = init_tfts(d_model)
        # self.tfts_gamma, self.tfts_betat = init_tfts(1024)
        # self.non_linear_func_2 = nn.GELU()
        # self.dap_downsample = nn.Linear(257, 1)
        # nn.init.zeros_(self.dap_downsample.weight)
        # nn.init.zeros_(self.dap_downsample.bias)
        # # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        # self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
        ###################

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        # down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up #[257,16,1024]
        

        ###########to generate prompt
        # if up.shape[0] == 257:
            
        #     adapter_trans = self.dap_norm(up).permute(1,2,0)
        #     adapter_prompt = self.dap_downsample(adapter_trans)
        #     adapter_prompt  = adapter_prompt.permute(2,0,1)
        # else:
        #     up = torch.cat((up[:1,:,:],up[2:,:,:]),dim=0)
        #     adapter_trans = self.dap_norm(up).permute(1,2,0)
        #     adapter_prompt = self.dap_downsample(adapter_trans)
        #     adapter_prompt  = adapter_prompt.permute(2,0,1)

        adapter_prompt = self.non_linear_func(up).mean(dim=0)
        adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
        ###########
        return output, adapter_prompt

class Prompt_Generator(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,d_model=None):
        super().__init__()
        self.n_embd = d_model

        ######## generate prompt
        
        self.dap_downsample = nn.Linear(257, 1)
        nn.init.zeros_(self.dap_downsample.weight)
        nn.init.zeros_(self.dap_downsample.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
        ###################

    def forward(self, x):
        ###########to generate prompt
        if x.shape[0] == 257:
            
            adapter_trans = self.dap_norm(x).permute(1,2,0)
            adapter_prompt = self.dap_downsample(adapter_trans)
            adapter_prompt  = adapter_prompt.permute(2,0,1)
        else:
            x = torch.cat((x[:1,:,:],x[2:,:,:]),dim=0)
            adapter_trans = self.dap_norm(x).permute(1,2,0)
            adapter_prompt = self.dap_downsample(adapter_trans)
            adapter_prompt  = adapter_prompt.permute(2,0,1)

        # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
        # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
        ###########
        return adapter_prompt


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x

class Disentangler2(nn.Module):
    def __init__(self, emb_dim_in, emb_dim_out):
        super(Disentangler2, self).__init__()
        self.fc1 = nn.Linear(emb_dim_in, emb_dim_out)
        self.bn1_fc = nn.BatchNorm1d(emb_dim_out)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x
    
class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q

class QueryFormer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,):
        super(QueryFormer, self).__init__()
        # self.fc1 = nn.Linear(emb_dim, emb_dim)
        # self.bn1_fc = nn.BatchNorm1d(emb_dim)
        self.sa = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.ca = CrossAttentionLayer(d_model, nhead, dropout)

    def forward(self, query, v_feat):
        query = self.sa(query,query,query)
        query_mutual = self.ca(query,v_feat)
        # x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = F.dropout(x, training=self.training)
        return query_mutual


class DisFormer(nn.Module):
    def __init__(self,d_model,nhead, dropout=0.1,):
        super().__init__()
        self.sa = MulitHeadAttention(d_model, nhead, dropout)
        self.sa_a = MulitHeadAttention(d_model, nhead, dropout)
        self.sa_o = MulitHeadAttention(d_model, nhead, dropout)
        # self.ca = MulitHeadAttention(d_model, nhead, dropout)

        # self.att_sa = MulitHeadAttention(d_model, nhead, dropout)
        # self.obj_sa = MulitHeadAttention(d_model, nhead, dropout)
        # self.com_sa = MulitHeadAttention(d_model, nhead, dropout)
        self.att_ca = CrossAttentionLayer(d_model, nhead, dropout)
        self.obj_ca = CrossAttentionLayer(d_model, nhead, dropout)
        self.com_ca = CrossAttentionLayer(d_model, nhead, dropout)
        

        # self.att_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.obj_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.com_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        
        # query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        
        self.dropout = nn.Dropout(dropout)

        # self.mlp_att = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     QuickGELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        # self.mlp_obj = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     QuickGELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        # self.mlp_com = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     QuickGELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        

          
        self.norm = nn.LayerNorm(d_model)

        self.norm_ca = nn.LayerNorm(d_model)

        self.norm_att = nn.LayerNorm(d_model)
        self.norm_obj = nn.LayerNorm(d_model)
        self.norm_com = nn.LayerNorm(d_model)

        self.norm_att1 = nn.LayerNorm(d_model)
        self.norm_obj1 = nn.LayerNorm(d_model)
        self.norm_com1 = nn.LayerNorm(d_model)
        
        self.att_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, d_model),
        )
        self.obj_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, d_model),
        )
        self.com_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, d_model),
        )

    def forward(self, batch_patch, att_query, obj_query,att_txt,obj_txt,com_txt):
        
        b = batch_patch.shape[0]
        
        if att_query.shape[0] ==1:
            att_query = att_query.repeat(b,1,1)
            obj_query = obj_query.repeat(b,1,1)
        com_query = torch.cat((att_query,obj_query),dim=1)#self.com_query.repeat(b,1,1)#torch.cat((att_query,obj_query),dim=1)#self.com_query.repeat(b,1,1)

        ############
        # att_query = self.norm_att1(self.att_ca(att_query,self.att_proj(att_txt)))
        # obj_query = self.norm_obj1(self.obj_ca(obj_query,self.obj_proj(obj_txt)))
        # com_query = self.norm_com1(self.com_ca(com_query,self.com_proj(com_txt)))
        att_query = torch.cat((self.att_proj(att_txt),att_query),dim=1)
        obj_query = torch.cat((self.obj_proj(obj_txt),obj_query),dim=1)
        com_query = torch.cat((self.com_proj(com_txt),com_query),dim=1)
        
        ############
        
        com_query = self.norm(com_query + self.sa(com_query,com_query,com_query))
        att_query = self.norm_att1(att_query + self.sa_a(att_query,att_query,att_query))
        obj_query = self.norm_obj1(obj_query + self.sa_o(obj_query,obj_query,obj_query))
        
        att_feat = self.norm_att(self.att_ca(att_query,batch_patch))
        obj_feat = self.norm_obj(self.att_ca(obj_query,batch_patch))
        com_feat = self.norm_com(self.att_ca(com_query,batch_patch))

        # att_feat = self.norm(att_query + self.ca(att_query, batch_patch, batch_patch))
        # obj_feat = self.norm(obj_query + self.ca(obj_query, batch_patch, batch_patch))
        # com_feat = self.norm(com_query + self.ca(com_query, batch_patch, batch_patch))

        # att_feat = self.norm_att(att_feat + self.dropout(self.mlp_att(att_feat)))
        # obj_feat = self.norm_obj(obj_feat + self.dropout(self.mlp_obj(obj_feat)))
        # com_feat = self.norm_com(com_feat + self.dropout(self.mlp_com(com_feat)))
        
        return com_feat, att_feat, obj_feat
        # batch_img_features = [com_feat.mean(dim=1),att_feat.mean(dim=1),obj_feat.mean(dim=1)]
        # normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        # return normalized_img_features
        # logits = list()
        # for i_element in range(normalized_img_features.shape[0]):    
        #     logits.append(torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*all_text_feat[i_element].T))

class DisAOFormer(nn.Module):
    def __init__(self,d_model,nhead, dropout=0.1,):
        super().__init__()
        self.sa_a = MulitHeadAttention(d_model, nhead, dropout)
        self.sa_o = MulitHeadAttention(d_model, nhead, dropout)
        # self.ca = MulitHeadAttention(d_model, nhead, dropout)

        # self.att_sa = MulitHeadAttention(d_model, nhead, dropout)
        # self.obj_sa = MulitHeadAttention(d_model, nhead, dropout)
        # self.com_sa = MulitHeadAttention(d_model, nhead, dropout)
        self.att_ca = CrossAttentionLayer(d_model, nhead, dropout)
        self.obj_ca = CrossAttentionLayer(d_model, nhead, dropout)
       
        # self.att_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.obj_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.com_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        
        # query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        
        self.dropout = nn.Dropout(dropout)

        # self.mlp_att = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     QuickGELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        # self.mlp_obj = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     QuickGELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        # self.mlp_com = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     QuickGELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model * 4, d_model)
        # )
        

        self.norm_att = nn.LayerNorm(d_model)
        self.norm_obj = nn.LayerNorm(d_model)
       

        self.norm_att1 = nn.LayerNorm(d_model)
        self.norm_obj1 = nn.LayerNorm(d_model)
      
        
        self.att_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, d_model),
        )
        self.obj_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, d_model),
        )

       
        

    def forward(self, batch_patch, att_query, obj_query,att_txt,obj_txt,com_txt):
        
        b = batch_patch.shape[0]
        
        if att_query.shape[0] ==1:
            att_query = att_query.repeat(b,1,1)
            obj_query = obj_query.repeat(b,1,1)  

        ############
        # att_query = self.norm_att1(self.att_ca(att_query,self.att_proj(att_txt)))
        # obj_query = self.norm_obj1(self.obj_ca(obj_query,self.obj_proj(obj_txt)))
        # com_query = self.norm_com1(self.com_ca(com_query,self.com_proj(com_txt)))
        att_query = torch.cat((self.att_proj(att_txt),att_query),dim=1)
        obj_query = torch.cat((self.obj_proj(obj_txt),obj_query),dim=1)
        
        
        ############
        att_query = self.norm_att1(att_query + self.sa_a(att_query,att_query,att_query))
        obj_query = self.norm_obj1(obj_query + self.sa_a(obj_query,obj_query,obj_query))
        
        att_feat = self.norm_att(self.att_ca(att_query,batch_patch))
        obj_feat = self.norm_obj(self.obj_ca(obj_query,batch_patch))
        
        return att_feat, obj_feat   

class Troika_DPQD(nn.Module):    

    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_model, context_length=config.context_length,download_root=config.clip_arch)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.cross_attn_dropout = config.cross_attn_dropout if hasattr(config, 'cross_attn_dropout') else 0.1
        self.prim_loss_weight = config.prim_loss_weight if hasattr(config, 'prim_loss_weight') else 1

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # pdb.set_trace()
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        output_dim = self.clip.visual.output_dim

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        # self.attr_disentangler = Disentangler(output_dim)
        # self.obj_disentangler = Disentangler(output_dim)

        self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout) for _ in range(config.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)
        
        ###############
        self.additional_visual_params = self.add_visual_tunable_params()
        self.additional_text_params = self.add_text_tunable_params()
        # self.additional_prompt_generator = self.add_prompt_generator()
        dim = self.clip.visual.transformer.width
        # self.DisFormer = DisFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        self.DisAOFormer = DisAOFormer(dim, dim//64, self.cross_attn_dropout)

        self.attr_final_proj = Disentangler2(dim,output_dim)
        self.obj_final_proj = Disentangler2(dim,output_dim)
        self.com_final_proj = Disentangler2(dim,output_dim)

        self.att_multi_proj = Disentangler2(dim*4,output_dim)
        self.obj_multi_proj = Disentangler2(dim*4,output_dim)
        
        # self.att_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.obj_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.com_former = QueryFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        
        # # query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    
        self.query_num = 3
        self.att_query_cls = nn.Parameter(torch.randn(len(self.attributes), dim)).cuda()
        self.obj_query_cls = nn.Parameter(torch.randn(len(self.classes), dim)).cuda()
        # self.com_query = nn.Parameter(torch.randn(1, self.query_num, output_dim))
        self.att_query_cls.data.normal_(mean=0.0, std=0.05)
        self.obj_query_cls.data.normal_(mean=0.0, std=0.05)
        # self.com_query.data.normal_(mean=0.0, std=0.05)
        # self.TextDisFormer = VTFormer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.visual2text = nn.Linear(dim,output_dim)
        
        # self.pri2com = nn.Sequential(
        #     nn.Linear(output_dim*2, output_dim),
        #     nn.BatchNorm1d(output_dim),
        # )
        self.contrastive_loss=_CONTRASTIVE_LOSS["SoftNearestNeighborsLoss"](
                temperature=1.0
            )

    def Sinkhorn(self, K, u, v, max_iter=10):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2

        for i in range(max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def add_visual_tunable_params(self):
        adapter_num = self.clip.visual.transformer.layers
        params = nn.ModuleList([AdapterAO(d_model=self.clip.visual.transformer.width, 
                                    bottleneck=self.config.adapter_dim, 
                                    dropout=self.config.adapter_dropout,
                                    adapter_scalar = "learnable_scalar"
                                ) for _ in range(adapter_num)])
        return params

    def add_text_tunable_params(self):
        adapter_num = self.clip.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.transformer.width, 
                                    bottleneck=self.config.adapter_dim, 
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params
    

    def zero_shot_visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)
        x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x[:, 0, :], x
    
    def zero_shot_text(self, pairs):
        """Function to get the clip representations.

        Args:
            model (nn.Module): the clip model
            test_dataset (CompositionDataset): the test/validation dataset
            config (argparse.ArgumentParser): config/args
            device (str): device type cpu/cuda:0

        Returns:
            torch.Tensor: returns the tensor with the attribute-object
                representations with clip model.
        """
        pairs = [(attr.replace(".", " ").lower(),
                obj.replace(".", " ").lower())
                for attr, obj in pairs]

        prompts_pairs = [f"a photo of {attr} {obj}" for attr, obj in pairs]
        tokenized_pairs = self.tokenizer(prompts_pairs, context_length=self.config.context_length).cuda()

        prompts_attrs = [f"a photo of {attr}" for attr in self.attributes]
        tokenized_attrs = self.tokenizer(prompts_attrs, context_length=self.config.context_length).cuda()

        prompts_objs = [f"a photo of {obj}" for obj in self.classes]
        tokenized_objs = self.tokenizer(prompts_objs, context_length=self.config.context_length).cuda()
        # test_batch_tokens = np.array_split(
        #     tokenized_prompts,
        #     len(tokenized_prompts) //
        #     config.text_encoder_batch_size)
        # rep = torch.Tensor().to(device).type(model.dtype)
        with torch.no_grad():
            att_feat,_ = self.text_encoder(tokenized_attrs, None, enable_pos_emb=True)
            att_feat = att_feat/att_feat.norm(dim=-1,keepdim=True)

            pair_feat,_ = self.text_encoder(tokenized_pairs,None,  enable_pos_emb=True)
            pair_feat = pair_feat/pair_feat.norm(dim=-1,keepdim=True)

            obj_feat,_ = self.text_encoder(tokenized_objs,None,  enable_pos_emb=True)
            obj_feat = obj_feat/obj_feat.norm(dim=-1,keepdim=True)
            # for tokenized_ele in tokenized_all:
            #     batch_tokens = batch_tokens.to(device)
            #     _text_features = model.text_encoder(
            #         batch_tokens, enable_pos_emb=True)
            #     text_features = _text_features / _text_features.norm(
            #         dim=-1, keepdim=True
            #     )
            #     rep = torch.cat((rep, text_features), dim=0)
        return pair_feat, att_feat, obj_feat
        # return [tokenized_pairs,tokenized_attrs,tokenized_objs]

    
    def encode_text_with_adapter(self, token_ids, token_tensors, enable_pos_emb):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.clip.token_embedding(token_ids)

        text_features = text_features.type(self.dtype)
        x = (
            text_features + self.clip.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        # text_feature = self.transformer(x)
        for i_block in range(self.clip.transformer.layers):
            # MHA
            # adapt_x = self.additional_text_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.transformer.resblocks[i_block].attention(
                self.clip.transformer.resblocks[i_block].ln_1(x)
            )
            # x = x + adapt_x + residual
            x = x + residual

            # FFN
            i_adapter = i_block #+ self.clip.transformer.layers
            adapt_x = self.additional_text_params[i_adapter](x.type(torch.float), add_residual=False).type(self.dtype)
            # adapt_x, adapt_prompt = self.additional_text_params[i_adapter](x.type(torch.float), add_residual=False)
            residual = x
            x = self.clip.transformer.resblocks[i_block].mlp(
                self.clip.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

           
            #x[1,:,:] = adapt_prompt

           
        text_feature = x
        x = text_feature.permute(1, 0, 2)
        x = self.clip.ln_final(x)
        tf = (
            x[
                torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
            ]  # POS of <EOS>
            @ self.clip.text_projection
        )
        return tf, text_feature
    
    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        ###########
        # BatchSize = x.shape[0]
        # x = torch.cat((
        #         x[:, :1, :],
        #         self.prompt_dropout(self.prompt_embeddings.expand(x.shape[0], -1, -1)),
        #         x[:, 1:, :]
        #     ), dim=1)
        #############
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        # att_token_pool = []
        # obj_token_pool = []
        for i_block in range(self.clip.visual.transformer.layers):
            
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            # x = x + adapt_x + residual
            x = x + residual

            # FFN
            i_adapter = i_block #+ self.clip.visual.transformer.layers
            # adapt_x = self.additional_visual_params[i_adapter](x.type(torch.float), add_residual=False).type(self.dtype)
            # adapt_x, adapter_prompt = self.additional_visual_params[i_adapter](x, add_residual=False)
            adapt_x, adapter_prompt, att_token, obj_token = self.additional_visual_params[i_adapter](x.type(torch.float), add_residual=False)
            # if i_block>5 and i_block<18:
            #     adapt_x, adapter_prompt, att_token, obj_token = self.additional_visual_params[i_adapter](x.type(torch.float), add_residual=False)
            
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            
            # if i_block>5 and i_block<18:
            #     x = x + adapt_x + residual
            # else:
            #     x = x + residual
           
            
            # x = x + residual
            if i_block==22:
                x = torch.cat([x[0:1, :, :], adapter_prompt, x[1:, :, :]], dim=0)
            # elif i_block==self.clip.visual.transformer.layers-1:
            #     x = x
            # elif i_block<23:
            #     x = torch.cat([x[0:1, :, :], adapter_prompt, x[3:, :, :]], dim=0)
            # elif i_block>12 and i_block<23:
            #     x = torch.cat([x[0:1, :, :], adapter_prompt, x[3:, :, :]], dim=0)
            else:
                # x = torch.cat([x[0:1, :, :], adapter_prompt, x[3:, :, :]], dim=0)
                x = x
                
            
            # att_token_pool.append(att_token)
            # obj_token_pool.append(obj_token)
        
        img_feature_ori = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature_ori)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        
        # all_att_token = torch.stack(att_token_pool,dim=0).permute(1,0,2)
        # all_obj_token = torch.stack(obj_token_pool,dim=0).permute(1,0,2)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            pdb.set_trace()

        return img_feature[:, 0, :], img_feature_ori, img_feature, None #all_obj_token


    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)
    
    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)
    
    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)
        
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor[0][
            :, 1 : len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1 : len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor
    
    
    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        comp_logits, attr_logits, obj_logits = predict
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss = loss_comp * self.config.pair_loss_weight +\
               loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight
        return loss


    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = comp_logits[:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
        return comp_logits

    
    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        contxt_feat = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, text_batch = self.encode_text_with_adapter(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )
        
            if i_element == 0:
                com_txt = text_batch[0:1,1:4,:]
                contxt_feat.append(com_txt)
            elif i_element == 1:
                att_txt = text_batch[0:1,1:4,:]
                contxt_feat.append(att_txt)
            else: 
                obj_txt = text_batch[0:1,1:4,:]
                contxt_feat.append(obj_txt)

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features, contxt_feat

    
    def forward_for_open(self, batch, text_feats, context_feats):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, batch_patch,_, patch_img_ori = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))

        
        com_txt = context_feats[0].repeat(b,1,1)
        att_txt = context_feats[1].repeat(b,1,1)
        obj_txt = context_feats[2].repeat(b,1,1)

        batch_patch_token = torch.cat((batch_patch[:,0:1,:],batch_patch[:,3:,:]),dim=1)
        att_query = batch_patch[:,1:2,:]  #all_att_token[:,23:,:]#batch_patch[:,1:2,:]
        obj_query = batch_patch[:,2:3,:] #all_obj_token[:,20:,:]#batch_patch[:,2:3,:]

       
        com_feat, att_feat, obj_feat = self.DisFormer(batch_patch_token,att_query,obj_query,att_txt,obj_txt,com_txt)
        # com_feat, att_feat, obj_feat = self.DisAOFormer(batch_patch_token,att_query,obj_query,att_txt,obj_txt,com_txt)
        # com_feat = com_feat.mean(dim=1)
        
        com_feat = self.com_final_proj(com_feat.mean(dim=1))
        att_feat = self.attr_final_proj(att_feat.mean(dim=1))
        obj_feat = self.obj_final_proj(obj_feat.mean(dim=1))
        # batch_img_features = [batch_img,att_feat.mean(dim=1),obj_feat.mean(dim=1)]
        # batch_img_features = [batch_img+com_feat.mean(dim=1),att_feat.mean(dim=1),obj_feat.mean(dim=1)]
        batch_img_features = [batch_img,att_feat,obj_feat]

        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]
        
        # batch_img_features = [batch_img, self.attr_disentangler(batch_img.type(torch.float)).type(self.dtype), self.obj_disentangler(batch_img.type(torch.float)).type(self.dtype)]
        # normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]
 
        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]

            # CMT
            # cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
            # batch_patch = self.patch_norm(batch_patch)
            # for layer in self.cmt:
            #     cmt_text_features = layer(cmt_text_features, batch_patch)
            # cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

            # cmt_text_features = cmt_text_features / cmt_text_features.norm(
            #     dim=-1, keepdim=True
            # )
            # idx_text_features_norm = idx_text_features / idx_text_features.norm(dim=-1,keepdim=True)
            logits.append(torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*idx_text_features.T))
            # logits.append(
            #     torch.einsum(
            #         "bd, bkd->bk", 
            #         normalized_img_features[i_element], 
            #         idx_text_features * self.clip.logit_scale.exp()
            # ))
        return logits

    
    def fine_align(self, img_features, txt_features):
        
        image_features = F.normalize(img_features[:,3:,:],dim=-1)
        text_features = F.normalize(txt_features,dim=-1)
        b,M,C = image_features.shape
        N,n_cls,C = txt_features.shape

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M,N,-1)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim
        xx=torch.zeros(b*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)

        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,n_cls)
        
        logit_scale = self.clip.logit_scale.exp()
        # logits = logit_scale * image_feature_pool @ text_feature_pool.t()
        logits2 = logit_scale * sim_op
        
        # logits2 = (logits2 + logits)

        return logits2
    
    def forward(self, batch, idx, training=False, pairs=None):
        batch_img = batch[0].cuda()
        # batch_img = batch
        labels = [batch[3].cuda(),batch[1].cuda(),batch[2].cuda()]
        b = batch_img.shape[0]
        l, _ = idx.shape

        ####### visual branch ##########
        # batch_img, batch_patch = self.zero_shot_visual(batch_img.type(self.clip.dtype))
        batch_img, batch_patch, patch_img_ori,_ = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))
        # batch_img, batch_patch = self.visual(batch_img.type(self.clip.dtype)) #batch_img.shape = [bs,768]; batch_patch.shape = [16,257,768]
        # pdb.set_trace()
        ####### text branch ############
        token_tensors = self.construct_token_tensors(idx)
        all_text_feat = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, text_batch = self.encode_text_with_adapter(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )

            all_text_feat.append(idx_text_features)
            if i_element == 0:
                com_txt = text_batch.permute(1,0,2)[0:1,1:4,:].repeat(b,1,1)
            elif i_element == 1:
                att_text_feat = _text_features
                att_txt = text_batch.permute(1,0,2)[0:1,1:4,:].repeat(b,1,1)
                
                att_txt_patch =text_batch[1:5,:,:]
            else:
                obj_text_feat = _text_features
                obj_txt = text_batch.permute(1,0,2)[0:1,1:4,:].repeat(b,1,1)

                obj_txt_patch =text_batch[1:5,:,:]

        att_logits_fine = self.fine_align(patch_img_ori,att_txt_patch)
        obj_logits_fine = self.fine_align(patch_img_ori,obj_txt_patch)
        ####### V-T interaction ############ 
        batch_patch_token = torch.cat((batch_patch[:,0:1,:],batch_patch[:,3:,:]),dim=1)
        att_query = batch_patch[:,1:2,:]#all_att_token[:,23:,:]#batch_patch[:,1:2,:]
        obj_query = batch_patch[:,2:3,:]#all_obj_token[:,20:,:]#batch_patch[:,2:3,:]
         
        
        loss_kd = 0.0
        
        att_logits = torch.matmul(F.normalize(att_query.squeeze(),dim=-1), F.normalize(self.att_query_cls,dim=-1).T)
        obj_logits = torch.matmul(F.normalize(obj_query.squeeze(),dim=-1), F.normalize(self.obj_query_cls,dim=-1).T)
        
        loss_fn = CrossEntropyLoss()
        loss_kd = 0.1*loss_fn(att_logits, batch[1].cuda()) + 0.1*loss_fn(obj_logits, batch[2].cuda())
        # loss_kd = 0.1*self.contrastive_loss(F.normalize(att_query.squeeze(),dim=-1), batch[1]) + 0.1*self.contrastive_loss(F.normalize(obj_query.squeeze(),dim=-1), batch[2])
        
        # if b==16:
        #     att_query = torch.cat([self.att_query_cls[batch[1]].unsqueeze(1),att_query],dim=1)
        #     obj_query = torch.cat([self.obj_query_cls[batch[2]].unsqueeze(1),obj_query],dim=1) 
        # else:
        vv, iin = torch.topk(att_logits,3,dim=1)
        att_query = torch.cat([self.att_query_cls[iin],att_query],dim=1)
        vv, iin = torch.topk(obj_logits,3,dim=1)
        obj_query = torch.cat([self.obj_query_cls[iin],obj_query],dim=1)

        att_feat, obj_feat = self.DisAOFormer(batch_patch_token,att_query,obj_query,att_txt,obj_txt,com_txt)
        # com_feat, att_feat, obj_feat = self.DisFormer(batch_patch_token,att_query,obj_query,att_txt,obj_txt,com_txt)
        # com_feat = com_feat.mean(dim=1)
        
        # com_feat = self.com_final_proj(com_feat.mean(dim=1))
        att_feat = self.attr_final_proj(att_feat.mean(dim=1))
        obj_feat = self.obj_final_proj(obj_feat.mean(dim=1))
        # batch_img_features = [batch_img,att_feat.mean(dim=1),obj_feat.mean(dim=1)]
        # batch_img_features = [batch_img+com_feat.mean(dim=1),att_feat.mean(dim=1),obj_feat.mean(dim=1)]
        batch_img_features = [batch_img,att_feat,obj_feat]

        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        ######## compute cosine similarity ###########
        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            # CMT
            # idx_text_features = all_text_feat[i_element]
            # cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
            
            # batch_patch_prj = self.patch_norm(batch_patch_prj)
            # for layer in self.cmt:
            #     cmt_text_features = layer(cmt_text_features, batch_patch_prj)
            # cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

            # cmt_text_features = cmt_text_features / cmt_text_features.norm(
            #     dim=-1, keepdim=True
            # )

            # logits.append(
            #     torch.einsum(
            #         "bd, bkd->bk", 
            #         normalized_img_features[i_element], 
            #         cmt_text_features * self.clip.logit_scale.exp()
            # ))
            if i_element == 1:
                att_logits_ = torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*all_text_feat[i_element].T)
                
                logits.append(att_logits_fine + att_logits_)

            if i_element == 2:
                obj_logits_ = torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*all_text_feat[i_element].T)

                logits.append(obj_logits_fine + obj_logits_)
            
            if i_element == 0:

                logits.append(torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*all_text_feat[i_element].T))
             
        
        # loss_kd = 0.0
        # loss_kd = 0.01*F.mse_loss(com_feat/ (1e-5+com_feat.norm(dim=-1, keepdim=True)), normalized_img_features[0],reduction="mean")
        
        if training and pairs is not None:
            zero_shot_image, _ = self.zero_shot_visual(batch[0].cuda().type(self.clip.dtype))
            zs_image = zero_shot_image/zero_shot_image.norm(dim=-1, keepdim=True)
            zs_pair_feat, zs_att_feat, zs_obj_feat = self.zero_shot_text(pairs)
            loss_image = F.l1_loss(normalized_img_features[0], zs_image.cuda(),reduction='mean') * 10
            loss_pair = 25 * F.l1_loss(all_text_feat[0],zs_pair_feat.cuda(),reduction='mean')
            loss_att = 25 * F.l1_loss(all_text_feat[1],zs_att_feat.cuda(),reduction='mean')
            loss_obj = 25 * F.l1_loss(all_text_feat[2],zs_obj_feat.cuda(),reduction='mean')
           
            return logits, loss_image+loss_pair+loss_att+loss_obj
        
        elif training:
            return logits, loss_kd#+loss_kd_ #loss_kd_att + loss_kd_obj
        
        else:
            return logits