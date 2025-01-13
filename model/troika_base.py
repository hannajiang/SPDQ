# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# from functools import reduce
# from operator import mul
# from copy import deepcopy
# from torch.nn.modules.utils import _pair
# from torch.nn.modules.loss import CrossEntropyLoss
# from clip_modules.clip_model import load_clip, QuickGELU
# from clip_modules.tokenization_clip import SimpleTokenizer
# from model.common import *
# import pdb
# from clip_modules.model_loader import load

# class Adapter(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="lora",
#                  adapter_scalar="0.1",
#                  adapter_layernorm_option="none"):
#         super().__init__()
#         self.n_embd = d_model
#         self.down_size = bottleneck

#         #_before
#         self.adapter_layernorm_option = adapter_layernorm_option

#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

#         if adapter_scalar == "learnable_scalar":
#             self.scale = nn.Parameter(torch.ones(1))
#         else:
#             self.scale = float(adapter_scalar)

#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj = nn.Linear(self.down_size, self.n_embd)

#         self.dropout = dropout
#         self.init_option = init_option

#         self._reset_parameters()

#     def _reset_parameters(self):
#         if self.init_option == "bert":
#             raise NotImplementedError
#         elif self.init_option == "lora":
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.up_proj.weight)
#                 nn.init.zeros_(self.down_proj.bias)
#                 nn.init.zeros_(self.up_proj.bias)

#     def forward(self, x, add_residual=True, residual=None):
#         residual = x if residual is None else residual
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

#         down = self.down_proj(x)
#         down = self.non_linear_func(down)
#         down = nn.functional.dropout(down, p=self.dropout, training=self.training)
#         up = self.up_proj(down)

#         up = up * self.scale

#         if self.adapter_layernorm_option == 'out':
#             up = self.adapter_layer_norm_before(up)

#         if add_residual:
#             output = up + residual
#         else:
#             output = up

#         return output
    


# def init_tfts(dim):
#     gamma = nn.Parameter(torch.ones(dim))
#     beta = nn.Parameter(torch.zeros(dim))
#     nn.init.normal_(gamma, mean=1, std=.02)
#     nn.init.normal_(beta, std=.02)
#     return gamma, beta

# def apply_tfts(x, gamma, beta):
#     assert gamma.shape == beta.shape
#     if x.shape[-1] == gamma.shape[0]:
#         return x * gamma + beta
#     elif x.shape[1] == gamma.shape[0]:
#         return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
#     else:
#         raise ValueError('the input tensor shape does not match the shape of the scale factor.')

# class AdapterConcept(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="lora",
#                  adapter_scalar="0.1",
#                  adapter_layernorm_option="none"):
#         super().__init__()
#         self.n_embd = d_model
#         self.down_size = bottleneck

#         #_before
#         self.adapter_layernorm_option = adapter_layernorm_option

#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
         
#         self.adapter_scalar = adapter_scalar
#         # if adapter_scalar == "learnable_scalar":
#         #     # self.scale = nn.Parameter(torch.ones(1))
#         #     self.scale_att = nn.Linear(self.n_embd, 1)
#         #     self.scale_obj = nn.Linear(self.n_embd, 1)
#         #     with torch.no_grad():
#         #         nn.init.kaiming_uniform_(self.scale_att.weight, a=math.sqrt(5))
#         #         nn.init.zeros_(self.scale_att.bias)
#         #         nn.init.kaiming_uniform_(self.scale_obj.weight, a=math.sqrt(5))
#         #         nn.init.zeros_(self.scale_obj.bias)
#         # else:
#         #     self.scale_att = float(adapter_scalar)
#         #     self.scale_obj = float(adapter_scalar)
#         if adapter_scalar == "learnable_scalar":
#             self.scale = nn.Parameter(torch.ones(1))
#         else:
#             self.scale = float(adapter_scalar)

#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj = nn.Linear(self.down_size, self.n_embd)

        
#         self.dropout = dropout
#         self.init_option = init_option

#         ##
#         self.up_proj_att = nn.Linear(self.down_size, self.n_embd)
#         self.up_proj_obj = nn.Linear(self.down_size, self.n_embd)
#         self.non_linear_func_2 = nn.GELU()
        

#         self._reset_parameters()


#     def _reset_parameters(self):
#         if self.init_option == "bert":
#             raise NotImplementedError
#         elif self.init_option == "lora":
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.down_proj.bias)
#                 nn.init.zeros_(self.up_proj.weight)
#                 nn.init.zeros_(self.up_proj.bias)

#                 nn.init.zeros_(self.up_proj_obj.weight)
#                 nn.init.zeros_(self.up_proj_obj.bias)
#                 nn.init.zeros_(self.up_proj_att.weight)
#                 nn.init.zeros_(self.up_proj_att.bias)
                

#     def forward(self, x, add_residual=True, branch = None, residual=None):
#         residual = x if residual is None else residual
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

#         down = self.down_proj(x)
#         # down = self.non_linear_func(down)
#         # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
#         if branch == 'obj':
#             up = self.up_proj_obj(down)
#         elif branch == 'att':
#             up = self.up_proj_att(down)
#         else:
#             up = self.up_proj(down)
            
        
#         # if self.adapter_scalar == "learnable_scalar":
#         #     scale_att = torch.sigmoid(self.scale_att(x))
#         #     scale_obj = torch.sigmoid(self.scale_obj(x))
#         #     up = scale_att * up_att + scale_obj * up_obj
#         # else:
#         #     up = up_att * self.scale_att + up_obj * self.scale_obj

#         up = up * self.scale

#         if self.adapter_layernorm_option == 'out':
#             up = self.adapter_layer_norm_before(up)

#         if add_residual:
#             output = up + residual
#         else:
#             output = up
        
        
#         # adapter_prompt_att = self.non_linear_func_2(up_att).mean(dim=0, keepdim=True)
#         # adapter_prompt_att = apply_tfts(adapter_prompt_att, self.tfts_gamma_att, self.tfts_beta_att)

#         # adapter_prompt_obj = self.non_linear_func_2(up_obj).mean(dim=0, keepdim=True)
#         # adapter_prompt_obj = apply_tfts(adapter_prompt_obj, self.tfts_gamma_obj, self.tfts_beta_obj)
        
#         # adapter_prompt = torch.cat([adapter_prompt_att,adapter_prompt_obj],dim=0)
#         return output #, adapter_prompt, self.proj_out_att(adapter_prompt_att.squeeze()), self.proj_out_obj(adapter_prompt_obj.squeeze())


# class Adapter_Generator(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="lora",
#                  adapter_scalar="0.1",
#                  adapter_layernorm_option="none"):
#         super().__init__()
#         self.n_embd = d_model
#         self.down_size = bottleneck

#         #_before
#         self.adapter_layernorm_option = adapter_layernorm_option

#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

#         if adapter_scalar == "learnable_scalar":
#             self.scale = nn.Parameter(torch.ones(1))
#         else:
#             self.scale = float(adapter_scalar)

#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj = nn.Linear(self.down_size, self.n_embd)

#         self.dropout = dropout
#         self.init_option = init_option
          
#         self._reset_parameters()

#         ######## generate prompt
#         # self.non_linear_func_generator = nn.GELU()
#         # self.tfts_gamma, self.tfts_beta = init_tfts(d_model)
        
#         self.dap_downsample = nn.Linear(257, 1)
#         nn.init.zeros_(self.dap_downsample.weight)
#         nn.init.zeros_(self.dap_downsample.bias)
#         # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
#         self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
#         ###################

#     def _reset_parameters(self):
#         if self.init_option == "bert":
#             raise NotImplementedError
#         elif self.init_option == "lora":
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.up_proj.weight)
#                 nn.init.zeros_(self.down_proj.bias)
#                 nn.init.zeros_(self.up_proj.bias)

#     def forward(self, x, add_residual=True, residual=None):
#         residual = x if residual is None else residual
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

#         down = self.down_proj(x)
#         # down = self.non_linear_func(down)
#         # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
#         up = self.up_proj(down)

#         up = up * self.scale

#         if self.adapter_layernorm_option == 'out':
#             up = self.adapter_layer_norm_before(up)

#         if add_residual:
#             output = up + residual
#         else:
#             output = up #[257,16,1024]
        

#         ###########to generate prompt
#         if up.shape[0] == 257:
            
#             adapter_trans = self.dap_norm(up).permute(1,2,0)
#             adapter_prompt = self.dap_downsample(adapter_trans)
#             adapter_prompt  = adapter_prompt.permute(2,0,1)
#         else:
#             up = torch.cat((up[:1,:,:],up[2:,:,:]),dim=0)
#             adapter_trans = self.dap_norm(up).permute(1,2,0)
#             adapter_prompt = self.dap_downsample(adapter_trans)
#             adapter_prompt  = adapter_prompt.permute(2,0,1)

#         # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
#         # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
#         ###########
#         return output, adapter_prompt

# class Prompt_Generator(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,d_model=None):
#         super().__init__()
#         self.n_embd = d_model

#         ######## generate prompt
        
#         self.dap_downsample = nn.Linear(257, 1)
#         nn.init.zeros_(self.dap_downsample.weight)
#         nn.init.zeros_(self.dap_downsample.bias)
#         # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
#         self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
#         ###################

#     def forward(self, x):
#         ###########to generate prompt
#         if x.shape[0] == 257:
#             adapter_trans = self.dap_norm(x).permute(1,2,0)
#             adapter_prompt = self.dap_downsample(adapter_trans)
#             adapter_prompt  = adapter_prompt.permute(2,0,1)
#         else:
#             x = torch.cat((x[:1,:,:],x[2:,:,:]),dim=0)
#             adapter_trans = self.dap_norm(x).permute(1,2,0)
#             adapter_prompt = self.dap_downsample(adapter_trans)
#             adapter_prompt  = adapter_prompt.permute(2,0,1)

#         # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
#         # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
#         ###########
#         return adapter_prompt




# class Text_Prompt_Generator(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,d_model=None):
#         super().__init__()
#         self.n_embd = d_model

#         ######## generate prompt
        
#         self.dap_downsample = nn.Linear(8, 1)
#         nn.init.zeros_(self.dap_downsample.weight)
#         nn.init.zeros_(self.dap_downsample.bias)
#         # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
#         self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
#         ###################

#     def forward(self, x):
#         ###########to generate prompt
#         if x.shape[0] == 8:
#             adapter_trans = self.dap_norm(x).permute(1,2,0)
#             adapter_prompt = self.dap_downsample(adapter_trans)
#             adapter_prompt  = adapter_prompt.permute(2,0,1)
#         else:
#             x = torch.cat((x[:1,:,:],x[2:,:,:]),dim=0)
#             adapter_trans = self.dap_norm(x).permute(1,2,0)
#             adapter_prompt = self.dap_downsample(adapter_trans)
#             adapter_prompt  = adapter_prompt.permute(2,0,1)

#         # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
#         # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
#         ###########
#         return adapter_prompt

# class Concept_Prompt_Generator(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,d_model=None):
#         super().__init__()
#         self.n_embd = d_model
#         self.down_size = d_model//4
#         adapter_scalar = 0.1

#         ##########
#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj_att = nn.Linear(self.down_size, self.n_embd)
#         self.up_proj_obj = nn.Linear(self.down_size, self.n_embd)
#         self.scale = float(adapter_scalar)
#         nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.up_proj_att.weight)
#         nn.init.zeros_(self.up_proj_obj.weight)

#         nn.init.zeros_(self.down_proj.bias)
#         nn.init.zeros_(self.up_proj_att.bias)
#         nn.init.zeros_(self.up_proj_obj.bias)

#         ######## generate prompt
#         self.dap_downsample_att = nn.Linear(257, 1)
#         nn.init.zeros_(self.dap_downsample_att.weight)
#         nn.init.zeros_(self.dap_downsample_att.bias)
#         # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
#         self.dap_norm_att = LayerNorm(self.n_embd, eps=1e-6)
   
#         ###################
#         self.dap_downsample_obj = nn.Linear(257, 1)
#         nn.init.zeros_(self.dap_downsample_obj.weight)
#         nn.init.zeros_(self.dap_downsample_obj.bias)
#         # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
#         self.dap_norm_obj = LayerNorm(self.n_embd, eps=1e-6)

        

#     def forward(self, x):
#         ###########to generate prompt
#         if x.shape[0] == 257:
#             x_down = self.down_proj(x)
#             x_down_nolinear = self.non_linear_func(x_down)

#             x_up_att = self.up_proj_att(x_down_nolinear)
#             x_up_obj = self.up_proj_obj(x_down_nolinear)

#             x_att = x_up_att * self.scale + x
#             x_obj = x_up_obj * self.scale + x

#             adapter_trans = self.dap_norm_att(x_att).permute(1,2,0)
#             adapter_prompt_att = self.dap_downsample_att(adapter_trans)
#             adapter_prompt_att  = adapter_prompt_att.permute(2,0,1)

#             adapter_trans = self.dap_norm_obj(x_obj).permute(1,2,0)
#             adapter_prompt_obj = self.dap_downsample_obj(adapter_trans)
#             adapter_prompt_obj  = adapter_prompt_obj.permute(2,0,1)

#         else:
#             x = torch.cat((x[:1,:,:],x[3:,:,:]),dim=0)
#             x_down = self.down_proj(x)
#             x_down_nolinear = self.non_linear_func(x_down)

#             x_up_att = self.up_proj_att(x_down_nolinear)
#             x_up_obj = self.up_proj_obj(x_down_nolinear)

#             x_att = x_up_att * self.scale + x
#             x_obj = x_up_obj * self.scale + x

            
#             adapter_trans = self.dap_norm_att(x_att).permute(1,2,0)
#             adapter_prompt_att = self.dap_downsample_att(adapter_trans)
#             adapter_prompt_att  = adapter_prompt_att.permute(2,0,1)

#             adapter_trans = self.dap_norm_obj(x_obj).permute(1,2,0)
#             adapter_prompt_obj = self.dap_downsample_obj(adapter_trans)
#             adapter_prompt_obj  = adapter_prompt_obj.permute(2,0,1)
            
#         adapter_prompt = torch.cat((adapter_prompt_att,adapter_prompt_obj),dim=0)
#         # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
#         # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
#         ###########
#         return adapter_prompt
    
# class Disentangler(nn.Module):
#     def __init__(self, emb_dim):
#         super(Disentangler, self).__init__()
#         self.fc1 = nn.Linear(emb_dim, emb_dim)
#         self.bn1_fc = nn.BatchNorm1d(emb_dim)

#     def forward(self, x):
#         x = F.relu(self.bn1_fc(self.fc1(x)))
#         x = F.dropout(x, training=self.training)
#         return x


# class SoftNearestNeighborsLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super().__init__()

#         self.temperature = temperature
    
#     def forward(self, candidates, labels):
#         """
#         Calculate the distance between each pair of candidates. 
#         Pairs with the same label are considered positive,while pairs with different labels are negative.

#         Arguements:
#             candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
#                                        Each candidate is expected to have associated positives and negatives
#                                        from the other candidates. The tensor shape is (B, C), where B is the
#                                        batch size and C represents candidate features.
#             labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
#         Return:
#             loss (torch.Tensor)
#         """
#         if len(candidates) != len(labels):
#             raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
#         device = candidates.device
#         b, embed_dim = candidates.shape

#         scale = embed_dim**-0.5 
        
#         mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
#         mask.fill_diagonal_(0)

#         distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2 
#         exp_distance_matrix = torch.exp(-distance_matrix * scale / self.temperature) 
        
#         numerators = (exp_distance_matrix * mask).sum(dim=1)
#         denominators = exp_distance_matrix.sum(dim=1) 

#         # Remove the candidates that has no positive
#         indices = numerators.nonzero()
#         numerators = numerators[indices]
#         denominators = denominators[indices]

#         r = torch.log(numerators / denominators)
#         loss = -r.mean()

#         return loss

# def soft_contrastive_loss(candidates, labels, temperature):
#     """
#     Calculate the distance between each pair of candidates. 
#     Pairs with the same label are considered positive,while pairs with different labels are negative.

#     Arguements:
#         candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
#                                     Each candidate is expected to have associated positives and negatives
#                                     from the other candidates. The tensor shape is (B, C), where B is the
#                                     batch size and C represents candidate features.
#         labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
#     Return:
#         loss (torch.Tensor)
#     """
#     if len(candidates) != len(labels):
#         raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
#     device = candidates.device
#     b, embed_dim = candidates.shape
    
#     scale = embed_dim**-0.5 
    
#     mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
#     mask.fill_diagonal_(0)

#     distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2 
#     exp_distance_matrix = torch.exp(-distance_matrix * scale / temperature) 
    
#     numerators = (exp_distance_matrix * mask).sum(dim=1)
#     denominators = exp_distance_matrix.sum(dim=1) 

#     # Remove the candidates that has no positive
#     indices = numerators.nonzero()
#     numerators = numerators[indices]
#     denominators = denominators[indices]

#     r = torch.log(numerators / denominators)
#     loss = -r.mean()

#     return loss



# class MulitHeadAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, q, k, v):
#         B, N, C = q.shape
#         B, M, C = k.shape
#         q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
#         k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
#         v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
        
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class CrossAttentionLayer(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1,):
#         super().__init__()
#         self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
#         self.norm = nn.LayerNorm(d_model)

#         self.dropout = nn.Dropout(dropout)

#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             QuickGELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model)
#         )

#     def forward(self, q, kv):
#         q = q + self.cross_attn(q, kv, kv)
#         q = q + self.dropout(self.mlp(self.norm(q)))
#         return q


# class Cross_Prompt_Generator(nn.Module):
#     # Referece: https://github.com/ShoufaChen/AdaptFormer
#     def __init__(self,d_model=None, att_num = None, obj_num = None):
#         super().__init__()
#         self.n_embd = d_model
#         ####### prompt pool ##########
#         self.att_num = att_num
#         self.obj_num = obj_num
#         self.att_prompt_pool = nn.Parameter(torch.randn(1,att_num,d_model))
#         self.obj_prompt_pool = nn.Parameter(torch.randn(1,obj_num,d_model))
#         self.cross_layer_att = CrossAttentionLayer(d_model, d_model//64, 0.1)
#         self.cross_layer_obj = CrossAttentionLayer(d_model, d_model//64, 0.1)

#         ######## generate prompt ##########
        
#         self.dap_downsample = nn.Linear(257, 1)
#         nn.init.zeros_(self.dap_downsample.weight)
#         nn.init.zeros_(self.dap_downsample.bias)
#         # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
#         self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
#         ###################

#     def forward(self, x):
#         ###########to generate prompt
#         token_num, b, d = x.shape
#         x_q = x[:1,:,:].permute(1,0,2)
#         adapter_prompt_att = self.cross_layer_att(x_q,self.att_prompt_pool.repeat(b,1,1)).permute(1,0,2)
#         adapter_prompt_obj = self.cross_layer_obj(x_q,self.obj_prompt_pool.repeat(b,1,1)).permute(1,0,2)
#         adapter_prompt = torch.cat((adapter_prompt_att,adapter_prompt_obj),dim=0)
#         # if x.shape[0] == 257:
#         #     adapter_trans = self.dap_norm(x).permute(1,2,0)
#         #     adapter_prompt = self.dap_downsample(adapter_trans)
#         #     adapter_prompt  = adapter_prompt.permute(2,0,1)
#         # else:
#         #     x = torch.cat((x[:1,:,:],x[2:,:,:]),dim=0)
#         #     adapter_trans = self.dap_norm(x).permute(1,2,0)
#         #     adapter_prompt = self.dap_downsample(adapter_trans)
#         #     adapter_prompt  = adapter_prompt.permute(2,0,1)

#         # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
#         # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
#         ###########
#         return adapter_prompt



# class EMAU(nn.Module):
#     '''The Expectation-Maximization Attention Unit (EMAU).
#     Arguments:
#         c (int): The input and output channel number.
#         k (int): The number of the bases.
#         stage_num (int): The iteration number for EM.
#     '''
#     def __init__(self, c, k, stage_num=3):
#         super(EMAU, self).__init__()
#         self.stage_num = stage_num
        
#         #self.mu = nn.Parameter(torch.normal(0, 1e-1, size=(1, c, k)), requires_grad=True)
#         mu = torch.Tensor(1, c, k)
#         mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
#         mu = self._l2norm(mu, dim=1)
#         self.register_buffer('mu', mu)
#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             # norm_layer(c))  
#             nn.BatchNorm2d(c))
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, _BatchNorm):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
 
#     def forward(self, x):
#         idn = x  #[bs,512,65,65]
#         # The first 1x1 conv
#         x = self.conv1(x) #[bs,512,65,65]
#         # The EM Attention
#         b, c, h, w = x.size()
#         x = x.view(b, c, h*w)               # b * c * n
#         mu = self.mu.repeat(b, 1, 1)        # b * c * k
#         with torch.no_grad():
#             for i in range(self.stage_num):
#                 x_t = x.permute(0, 2, 1)    # b * n * c
#                 z = torch.bmm(x_t, mu)      # b * n * k
#                 z = F.softmax(z, dim=2)     # b * n * k
#                 z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
#                 mu = torch.bmm(x, z_)       # b * c * k
#                 mu = self._l2norm(mu, dim=1)
                
#         # !!! The moving averaging operation is writtern in train.py, which is significant.
#         z_t = z.permute(0, 2, 1)            # b * k * n
#         x = mu.matmul(z_t)                  # b * c * n
#         x = x.view(b, c, h, w)              # b * c * h * w
#         x = F.relu(x, inplace=True)
#         # The second 1x1 conv
#         x = self.conv2(x)
#         x = x + idn
#         x = F.relu(x, inplace=True)
#         return x, mu

#     def _l2norm(self, inp, dim):
#         '''Normlize the inp tensor with l2-norm.
#         Returns a tensor where each sub-tensor of input along the given dim is 
#         normalized such that the 2-norm of the sub-tensor is equal to 1.
#         Arguments:
#             inp (tensor): The input tensor.
#             dim (int): The dimension to slice over to get the ssub-tensors.
#         Returns:
#             (tensor) The normalized tensor.
#         '''
#         return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

# class EM(nn.Module):
#     def __init__(self, dim, base_num, iter_num=3, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.base_num = base_num
#         self.iter_num = iter_num
#         self.num_heads = num_heads

#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.base = nn.Parameter(torch.normal(0, 1e-1, size=(1, base_num, dim)), requires_grad=True)

#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.sim_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         # self.cross_attn_layer = CrossAttentionLayer(dim, num_heads, dropout=0.1)

    
#     def forward(self, feat, base_init=None): 
#         # bs, token_num, dim
#         B, M, C = feat.shape
#         if base_init is None:
#             base = self.base.repeat(B, 1, 1)
#         else:
#             base = base_init
#         B, N, C = base.shape
#         q = self.q_proj(base).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
#         k = self.k_proj(feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
#         v = self.v_proj(feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        
#         # with torch.no_grad():
#         for i in range(self.iter_num):
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             q = (attn @ v)    #.transpose(1, 2).reshape(B, N, C)
        
#         # reconstruct feat
#         # sim = (k @ q.transpose(-2,-1)) * self.scale
#         # sim = self.sim_drop(sim.softmax(dim=-1))
#         # feat_rec = (sim @ q).transpose(1, 2).reshape(B, M, C)
#         # feat_rec = self.proj(feat_rec)
#         # feat_final = self.proj_drop(feat_rec) + feat
#         q = q.transpose(1,2).reshape(B,N,C)
        
#         # feat_enhance = self.cross_attn_layer(feat,q)
#         return q

# class Troika_Base(nn.Module):
#     def __init__(self, config, attributes, classes, offset):
#         super().__init__()
#         self.clip = load_clip(name=config.clip_model, context_length=config.context_length,download_root=config.clip_arch)
#         self.tokenizer = SimpleTokenizer()
#         self.config = config
#         self.attributes = attributes
#         self.classes = classes
#         self.attr_dropout = nn.Dropout(config.attr_dropout)
#         self.cross_attn_dropout = config.cross_attn_dropout if hasattr(config, 'cross_attn_dropout') else 0.1
#         self.prim_loss_weight = config.prim_loss_weight if hasattr(config, 'prim_loss_weight') else 1

#         self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
#         self.offset = offset
#         self.enable_pos_emb = True
#         dtype = self.clip.dtype
#         # dtype = None
#         if dtype is None:
#             self.dtype = torch.float16
#         else:
#             self.dtype = dtype
#         self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
#         # pdb.set_trace()
#         # freeze CLIP's parameters
#         for p in self.parameters():
#             p.requires_grad = False

#         # only consider ViT as visual encoder
#         assert 'ViT' in config.clip_model

#         self.additional_visual_params = self.add_visual_tunable_params()

#         output_dim = self.clip.visual.output_dim

#         self.soft_att_obj = nn.Parameter(self.soft_att_obj)
#         self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
#         self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
#         self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

#         self.attr_disentangler = Disentangler(output_dim)
#         self.obj_disentangler = Disentangler(output_dim)

#         # self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout) for _ in range(config.cmt_layers)])
#         self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
#         self.patch_norm = nn.LayerNorm(output_dim)
        
#         ###############
#         self.additional_text_params = self.add_text_tunable_params()
#         self.additional_prompt_generator = self.add_visual_prompt_generator()
        
#         # self.att_qformer = CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout)
#         # self.obj_qformer = CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout)
#         # self.additional_prompt_generator_att = self.add_visual_prompt_generator()
#         # self.additional_prompt_generator_obj = self.add_visual_prompt_generator()
#         # self.additional_txt_prompt_generator = self.add_text_prompt_generator()
#         self.visual_prompt_num = 2
    
#     def add_visual_prompt_generator(self):
#         adapter_num = self.clip.visual.transformer.layers
#         # params = nn.ModuleList([Concept_Prompt_Generator(d_model=self.clip.visual.transformer.width,att_num=len(self.attributes),obj_num=len(self.classes)) for _ in range(adapter_num)])
#         params = nn.ModuleList([Prompt_Generator(d_model=self.clip.visual.transformer.width) for _ in range(adapter_num)])
#         return params
    
#     def add_text_prompt_generator(self):
#         adapter_num = self.clip.transformer.layers
#         params = nn.ModuleList([Text_Prompt_Generator(d_model=self.clip.transformer.width) for _ in range(adapter_num)])
#         return params
    
#     def add_visual_tunable_params(self):
#         # adapter_num = 2 * self.clip.visual.transformer.layers
#         adapter_num = self.clip.visual.transformer.layers
#         params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, 
#                                     bottleneck=self.config.adapter_dim, 
#                                     dropout=self.config.adapter_dropout
#                                 ) for _ in range(adapter_num)])
#         return params

#     def add_text_tunable_params(self):
#         # adapter_num = 2 * self.clip.transformer.layers
#         adapter_num = self.clip.transformer.layers
#         # params = nn.ModuleList([AdapterConcept(d_model=self.clip.transformer.width, 
#         #                             bottleneck=self.config.adapter_dim, 
#         #                             dropout=self.config.adapter_dropout
#         #                         ) for _ in range(adapter_num)])
#         params = nn.ModuleList([Adapter(d_model=self.clip.transformer.width, 
#                                     bottleneck=self.config.adapter_dim, 
#                                     dropout=self.config.adapter_dropout
#                                 ) for _ in range(adapter_num)])
#         return params
    


#     def zero_shot_visual(self, x: torch.Tensor):
#         x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.clip.visual.positional_embedding.to(x.dtype)
#         x = self.clip.visual.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         img_feature = self.clip.visual.transformer(x)
#         x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

#         x = self.clip.visual.ln_post(x)
#         if self.clip.visual.proj is not None:
#             x = x @ self.clip.visual.proj

#         return x[:, 0, :], x
    
#     def zero_shot_text(self, pairs):
#         """Function to get the clip representations.

#         Args:
#             model (nn.Module): the clip model
#             test_dataset (CompositionDataset): the test/validation dataset
#             config (argparse.ArgumentParser): config/args
#             device (str): device type cpu/cuda:0

#         Returns:
#             torch.Tensor: returns the tensor with the attribute-object
#                 representations with clip model.
#         """
#         pairs = [(attr.replace(".", " ").lower(),
#                 obj.replace(".", " ").lower())
#                 for attr, obj in pairs]

#         prompts_pairs = [f"a photo of {attr} {obj}" for attr, obj in pairs]
#         tokenized_pairs = self.tokenizer(prompts_pairs, context_length=self.config.context_length).cuda()

#         prompts_attrs = [f"a photo of {attr}" for attr in self.attributes]
#         tokenized_attrs = self.tokenizer(prompts_attrs, context_length=self.config.context_length).cuda()

#         prompts_objs = [f"a photo of {obj}" for obj in self.classes]
#         tokenized_objs = self.tokenizer(prompts_objs, context_length=self.config.context_length).cuda()
#         # test_batch_tokens = np.array_split(
#         #     tokenized_prompts,
#         #     len(tokenized_prompts) //
#         #     config.text_encoder_batch_size)
#         # rep = torch.Tensor().to(device).type(model.dtype)
#         with torch.no_grad():
#             att_feat,_ = self.text_encoder(tokenized_attrs, None, enable_pos_emb=True)
#             att_feat = att_feat/att_feat.norm(dim=-1,keepdim=True)

#             pair_feat,_ = self.text_encoder(tokenized_pairs,None,  enable_pos_emb=True)
#             pair_feat = pair_feat/pair_feat.norm(dim=-1,keepdim=True)

#             obj_feat,_ = self.text_encoder(tokenized_objs,None,  enable_pos_emb=True)
#             obj_feat = obj_feat/obj_feat.norm(dim=-1,keepdim=True)
#             # for tokenized_ele in tokenized_all:
#             #     batch_tokens = batch_tokens.to(device)
#             #     _text_features = model.text_encoder(
#             #         batch_tokens, enable_pos_emb=True)
#             #     text_features = _text_features / _text_features.norm(
#             #         dim=-1, keepdim=True
#             #     )
#             #     rep = torch.cat((rep, text_features), dim=0)
#         return pair_feat, att_feat, obj_feat
#         # return [tokenized_pairs,tokenized_attrs,tokenized_objs]

#     def visual(self, x: torch.Tensor):
#         x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.clip.visual.positional_embedding.to(x.dtype)
#         original_x = x

#         #########
#         prompt = self.additional_em_params[0](x[:, 1:, :]) #self.em_module(x[:, 1:, :]) # 24,4,1024
        
#         x = torch.cat((
#                 x[:, :1, :],
#                 self.prompt_dropout(prompt),
#                 x[:, 1:, :]
#             ), dim=1)
#         #########
#         x = self.clip.visual.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         original_x = original_x.permute(1,0,2)
#         if self.config.prompt_deep:
#             for i in range(self.clip.visual.transformer.layers):
#                 if i == 0:
#                     x = self.clip.visual.transformer.resblocks[i](x)
#                     original_x = self.clip.visual.transformer.resblocks[i](original_x)
#                 else:
#                     if i <= self.clip.visual.transformer.layers:#self.deep_prompt_embeddings.shape[0]:
#                         prompt = self.additional_em_params[i](original_x[1:, :, :].permute(1,0,2))#self.em_module(original_x[1:, :, :].permute(1,0,2))
#                         # prompt = self.em_module(x[(1+self.config.prompt_num_tokens):, :, :].permute(1,0,2),prompt)
#                         deep_prompt_emb = self.prompt_dropout(prompt)
                        
#                         x = torch.cat((
#                             x[:1, :, :],
#                             deep_prompt_emb.permute(1,0,2),
#                             x[(1+self.config.prompt_num_tokens):, :, :]
#                         ), dim=0)

#                     x = self.clip.visual.transformer.resblocks[i](x)
#                     original_x = self.clip.visual.transformer.resblocks[i](original_x)
#             img_feature = x
#         else:
#             img_feature = self.clip.visual.transformer(x)
#         x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

#         x = self.clip.visual.ln_post(x)
#         if self.clip.visual.proj is not None:
#             x = x @ self.clip.visual.proj
#         # x = self.clip.visual.ln_pre(x)

#         # x = x.permute(1, 0, 2)  # NLD -> LND
#         # img_feature = self.clip.visual.transformer(x)
#         # x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

#         # x = self.clip.visual.ln_post(x)
        
#         # # if self.clip.visual.proj is not None:
#         # #     x = x @ self.clip.visual.proj
        
#         # #return x[:, 0, :], x
#         # ############# add em algorithm ################
        
#         # x_em = self.em_module(x)
#         return x[:, 0, :], x

#         # return x[:, 0, :], x
    
#     def visual_adapt(self, x: torch.Tensor, prompt: torch.Tensor):
#         x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.clip.visual.positional_embedding.to(x.dtype)
#         #########
        
#         x = torch.cat((
#                 x[:, :1, :],
#                 self.prompt_dropout(prompt),
#                 x[:, 1:, :]
#             ), dim=1)
#         # pdb.set_trace()
#         #########
#         x = self.clip.visual.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         img_feature = self.clip.visual.transformer(x)
#         x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

#         x = self.clip.visual.ln_post(x)
#         if self.clip.visual.proj is not None:
#             x = x @ self.clip.visual.proj
        
#         #return x[:, 0, :], x
#         ############# add em algorithm ################
#         # x_em = self.em_module(x)

#         return x[:, 0, :], x
     
#     def encode_text_with_adapter(self, token_ids, token_tensors, enable_pos_emb, flag=None):
#         """The forward function to compute representations for the prompts.

#         Args:
#             token_ids (torch.tensor): the token ids, which
#                 contains the <eos> token.
#             token_tensors (torch.Tensor, optional): the tensor
#                 embeddings for the token ids. Defaults to None.
#             enable_pos_emb (bool, optional): adds the learned
#                 positional embeddigngs if true. Defaults to False.

#         Returns:
#             torch.Tensor: the vector representation of the prompt.
#         """
#         if token_tensors is not None:
#             text_features = token_tensors
#         else:
#             text_features = self.clip.token_embedding(token_ids)

#         text_features = text_features.type(self.dtype)
#         x = (
#             text_features + self.clip.positional_embedding.type(self.dtype)
#             if enable_pos_emb
#             else text_features
#         )
#         x = x.permute(1, 0, 2)
#         # text_feature = self.transformer(x)
#         for i_block in range(self.clip.transformer.layers):
#             # MHA
#             # adapt_x = self.additional_text_params[i_block](x, add_residual=False)
#             # if i_block==0:
#             #     adapter_prompt = self.additional_txt_prompt_generator[i_block](x.type(torch.float))
#             #     # x = x + adapter_prompt.repeat(8,1,1)
#             #     x = torch.cat([adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
#             #     # x = torch.cat([x[0:1, :, :], adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
#             # else:
#             #     adapter_prompt = self.additional_txt_prompt_generator[i_block](x.type(torch.float))
#             #     # x = x + adapter_prompt.repeat(8,1,1)
#             #     x = torch.cat([adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
            
#             residual = x
#             x = self.clip.transformer.resblocks[i_block].attention(
#                 self.clip.transformer.resblocks[i_block].ln_1(x)
#             )
#             # x = x + adapt_x + residual
#             x = x + residual

#             # FFN
#             i_adapter = i_block #+ self.clip.transformer.layers
            
#             # adapt_x = self.additional_text_params[i_adapter](x.type(torch.float), add_residual=False, branch= flag).type(self.dtype)
#             adapt_x = self.additional_text_params[i_adapter](x.type(torch.float), add_residual=False).type(self.dtype)
#             residual = x
#             x = self.clip.transformer.resblocks[i_block].mlp(
#                 self.clip.transformer.resblocks[i_block].ln_2(x)
#             )
#             x = x + adapt_x + residual
#             # x = x + residual
        
#         text_feature = x
#         x = text_feature.permute(1, 0, 2)
#         x = self.clip.ln_final(x)
#         tf = (
#             x[
#                 torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
#             ]  # POS of <EOS>
#             @ self.clip.text_projection
#         )
#         return tf, text_feature
    
#     def encode_image_with_adapter(self, x: torch.Tensor, flag=None):
#         x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.clip.visual.positional_embedding.to(x.dtype)

#         x = self.clip.visual.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         # img_feature = self.clip.visual.transformer(x)
#         for i_block in range(self.clip.visual.transformer.layers):
#             ###
#             # if i_block>0:
#             #     deep_prompt_emb = self.prompt_dropout(
#             #                     self.deep_prompt_embeddings[i_block-1].expand(BatchSize, -1, -1))                
#             #     x = torch.cat((
#             #         x[:1, :, :],
#             #         deep_prompt_emb.permute(1,0,2),
#             #         x[(1+self.num_tokens):, :, :]
#             #     ), dim=0)
#             # MHA
#             # adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
#             # if flag == 'att':
#             #     adapter_prompt = self.additional_prompt_generator_att[i_block](x.type(torch.float))
#             # elif flag == 'obj':
#             #     adapter_prompt = self.additional_prompt_generator_obj[i_block](x.type(torch.float))
#             # elif flag is None:
#             # adapter_prompt = self.additional_prompt_generator[i_block](x.type(torch.float))

#             # if i_block==0:
#             #     # adapter_prompt = self.additional_prompt_generator[i_block](x.type(torch.float))
#             #     x = torch.cat([x[0:1, :, :], adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
#             # else:
                
#             #     # adapter_prompt = self.additional_prompt_generator[i_block](x.type(torch.float))
#             #     x = torch.cat([x[0:1, :, :], adapter_prompt.type(self.dtype), x[(1+self.visual_prompt_num):, :, :]], dim=0)#, x[2:, :, :]], dim=0)#x[(1+self.visual_prompt_num):, :, :]], dim=0)
            
#             residual = x
#             x = self.clip.visual.transformer.resblocks[i_block].attention(
#                 self.clip.visual.transformer.resblocks[i_block].ln_1(x)
#             )
#             # x = x + adapt_x + residual
#             x = x + residual

#             # FFN
#             i_adapter = i_block #+ self.clip.visual.transformer.layers
#             # adapt_x, adapter_prompt = self.additional_visual_params[i_adapter](x, add_residual=False)
#             adapt_x = self.additional_visual_params[i_adapter](x.type(torch.float), add_residual=False).type(self.dtype)
#             residual = x
#             x = self.clip.visual.transformer.resblocks[i_block].mlp(
#                 self.clip.visual.transformer.resblocks[i_block].ln_2(x)
#             )
#             x = x + adapt_x + residual
#             # x = x + residual
#             # if i_block==0:
#             #     x = torch.cat([x[0:1, :, :], adapter_prompt, x[1:, :, :]], dim=0)
#             # # elif i_block==self.clip.visual.transformer.layers-1:
#             # #     x = x
#             # else:
#             #     x = torch.cat([x[0:1, :, :], adapter_prompt, x[2:, :, :]], dim=0)


#         img_feature = x.permute(1, 0, 2)  # LND -> NLD

#         img_feature_ori = self.clip.visual.ln_post(img_feature)
#         if self.clip.visual.proj is not None:
#             img_feature = img_feature_ori @ self.clip.visual.proj

#         return img_feature[:, 0, :], img_feature

#     def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
#         return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

#     def encode_image(self, x: torch.Tensor):
#         x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.clip.visual.positional_embedding.to(x.dtype)
#         x = self.clip.visual.ln_pre(x)


#         x = x.permute(1, 0, 2)  # NLD -> LND
#         img_feature = self.clip.visual.transformer(x)
#         x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

#         x = self.clip.visual.ln_post(x)
        
#         if self.clip.visual.proj is not None:
#             x = x @ self.clip.visual.proj
        
#         return x[:, 0, :], x
#         # ############# add em 
#         # return self.encode_image_with_adapter(x)
    
#     def construct_soft_prompt(self):
#         # token_ids indicates the position of [EOS]
#         token_ids = self.tokenizer(self.config.prompt_template,
#                               context_length=self.config.context_length).cuda()

#         tokenized = torch.cat(
#             [
#                 self.tokenizer(tok, context_length=self.config.context_length)
#                 for tok in self.attributes + self.classes
#             ]
#         )
#         orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
#         soft_att_obj = torch.zeros(
#             (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
#         )
#         for idx, rep in enumerate(orig_token_embedding):
#             eos_idx = tokenized[idx].argmax()
#             soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

#         ctx_init = self.config.ctx_init
#         assert isinstance(ctx_init, list)
#         n_ctx = [len(ctx.split()) for ctx in ctx_init]
#         prompt = self.tokenizer(ctx_init,
#                             context_length=self.config.context_length).cuda()
#         with torch.no_grad():
#             embedding = self.clip.token_embedding(prompt)

#         comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
#         attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
#         obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)
        
#         return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


#     def construct_token_tensors(self, pair_idx):
#         attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
#         token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
#         for i_element in range(self.token_ids.shape[0]):
#             class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
#             token_tensor.append(self.clip.token_embedding(
#                 class_token_ids.cuda()
#             ).type(self.clip.dtype))

#         eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
#         soft_att_obj = self.attr_dropout(self.soft_att_obj)
#         # comp
#         token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
#             attr_idx
#         ].type(self.clip.dtype)
#         token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
#             obj_idx + self.offset
#         ].type(self.clip.dtype)
#         token_tensor[0][
#             :, 1 : len(self.comp_ctx_vectors) + 1, :
#         ] = self.comp_ctx_vectors.type(self.clip.dtype)
#         # attr
#         token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
#             :self.offset
#         ].type(self.clip.dtype)
#         token_tensor[1][
#             :, 1 : len(self.attr_ctx_vectors) + 1, :
#         ] = self.attr_ctx_vectors.type(self.clip.dtype)
#         # obj
#         token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
#             self.offset:
#         ].type(self.clip.dtype)
#         token_tensor[2][
#             :, 1 : len(self.obj_ctx_vectors) + 1, :
#         ] = self.obj_ctx_vectors.type(self.clip.dtype)

#         return token_tensor
    
#     def loss_calu(self, predict, target):
#         loss_fn = CrossEntropyLoss()
#         _, batch_attr, batch_obj, batch_target = target
#         comp_logits, attr_logits, obj_logits = predict
#         batch_attr = batch_attr.cuda()
#         batch_obj = batch_obj.cuda()
#         batch_target = batch_target.cuda()
#         loss_comp = loss_fn(comp_logits, batch_target)
#         loss_attr = loss_fn(attr_logits, batch_attr)
#         loss_obj = loss_fn(obj_logits, batch_obj)
#         loss = loss_comp * self.config.pair_loss_weight +\
#                loss_attr * self.config.attr_loss_weight +\
#                loss_obj * self.config.obj_loss_weight
#         return loss


#     def logit_infer(self, predict, pairs):
#         comp_logits, attr_logits, obj_logits = predict
#         attr_pred = F.softmax(attr_logits, dim=-1)
#         obj_pred = F.softmax(obj_logits, dim=-1)
#         for i_comp in range(comp_logits.shape[-1]):
#             weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
#             weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
#             comp_logits[:, i_comp] = comp_logits[:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
#         return comp_logits

    
#     def encode_text_for_open(self, idx):
#         token_tensors = self.construct_token_tensors(idx)
#         text_features = []
#         for i_element in range(self.token_ids.shape[0]):
#             _text_features, _ = self.encode_text(
#                 self.token_ids[i_element],
#                 token_tensors[i_element],
#                 enable_pos_emb=self.enable_pos_emb,
#             )

#             idx_text_features = _text_features / _text_features.norm(
#                 dim=-1, keepdim=True
#             )
#             text_features.append(idx_text_features)
#         return text_features

    
#     def forward_for_open(self, batch, text_feats):
#         batch_img = batch[0].cuda()
#         b = batch_img.shape[0]
#         # l, _ = idx.shape
#         batch_img, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))
#         batch_img_features = [batch_img, self.attr_disentangler(batch_img.type(torch.float)).type(self.dtype), self.obj_disentangler(batch_img.type(torch.float)).type(self.dtype)]
#         normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

#         logits = list()
#         for i_element in range(self.token_ids.shape[0]):
#             idx_text_features = text_feats[i_element]

#             # CMT
#             cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
#             batch_patch = self.patch_norm(batch_patch)
#             for layer in self.cmt:
#                 cmt_text_features = layer(cmt_text_features, batch_patch)
#             cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

#             cmt_text_features = cmt_text_features / cmt_text_features.norm(
#                 dim=-1, keepdim=True
#             )

#             logits.append(
#                 torch.einsum(
#                     "bd, bkd->bk", 
#                     normalized_img_features[i_element], 
#                     cmt_text_features * self.clip.logit_scale.exp()
#             ))
#         return logits

    
#     def forward(self, batch, idx, training=True, pairs=None):
#         batch_img = batch[0].cuda()
        
#         # labels = [batch[3].cuda(),batch[1].cuda(),batch[2].cuda()]
#         # b = batch_img.shape[0]
#         # l, _ = idx.shape
        
#         # batch_img, batch_patch = self.zero_shot_visual(batch_img.type(self.clip.dtype))
#         batch_img, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))
#         # batch_img_att, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype),flag='att')
#         # batch_img_obj, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype),flag='obj')
#         # batch_img, batch_patch = self.visual(batch_img.type(self.clip.dtype)) #batch_img.shape = [bs,768]; batch_patch.shape = [16,257,768]
        
#         # batch_img_adapt, batch_patch_adapt = self.visual_adapt(batch_img.type(self.clip.dtype), prompt)
#         # batch_img_features = [batch_img_adapt, self.attr_disentangler(batch_img_adapt), self.obj_disentangler(batch_img_adapt)]
#         # batch_img_features = [batch_img, batch_patch[:,1,:], batch_patch[:,2,:]]
#         # batch_img_features = [batch_img, self.attr_disentangler(batch_patch[:,1,:].type(torch.float)).type(self.dtype), self.obj_disentangler(batch_patch[:,2,:].type(torch.float)).type(self.dtype)]
#         batch_img_features = [batch_img, self.attr_disentangler(batch_img).type(self.dtype), self.obj_disentangler(batch_img).type(self.dtype)]
#         # batch_img_features = [batch_img, batch_img, batch_img]
#         # batch_img_features = [batch_img_com, batch_img_att.type(self.dtype), batch_img_obj.type(self.dtype)]
        
#         normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

#         token_tensors = self.construct_token_tensors(idx)
        
        
#         logits = list()
#         all_text_feat = []
#         flags = ['com','att','obj']
#         for i_element in range(self.token_ids.shape[0]):
#             _text_features, _ = self.encode_text_with_adapter(
#                 self.token_ids[i_element],
#                 token_tensors[i_element],
#                 enable_pos_emb=self.enable_pos_emb,
#                 flag = flags[i_element]
#             )

#             idx_text_features = _text_features / _text_features.norm(
#                 dim=-1, keepdim=True
#             )
            
#             logits.append(torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*idx_text_features.T))
            
#             all_text_feat.append(idx_text_features)
        
#         if training and pairs is not None:
#             # zero_shot_image, _ = self.zero_shot_visual(batch[0].cuda().type(self.clip.dtype))
#             # zs_image = zero_shot_image/zero_shot_image.norm(dim=-1, keepdim=True)
#             zs_pair_feat, zs_att_feat, zs_obj_feat = self.zero_shot_text(pairs)
#             # loss_image = F.l1_loss(normalized_img_features[0], zs_image.cuda(),reduction='mean') * 10
#             loss_pair = 25 * F.l1_loss(all_text_feat[0],zs_pair_feat.cuda(),reduction='mean')
#             loss_att = 25 * F.l1_loss(all_text_feat[1],zs_att_feat.cuda(),reduction='mean')
#             loss_obj = 25 * F.l1_loss(all_text_feat[2],zs_obj_feat.cuda(),reduction='mean')
           
#             # loss_contra = soft_contrastive_loss(batch_patch[:,1,:], batch[1].cuda(),0.1) + soft_contrastive_loss(batch_patch[:,2,:], batch[2].cuda(),0.1)
            
#             return logits, loss_pair+loss_att+loss_obj #+ 0.1*loss_contra loss_image+
        
#         else:
#             return logits



######################### original ###################
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
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
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
        # self.non_linear_func_generator = nn.GELU()
        # self.tfts_gamma, self.tfts_beta = init_tfts(d_model)
        
        self.dap_downsample = nn.Linear(257, 1)
        nn.init.zeros_(self.dap_downsample.weight)
        nn.init.zeros_(self.dap_downsample.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
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
        if up.shape[0] == 257:
            
            adapter_trans = self.dap_norm(up).permute(1,2,0)
            adapter_prompt = self.dap_downsample(adapter_trans)
            adapter_prompt  = adapter_prompt.permute(2,0,1)
        else:
            up = torch.cat((up[:1,:,:],up[2:,:,:]),dim=0)
            adapter_trans = self.dap_norm(up).permute(1,2,0)
            adapter_prompt = self.dap_downsample(adapter_trans)
            adapter_prompt  = adapter_prompt.permute(2,0,1)

        # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
        # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
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




class Text_Prompt_Generator(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,d_model=None):
        super().__init__()
        self.n_embd = d_model

        ######## generate prompt
        
        self.dap_downsample = nn.Linear(8, 1)
        nn.init.zeros_(self.dap_downsample.weight)
        nn.init.zeros_(self.dap_downsample.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
   
        ###################

    def forward(self, x):
        ###########to generate prompt
        if x.shape[0] == 8:
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

class Concept_Prompt_Generator(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,d_model=None):
        super().__init__()
        self.n_embd = d_model
        self.down_size = d_model//4
        adapter_scalar = 0.1

        ##########
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_att = nn.Linear(self.down_size, self.n_embd)
        self.up_proj_obj = nn.Linear(self.down_size, self.n_embd)
        self.scale = float(adapter_scalar)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj_att.weight)
        nn.init.zeros_(self.up_proj_obj.weight)

        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj_att.bias)
        nn.init.zeros_(self.up_proj_obj.bias)

        ######## generate prompt
        self.dap_downsample_att = nn.Linear(257, 1)
        nn.init.zeros_(self.dap_downsample_att.weight)
        nn.init.zeros_(self.dap_downsample_att.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm_att = LayerNorm(self.n_embd, eps=1e-6)
   
        ###################
        self.dap_downsample_obj = nn.Linear(257, 1)
        nn.init.zeros_(self.dap_downsample_obj.weight)
        nn.init.zeros_(self.dap_downsample_obj.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm_obj = LayerNorm(self.n_embd, eps=1e-6)

        

    def forward(self, x):
        ###########to generate prompt
        if x.shape[0] == 257:
            x_down = self.down_proj(x)
            x_down_nolinear = self.non_linear_func(x_down)

            x_up_att = self.up_proj_att(x_down_nolinear)
            x_up_obj = self.up_proj_obj(x_down_nolinear)

            x_att = x_up_att * self.scale + x
            x_obj = x_up_obj * self.scale + x

            
            adapter_trans = self.dap_norm_att(x_att).permute(1,2,0)
            adapter_prompt_att = self.dap_downsample_att(adapter_trans)
            adapter_prompt_att  = adapter_prompt_att.permute(2,0,1)

            adapter_trans = self.dap_norm_obj(x_obj).permute(1,2,0)
            adapter_prompt_obj = self.dap_downsample_obj(adapter_trans)
            adapter_prompt_obj  = adapter_prompt_obj.permute(2,0,1)

        else:
            x = torch.cat((x[:1,:,:],x[3:,:,:]),dim=0)
            x_down = self.down_proj(x)
            x_down_nolinear = self.non_linear_func(x_down)

            x_up_att = self.up_proj_att(x_down_nolinear)
            x_up_obj = self.up_proj_obj(x_down_nolinear)

            x_att = x_up_att * self.scale + x
            x_obj = x_up_obj * self.scale + x

            
            adapter_trans = self.dap_norm_att(x_att).permute(1,2,0)
            adapter_prompt_att = self.dap_downsample_att(adapter_trans)
            adapter_prompt_att  = adapter_prompt_att.permute(2,0,1)

            adapter_trans = self.dap_norm_obj(x_obj).permute(1,2,0)
            adapter_prompt_obj = self.dap_downsample_obj(adapter_trans)
            adapter_prompt_obj  = adapter_prompt_obj.permute(2,0,1)
            
        adapter_prompt = torch.cat((adapter_prompt_att,adapter_prompt_obj),dim=0)
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


class SoftNearestNeighborsLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature
    
    def forward(self, candidates, labels):
        """
        Calculate the distance between each pair of candidates. 
        Pairs with the same label are considered positive,while pairs with different labels are negative.

        Arguements:
            candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
                                       Each candidate is expected to have associated positives and negatives
                                       from the other candidates. The tensor shape is (B, C), where B is the
                                       batch size and C represents candidate features.
            labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
        Return:
            loss (torch.Tensor)
        """
        if len(candidates) != len(labels):
            raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
        device = candidates.device
        b, embed_dim = candidates.shape

        scale = embed_dim**-0.5 
        
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
        mask.fill_diagonal_(0)

        distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2 
        exp_distance_matrix = torch.exp(-distance_matrix * scale / self.temperature) 
        
        numerators = (exp_distance_matrix * mask).sum(dim=1)
        denominators = exp_distance_matrix.sum(dim=1) 

        # Remove the candidates that has no positive
        indices = numerators.nonzero()
        numerators = numerators[indices]
        denominators = denominators[indices]

        r = torch.log(numerators / denominators)
        loss = -r.mean()

        return loss

def soft_contrastive_loss(candidates, labels, temperature):
    """
    Calculate the distance between each pair of candidates. 
    Pairs with the same label are considered positive,while pairs with different labels are negative.

    Arguements:
        candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
                                    Each candidate is expected to have associated positives and negatives
                                    from the other candidates. The tensor shape is (B, C), where B is the
                                    batch size and C represents candidate features.
        labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
    Return:
        loss (torch.Tensor)
    """
    if len(candidates) != len(labels):
        raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
    device = candidates.device
    b, embed_dim = candidates.shape
    
    scale = embed_dim**-0.5 
    
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
    mask.fill_diagonal_(0)

    distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2 
    exp_distance_matrix = torch.exp(-distance_matrix * scale / temperature) 
    
    numerators = (exp_distance_matrix * mask).sum(dim=1)
    denominators = exp_distance_matrix.sum(dim=1) 

    # Remove the candidates that has no positive
    indices = numerators.nonzero()
    numerators = numerators[indices]
    denominators = denominators[indices]

    r = torch.log(numerators / denominators)
    loss = -r.mean()

    return loss



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


class Cross_Prompt_Generator(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,d_model=None, att_num = None, obj_num = None):
        super().__init__()
        self.n_embd = d_model
        

        ####### prompt pool ##########
        self.att_num = att_num
        self.obj_num = obj_num
        self.att_prompt_pool = nn.Parameter(torch.randn(1,att_num,d_model))
        self.obj_prompt_pool = nn.Parameter(torch.randn(1,obj_num,d_model))
        self.cross_layer_att = CrossAttentionLayer(d_model, d_model//64, 0.1)
        self.cross_layer_obj = CrossAttentionLayer(d_model, d_model//64, 0.1)

        ######## generate prompt ##########
        
        self.dap_downsample = nn.Linear(257, 1)
        nn.init.zeros_(self.dap_downsample.weight)
        nn.init.zeros_(self.dap_downsample.bias)
        # self.dap_film = nn.Linear(dap_config.TASK_EMB, config.hidden_size * 2)
        self.dap_norm = LayerNorm(self.n_embd, eps=1e-6)
        ###################

    def forward(self, x):
        ###########to generate prompt
        token_num, b, d = x.shape
        x_q = x[:1,:,:].permute(1,0,2)
        adapter_prompt_att = self.cross_layer_att(x_q,self.att_prompt_pool.repeat(b,1,1)).permute(1,0,2)
        adapter_prompt_obj = self.cross_layer_obj(x_q,self.obj_prompt_pool.repeat(b,1,1)).permute(1,0,2)
        adapter_prompt = torch.cat((adapter_prompt_att,adapter_prompt_obj),dim=0)
        # if x.shape[0] == 257:
        #     adapter_trans = self.dap_norm(x).permute(1,2,0)
        #     adapter_prompt = self.dap_downsample(adapter_trans)
        #     adapter_prompt  = adapter_prompt.permute(2,0,1)
        # else:
        #     x = torch.cat((x[:1,:,:],x[2:,:,:]),dim=0)
        #     adapter_trans = self.dap_norm(x).permute(1,2,0)
        #     adapter_prompt = self.dap_downsample(adapter_trans)
        #     adapter_prompt  = adapter_prompt.permute(2,0,1)

        # adapter_prompt = self.non_linear_func(up).mean(dim=0, keepdim=True)
        # adapter_prompt = apply_tfts(adapter_prompt, self.tfts_gamma, self.tfts_beta)
        ###########
        return adapter_prompt



class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num
        
        #self.mu = nn.Parameter(torch.normal(0, 1e-1, size=(1, c, k)), requires_grad=True)
        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            # norm_layer(c))  
            nn.BatchNorm2d(c))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 
    def forward(self, x):
        idn = x  #[bs,512,65,65]
        # The first 1x1 conv
        x = self.conv1(x) #[bs,512,65,65]
        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(x_t, mu)      # b * n * k
                z = F.softmax(z, dim=2)     # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       # b * c * k
                mu = self._l2norm(mu, dim=1)
                
        # !!! The moving averaging operation is writtern in train.py, which is significant.
        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        x = x.view(b, c, h, w)              # b * c * h * w
        x = F.relu(x, inplace=True)
        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)
        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class EM(nn.Module):
    def __init__(self, dim, base_num, iter_num=3, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.base_num = base_num
        self.iter_num = iter_num
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.base = nn.Parameter(torch.normal(0, 1e-1, size=(1, base_num, dim)), requires_grad=True)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.sim_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.cross_attn_layer = CrossAttentionLayer(dim, num_heads, dropout=0.1)

    
    def forward(self, feat, base_init=None): 
        # bs, token_num, dim
        B, M, C = feat.shape
        if base_init is None:
            base = self.base.repeat(B, 1, 1)
        else:
            base = base_init
        B, N, C = base.shape
        q = self.q_proj(base).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(feat).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        
        # with torch.no_grad():
        for i in range(self.iter_num):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            q = (attn @ v)    #.transpose(1, 2).reshape(B, N, C)
        
        # reconstruct feat
        # sim = (k @ q.transpose(-2,-1)) * self.scale
        # sim = self.sim_drop(sim.softmax(dim=-1))
        # feat_rec = (sim @ q).transpose(1, 2).reshape(B, M, C)
        # feat_rec = self.proj(feat_rec)
        # feat_final = self.proj_drop(feat_rec) + feat
        q = q.transpose(1,2).reshape(B,N,C)
        
        # feat_enhance = self.cross_attn_layer(feat,q)
        return q

class Troika_Base(nn.Module):
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
        # dtype = None
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

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.clip.visual.output_dim

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)

        # self.cmt = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout) for _ in range(config.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)
        
        ###############
        self.additional_text_params = self.add_text_tunable_params()
        self.additional_prompt_generator = self.add_visual_prompt_generator()
        
        # self.att_qformer = CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.obj_qformer = CrossAttentionLayer(output_dim, output_dim//64, self.cross_attn_dropout)
        # self.additional_prompt_generator_att = self.add_visual_prompt_generator()
        # self.additional_prompt_generator_obj = self.add_visual_prompt_generator()
        self.additional_txt_prompt_generator = self.add_text_prompt_generator()
        self.visual_prompt_num = 2
    
    def add_visual_prompt_generator(self):
        adapter_num = self.clip.visual.transformer.layers
        # params = nn.ModuleList([Concept_Prompt_Generator(d_model=self.clip.visual.transformer.width,att_num=len(self.attributes),obj_num=len(self.classes)) for _ in range(adapter_num)])
        params = nn.ModuleList([Prompt_Generator(d_model=self.clip.visual.transformer.width) for _ in range(adapter_num)])
        return params
    
    def add_text_prompt_generator(self):
        adapter_num = self.clip.transformer.layers
        params = nn.ModuleList([Text_Prompt_Generator(d_model=self.clip.transformer.width) for _ in range(adapter_num)])
        return params
    
    def add_visual_tunable_params(self):
        # adapter_num = 2 * self.clip.visual.transformer.layers
        adapter_num = self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, 
                                    bottleneck=self.config.adapter_dim, 
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params

    def add_text_tunable_params(self):
        # adapter_num = 2 * self.clip.transformer.layers
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

    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        original_x = x

        #########
        prompt = self.additional_em_params[0](x[:, 1:, :]) #self.em_module(x[:, 1:, :]) # 24,4,1024
        
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(prompt),
                x[:, 1:, :]
            ), dim=1)
        #########
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        original_x = original_x.permute(1,0,2)
        if self.config.prompt_deep:
            for i in range(self.clip.visual.transformer.layers):
                if i == 0:
                    x = self.clip.visual.transformer.resblocks[i](x)
                    original_x = self.clip.visual.transformer.resblocks[i](original_x)
                else:
                    if i <= self.clip.visual.transformer.layers:#self.deep_prompt_embeddings.shape[0]:
                        prompt = self.additional_em_params[i](original_x[1:, :, :].permute(1,0,2))#self.em_module(original_x[1:, :, :].permute(1,0,2))
                        # prompt = self.em_module(x[(1+self.config.prompt_num_tokens):, :, :].permute(1,0,2),prompt)
                        deep_prompt_emb = self.prompt_dropout(prompt)
                        
                        x = torch.cat((
                            x[:1, :, :],
                            deep_prompt_emb.permute(1,0,2),
                            x[(1+self.config.prompt_num_tokens):, :, :]
                        ), dim=0)

                    x = self.clip.visual.transformer.resblocks[i](x)
                    original_x = self.clip.visual.transformer.resblocks[i](original_x)
            img_feature = x
        else:
            img_feature = self.clip.visual.transformer(x)
        x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        # x = self.clip.visual.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        # x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

        # x = self.clip.visual.ln_post(x)
        
        # # if self.clip.visual.proj is not None:
        # #     x = x @ self.clip.visual.proj
        
        # #return x[:, 0, :], x
        # ############# add em algorithm ################
        
        # x_em = self.em_module(x)
        return x[:, 0, :], x

        # return x[:, 0, :], x
    
    def visual_adapt(self, x: torch.Tensor, prompt: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        #########
        
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(prompt),
                x[:, 1:, :]
            ), dim=1)
        # pdb.set_trace()
        #########
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)
        x = img_feature.permute(1, 0, 2)  # LND -> NLD # 128,257,1024

        x = self.clip.visual.ln_post(x)
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        
        #return x[:, 0, :], x
        ############# add em algorithm ################
        # x_em = self.em_module(x)

        return x[:, 0, :], x
     
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
            # if i_block==0:
            #     adapter_prompt = self.additional_txt_prompt_generator[i_block](x.type(torch.float))
            #     # x = x + adapter_prompt.repeat(8,1,1)
            #     x = torch.cat([adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
            #     # x = torch.cat([x[0:1, :, :], adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
            # else:
            #     adapter_prompt = self.additional_txt_prompt_generator[i_block](x.type(torch.float))
            #     # x = x + adapter_prompt.repeat(8,1,1)
            #     x = torch.cat([adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
            
            residual = x
            x = self.clip.transformer.resblocks[i_block].attention(
                self.clip.transformer.resblocks[i_block].ln_1(x)
            )
            # x = x + adapt_x + residual
            x = x + residual

            # FFN
            i_adapter = i_block #+ self.clip.transformer.layers
            adapt_x = self.additional_text_params[i_adapter](x.type(torch.float), add_residual=False).type(self.dtype)
            residual = x
            x = self.clip.transformer.resblocks[i_block].mlp(
                self.clip.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            # x = x + residual
        
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
    
    def encode_image_with_adapter(self, x: torch.Tensor, flag=None):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            ###
            # if i_block>0:
            #     deep_prompt_emb = self.prompt_dropout(
            #                     self.deep_prompt_embeddings[i_block-1].expand(BatchSize, -1, -1))                
            #     x = torch.cat((
            #         x[:1, :, :],
            #         deep_prompt_emb.permute(1,0,2),
            #         x[(1+self.num_tokens):, :, :]
            #     ), dim=0)
            # MHA
            # adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            # if flag == 'att':
            #     adapter_prompt = self.additional_prompt_generator_att[i_block](x.type(torch.float))
            # elif flag == 'obj':
            #     adapter_prompt = self.additional_prompt_generator_obj[i_block](x.type(torch.float))
            # elif flag is None:
            # adapter_prompt = self.additional_prompt_generator[i_block](x.type(torch.float))

            # if i_block==0:
            #     # adapter_prompt = self.additional_prompt_generator[i_block](x.type(torch.float))
            #     x = torch.cat([x[0:1, :, :], adapter_prompt.type(self.dtype), x[1:, :, :]], dim=0)
            # else:
                
            #     # adapter_prompt = self.additional_prompt_generator[i_block](x.type(torch.float))
            #     x = torch.cat([x[0:1, :, :], adapter_prompt.type(self.dtype), x[(1+self.visual_prompt_num):, :, :]], dim=0)#, x[2:, :, :]], dim=0)#x[(1+self.visual_prompt_num):, :, :]], dim=0)
            
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            # x = x + adapt_x + residual
            x = x + residual

            # FFN
            i_adapter = i_block #+ self.clip.visual.transformer.layers
            # adapt_x, adapter_prompt = self.additional_visual_params[i_adapter](x, add_residual=False)
            adapt_x = self.additional_visual_params[i_adapter](x.type(torch.float), add_residual=False).type(self.dtype)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            # x = x + residual
            # if i_block==0:
            #     x = torch.cat([x[0:1, :, :], adapter_prompt, x[1:, :, :]], dim=0)
            # # elif i_block==self.clip.visual.transformer.layers-1:
            # #     x = x
            # else:
            #     x = torch.cat([x[0:1, :, :], adapter_prompt, x[2:, :, :]], dim=0)


        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature_ori = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature_ori @ self.clip.visual.proj

        return img_feature[:, 0, :], img_feature


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
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    
    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img.type(torch.float)).type(self.dtype), self.obj_disentangler(batch_img.type(torch.float)).type(self.dtype)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]

            # CMT
            cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
            batch_patch = self.patch_norm(batch_patch)
            for layer in self.cmt:
                cmt_text_features = layer(cmt_text_features, batch_patch)
            cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

            cmt_text_features = cmt_text_features / cmt_text_features.norm(
                dim=-1, keepdim=True
            )

            logits.append(
                torch.einsum(
                    "bd, bkd->bk", 
                    normalized_img_features[i_element], 
                    cmt_text_features * self.clip.logit_scale.exp()
            ))
        return logits

    
    def forward(self, batch, idx, training=True, pairs=None):
        batch_img = batch[0].cuda()
        
        # labels = [batch[3].cuda(),batch[1].cuda(),batch[2].cuda()]
        # b = batch_img.shape[0]
        # l, _ = idx.shape

        batch_img, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype),flag=None)
        # batch_img_att, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype),flag='att')
        # batch_img_obj, batch_patch = self.encode_image_with_adapter(batch_img.type(self.clip.dtype),flag='obj')
        # batch_img, batch_patch = self.visual(batch_img.type(self.clip.dtype)) #batch_img.shape = [bs,768]; batch_patch.shape = [16,257,768]
        
        # batch_img_adapt, batch_patch_adapt = self.visual_adapt(batch_img.type(self.clip.dtype), prompt)
        # batch_img_features = [batch_img_adapt, self.attr_disentangler(batch_img_adapt), self.obj_disentangler(batch_img_adapt)]
        # batch_img_features = [batch_img, batch_patch[:,1,:], batch_patch[:,2,:]]
        # batch_img_features = [batch_img, self.attr_disentangler(batch_patch[:,1,:].type(torch.float)).type(self.dtype), self.obj_disentangler(batch_patch[:,2,:].type(torch.float)).type(self.dtype)]
        batch_img_features = [batch_img, self.attr_disentangler(batch_img).type(self.dtype), self.obj_disentangler(batch_img).type(self.dtype)]
        # batch_img_features = [batch_img_com, batch_img_att.type(self.dtype), batch_img_obj.type(self.dtype)]
        
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        token_tensors = self.construct_token_tensors(idx)
        
        
        logits = list()
        all_text_feat = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text_with_adapter(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            
            logits.append(torch.matmul(normalized_img_features[i_element],self.clip.logit_scale.exp()*idx_text_features.T))
            
            all_text_feat.append(idx_text_features)
        
        if training and pairs is not None:
            zero_shot_image, _ = self.zero_shot_visual(batch[0].cuda().type(self.clip.dtype))
            zs_image = zero_shot_image/zero_shot_image.norm(dim=-1, keepdim=True)
            zs_pair_feat, zs_att_feat, zs_obj_feat = self.zero_shot_text(pairs)
            loss_image = F.l1_loss(normalized_img_features[0], zs_image.cuda(),reduction='mean') * 10
            loss_pair = 25 * F.l1_loss(all_text_feat[0],zs_pair_feat.cuda(),reduction='mean')
            loss_att = 25 * F.l1_loss(all_text_feat[1],zs_att_feat.cuda(),reduction='mean')
            loss_obj = 25 * F.l1_loss(all_text_feat[2],zs_obj_feat.cuda(),reduction='mean')
           
            # loss_contra = soft_contrastive_loss(batch_patch[:,1,:], batch[1].cuda(),0.1) + soft_contrastive_loss(batch_patch[:,2,:], batch[2].cuda(),0.1)
            
            return logits, loss_image+loss_pair+loss_att+loss_obj #+ 0.1*loss_contra
        
        else:
            return logits