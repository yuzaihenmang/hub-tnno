'''
deepseek
'''

# 一、输入层：词嵌入
# 位置：DeepseekV3Model.forward 函数内（约第 1440-1450 行）
inputs_embeds = self.embed_tokens(input_ids)  # 词嵌入映射

# 二、注意力层：投影→RoPE→注意力计算→输出投影
# 1. Q/K/V 投影与拆分（位置：DeepseekV3Attention.forward 函数内，约第 700-730 行）
q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))  # Q 投影（含 LoRA）
q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

# 2. RoPE 编码应用（位置：DeepseekV3Attention.forward 函数内，约第 740-750 行）
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

# 3. 拼接 Q/K 并计算注意力（位置：DeepseekV3Attention.forward 函数内，约第 755-775 行）
query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states)

# 4. 注意力输出投影（位置：DeepseekV3Attention.forward 函数内，约第 780-785 行）
attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
attn_output = self.o_proj(attn_output)

# 三、归一化层：Pre-LN 执行
# 1. 注意力层前归一化（位置：DeepseekV3DecoderLayer.forward 函数内，约第 1000 行）
hidden_states = self.input_layernorm(hidden_states)

# 2. 注意力层后残差+归一化（位置：DeepseekV3DecoderLayer.forward 函数内，约第 1010-1015 行）
hidden_states = residual + hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)

# 四、MLP/MoE 层：特征变换
# （一）普通 MLP 层
# 位置：DeepseekV3MLP.forward 函数内（约第 580 行）
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
# （二）MoE 层
# 1. 门控路由（位置：MoEGate.forward 函数内，约第 630-660 行）
logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
scores = logits.sigmoid()
_, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
topk_weight = scores.gather(1, topk_idx)

# 2. 专家计算与结果聚合（位置：DeepseekV3MoE.forward 函数内，约第 800-830 行）
y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
if self.config.n_shared_experts is not None:
    y = y + self.shared_experts(identity)
