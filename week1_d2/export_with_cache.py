import torch
import numpy as np

# 加载模型（这次加载完整以便演示）
model = torch.load("/root/autodl-tmp/opt-125m-test/pytorch_model.bin", map_location="cpu")

# 导出第一个 Attention 层的 K/V 投影权重
# 目的：在 C++ 里模拟推理时的 K/V 计算
k_proj = model['model.decoder.layers.0.self_attn.k_proj.weight'].numpy()  # shape: (768, 768)
v_proj = model['model.decoder.layers.0.self_attn.v_proj.weight'].numpy()

# 保存
np.save("k_proj.npy", k_proj)
np.save("v_proj.npy", v_proj)

# 模拟生成 10 个 Token 的 K/V Cache
# 每个 Token 生成后，K/V 维度增加 1
cache_k = np.random.randn(1, 12, 10, 64).astype(np.float32)  # (batch, heads, seq_len, head_dim)
cache_v = np.random.randn(1, 12, 10, 64).astype(np.float32)

# 保存 Cache
np.save("cache_k.npy", cache_k)
np.save("cache_v.npy", cache_v)

print("✅ 权重 + Cache 导出成功！")
print(f"K_Proj shape: {k_proj.shape}")
print(f"Cache_K shape: {cache_k.shape}")
