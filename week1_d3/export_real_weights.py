import torch
import numpy as np

# 加载模型
model = torch.load("/root/autodl-tmp/opt-125m-test/pytorch_model.bin", map_location="cpu")

# 导出第一层 Attention 的 Q/K/V 投影权重
# 维度: (768, 768)
q_proj = model['model.decoder.layers.0.self_attn.q_proj.weight'].numpy()
k_proj = model['model.decoder.layers.0.self_attn.k_proj.weight'].numpy()
v_proj = model['model.decoder.layers.0.self_attn.v_proj.weight'].numpy()

# 保存为 .npy 格式（C++ 易读取）
np.save("q_proj.npy", q_proj)
np.save("k_proj.npy", k_proj)
np.save("v_proj.npy", v_proj)

print("✅ 真实权重导出成功！")
print(f"Q/K/V 投影矩阵维度: {q_proj.shape}")
