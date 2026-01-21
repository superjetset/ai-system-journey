# export_real_weights.py
import torch
import numpy as np

model = torch.load("/root/autodl-tmp/opt-125m-test/pytorch_model.bin", map_location="cpu")

# 导出 Q/K/V 投影权重
q_proj = model['model.decoder.layers.0.self_attn.q_proj.weight'].numpy()  # (768, 768)
k_proj = model['model.decoder.layers.0.self_attn.k_proj.weight'].numpy()
v_proj = model['model.decoder.layers.0.self_attn.v_proj.weight'].numpy()

np.save("q_proj.npy", q_proj)
np.save("k_proj.npy", k_proj)
np.save("v_proj.npy", v_proj)

print("真实权重导出成功！")
