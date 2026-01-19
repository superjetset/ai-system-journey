import torch
import numpy as np

# 加载模型
model = torch.load("/root/autodl-tmp/opt-125m-test/pytorch_model.bin", map_location="cpu")

# 导出第一个权重矩阵 (embedding)
embedding = model['model.decoder.embed_tokens.weight'].numpy()
print(f"Embedding shape: {embedding.shape}")  # 预期: (50272, 768)

# 保存为二进制文件
with open("embedding.bin", "wb") as f:
    # 先写 shape
    np.array(embedding.shape, dtype=np.int32).tofile(f)
    # 再写数据
    embedding.tofile(f)

print("✅ 权重导出成功！文件大小: {:.2f} MB".format(embedding.nbytes / 1e6))
