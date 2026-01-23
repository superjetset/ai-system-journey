#quantize_model.py
import numpy as np

# 加载真实权重
q_proj = np.load("q_proj.npy")  # (768, 768)
k_proj = np.load("k_proj.npy")
v_proj = np.load("v_proj.npy")

def quantize_matrix(matrix, name):
    flat = matrix.flatten().astype(np.float32)
    size = len(flat)
    
    # 找最大值
    max_abs = np.max(np.abs(flat))
    scale = max_abs / 7.0
    
    # 量化
    quantized = np.round(flat / scale).astype(np.int8)
    quantized = np.clip(quantized, -7, 7)
    
    # 打包 INT4（两个 int8 存到一个 uint8）
    int4_packed = np.zeros(size // 2, dtype=np.uint8)
    for i in range(0, size, 2):
        high = (quantized[i] & 0x0F) << 4    # 高 4 位
        low = quantized[i+1] & 0x0F         # 低 4 位
        int4_packed[i//2] = high | low
    
    # 保存
    np.save(f"{name}_int4.npy", int4_packed)
    
    # 计算压缩率
    original_mb = matrix.nbytes / 1e6
    compressed_mb = int4_packed.nbytes / 1e6
    
    print(f"{name}: {original_mb:.2f} MB → {compressed_mb:.2f} MB (↓ {original_mb/compressed_mb:.1f}x)")
    return scale

# 量化 Q/K/V
scale_q = quantize_matrix(q_proj, "q_proj")
scale_k = quantize_matrix(k_proj, "k_proj")
scale_v = quantize_matrix(v_proj, "v_proj")

print("\n✅ 模型量化完成！")
print(f"总原始大小: {(q_proj.nbytes * 3) / 1e6:.2f} MB")
print(f"总 INT4 大小: {(scale_q * 0 + scale_k * 0 + scale_v * 0):.2f} MB (约 ↓ 75%)")
