// file:quantize.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

// 对称量化: 把 float 压缩到 0-15 (INT4 范围)
void quantize_symmetric(const float* fp16_data, uint8_t* int4_data, int size, float& scale) {
    // 找最大值
    float max_abs = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_abs = std::max(max_abs, std::abs(fp16_data[i]));
    }
    
    // 计算缩放因子（反量化时用）
    scale = max_abs / 7.0f;  // INT4 范围 [-7, 7]，留 1 个 bit 给符号
    
    // 量化
    for (int i = 0; i < size; ++i) {
        int8_t quantized = static_cast<int8_t>(std::round(fp16_data[i] / scale));
        // 用 int 计算，再转回 int8_t
        int temp = std::round(fp16_data[i] / scale);
        temp = std::max(-7, std::min(7, temp));
        quantized = static_cast<int8_t>(temp);
        
        // 打包两个 INT4 到一个 uint8 (高 4 位 + 低 4 位)
        if (i % 2 == 0) {
            int4_data[i/2] = (quantized & 0x0F) << 4;
        } else {
            int4_data[i/2] |= (quantized & 0x0F);
        }
    }
}

// 反量化
void dequantize_symmetric(const uint8_t* int4_data, float* fp16_data, int size, float scale) {
    for (int i = 0; i < size; ++i) {
        int8_t quantized;
        if (i % 2 == 0) {
            quantized = (int4_data[i/2] >> 4) & 0x0F;
        } else {
            quantized = int4_data[i/2] & 0x0F;
        }
        
        // 符号扩展（4位有符号数 → 8位有符号数）
        if (quantized & 0x08) quantized |= 0xF0;
        
        fp16_data[i] = quantized * scale;
    }
}

int main() {
    // 测试数据 (8 个 FP16 数值)
    std::vector<float> fp16_data = {0.123, -0.456, 0.789, -0.234, 0.567, -0.891, 0.345, -0.678};
    int size = fp16_data.size();
    
    // INT4 打包后只需要一半字节
    std::vector<uint8_t> int4_data(size / 2);
    float scale;
    
    // 量化
    quantize_symmetric(fp16_data.data(), int4_data.data(), size, scale);
    std::cout << "✅ 量化完成！原大小: " << size * 4 << " 字节 → INT4: " << int4_data.size() << " 字节" << std::endl;
    std::cout << "缩放因子: " << scale << std::endl;
    
    // 反量化验证
    std::vector<float> fp16_restored(size);
    dequantize_symmetric(int4_data.data(), fp16_restored.data(), size, scale);
    
    // 计算误差
    float max_error = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_error = std::max(max_error, std::abs(fp16_data[i] - fp16_restored[i]));
    }
    std::cout << "最大量化误差: " << max_error << std::endl;
    std::cout << "相对误差: " << (max_error / scale * 100) << "%" << std::endl;
    
    return 0;
}
