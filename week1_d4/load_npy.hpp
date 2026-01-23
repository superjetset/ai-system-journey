#pragma once
#include <vector>
#include <fstream>
#include <cstring>
#include <iostream>

// 极简 .npy 读取器（仅支持 float32）
std::vector<float> load_npy(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "❌ 文件打开失败: " << filename << std::endl;
        exit(1);
    }
    
    // 读取文件头（跳过）
    char header[256];
    file.read(header, 128);
    
    // 计算数据偏移（找到 '\n' 后）
    size_t offset = 0;
    for (int i = 0; i < 128; ++i) {
        if (header[i] == '\n') {
            offset = i + 1;
            break;
        }
    }
    file.seekg(offset);
    
    // 读取数据（float32 = 4 字节）
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t num_floats = (file_size - offset) / 4;
    
    std::vector<float> data(num_floats);
    file.seekg(offset);
    file.read(reinterpret_cast<char*>(data.data()), num_floats * 4);
    
    std::cout << "✅ 加载 " << filename << ": " << num_floats << " 个 float" << std::endl;
    return data;
}
