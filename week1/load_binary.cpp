#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

int main() {
    const char* filename = "embedding.bin";
    
    // 打开文件
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open() && "❌ 文件打开失败");
    
    // 读取 shape
    int32_t shape[2];
    file.read(reinterpret_cast<char*>(shape), sizeof(shape));
    std::cout << "✅ 读取到权重维度: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
    
    // 读取权重数据
    size_t num_elements = shape[0] * shape[1];
    std::vector<float> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
    
    // 验证前 5 个值
    std::cout << "前5个权重值: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    file.close();
    std::cout << "✅ C++ 加载成功！" << std::endl;
    return 0;
}
