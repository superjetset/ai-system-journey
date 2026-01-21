#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

// 模拟不带 Cache：每次重新计算所有 Token 的 K/V
void generate_without_cache(int total_tokens) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // 每次生成都从头算
    for (int i = 0; i < total_tokens; ++i) {
        // 模拟计算所有 Token 的 K/V（复杂度 O(n²)）
        size_t compute_size = 1 * 12 * (i + 1) * 64;
        std::vector<float> temp(compute_size);  // 临时分配内存
        std::fill(temp.begin(), temp.end(), 0.0f);  // 模拟计算
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "无 Cache 生成 " << total_tokens << " 个 Token: " 
              << duration.count() << " ms" << std::endl;
}

int main() {
    generate_without_cache(100);
    return 0;
}
