#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cassert>

// 模拟 shape: (batch, heads, seq_len, head_dim)
struct KVCache {
    std::vector<float> k;
    std::vector<float> v;
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    
    // 初始化空 Cache
    KVCache(int b, int h, int d) : batch_size(b), num_heads(h), seq_len(0), head_dim(d) {}
    
    // 追加新的 K/V (增量更新)
    void append(const std::vector<float>& new_k, const std::vector<float>& new_v) {
        assert(new_k.size() == batch_size * num_heads * 1 * head_dim);
        
        k.insert(k.end(), new_k.begin(), new_k.end());
        v.insert(v.end(), new_v.begin(), new_v.end());
        seq_len += 1;
    }
    
    // 获取当前 Cache 的内存占用 (MB)
    float memory_mb() const {
        size_t bytes = (k.size() + v.size()) * sizeof(float);
        return bytes / 1024.0 / 1024.0;
    }
};

// 模拟生成 1 个 Token 的 K/V
void generate_token(KVCache& cache, int token_id) {
    // 模拟计算：随机生成新 K/V
    std::vector<float> new_k(cache.batch_size * cache.num_heads * cache.head_dim);
    std::vector<float> new_v(cache.batch_size * cache.num_heads * cache.head_dim);
    
    // 填充数据（实际是用矩阵乘法计算）
    std::fill(new_k.begin(), new_k.end(), token_id * 0.01f);
    std::fill(new_v.begin(), new_v.end(), token_id * 0.02f);
    
    // 追加到 Cache
    cache.append(new_k, new_v);
}

int main() {
    // 初始化：batch=1, heads=12, head_dim=64
    KVCache cache(1, 12, 64);
    
    std::cout << "=== KV Cache 性能测试 ===" << std::endl;
    
    // 模拟生成 100 个 Token
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        generate_token(cache, i);
        
        if (i % 20 == 0) {
            std::cout << "Token " << i << " | 内存占用: " 
                      << cache.memory_mb() << " MB" << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n✅ 生成 100 个 Token 完成！" << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "最终内存占用: " << cache.memory_mb() << " MB" << std::endl;
    std::cout << "Seq Length: " << cache.seq_len << std::endl;
    
    return 0;
}
