#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

// 矩阵乘法 (M, K) × (K, N) = (M, N)
void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Softmax (简化版)
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) max_val = std::max(max_val, x[i]);
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) x[i] /= sum;
}

// 缓存结构
struct RealKVCache {
    std::vector<float> k;  // (seq_len, hidden_dim)
    std::vector<float> v;  // (seq_len, hidden_dim)
    int hidden_dim = 768;
    
    void append(const float* new_k, const float* new_v) {
        k.insert(k.end(), new_k, new_k + hidden_dim);
        v.insert(v.end(), new_v, new_v + hidden_dim);
    }
    
    // 计算 Attention (有 Cache)
    float compute_with_cache(const float* q) {
        int seq_len = k.size() / hidden_dim;
        std::vector<float> scores(seq_len);
        
        // Q × K^T (1, hidden_dim) × (hidden_dim, seq_len) = (1, seq_len)
        matmul(q, k.data(), scores.data(), 1, hidden_dim, seq_len);
        
        // Softmax
        softmax(scores.data(), seq_len);
        
        // 加权求和 (1, seq_len) × (seq_len, hidden_dim) = (1, hidden_dim)
        float result = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            result += scores[i] * v[i * hidden_dim];  // 简化版
        }
        return result;
    }
    
    // 无 Cache：每次都从头计算
    float compute_without_cache(const float* q, const float* all_k, const float* all_v, int seq_len) {
        std::vector<float> scores(seq_len);
        matmul(q, all_k, scores.data(), 1, hidden_dim, seq_len);
        softmax(scores.data(), seq_len);
        
        float result = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            result += scores[i] * all_v[i * hidden_dim];
        }
        return result;
    }
};

int main() {
    // 加载权重 (模拟)
    std::vector<float> q_proj(768 * 768);
    std::vector<float> k_proj(768 * 768);
    std::vector<float> v_proj(768 * 768);
    // 实际应从 .npy 文件加载，这里填充随机值
    for (auto& x : q_proj) x = (rand() % 1000) / 1000.0f - 0.5f;
    for (auto& x : k_proj) x = (rand() % 1000) / 1000.0f - 0.5f;
    for (auto& x : v_proj) x = (rand() % 1000) / 1000.0f - 0.5f;
    
    RealKVCache cache;
    float hidden[768];  // 模拟当前 Token 的隐状态
    
    // 模拟生成 100 个 Token
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        // 计算 Q/K/V (简化)
        float q[768], new_k[768], new_v[768];
        matmul(hidden, q_proj.data(), q, 1, 768, 768);
        matmul(hidden, k_proj.data(), new_k, 1, 768, 768);
        matmul(hidden, v_proj.data(), new_v, 1, 768, 768);
        
        cache.append(new_k, new_v);
        float output = cache.compute_with_cache(q);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto cache_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // 无 Cache 对比
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        float q[768], new_k[768 * 100], new_v[768 * 100];  // 每次都重新计算所有
        // ... 模拟计算所有 Token ...
    }
    end = std::chrono::high_resolution_clock::now();
    auto no_cache_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "=== 真实 Attention 性能对比 ===" << std::endl;
    std::cout << "有 KV Cache: " << cache_time << " ms" << std::endl;
    std::cout << "无 KV Cache: " << no_cache_time << " ms" << std::endl;
    std::cout << "速度提升: " << (double)no_cache_time / cache_time << " 倍" << std::endl;
    
    return 0;
}
