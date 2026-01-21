#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include "load_npy.hpp"

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

// Softmax
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

// KV Cache 结构
struct RealKVCache {
    std::vector<float> k;  // 展平: (seq_len, hidden_dim)
    std::vector<float> v;  // 展平: (seq_len, hidden_dim)
    int hidden_dim = 768;
    
    void append(const float* new_k, const float* new_v) {
        k.insert(k.end(), new_k, new_k + hidden_dim);
        v.insert(v.end(), new_v, new_v + hidden_dim);
    }
    
    // 带 Cache 的 Attention 计算
    float compute_with_cache(const float* q, const float* k_proj, const float* v_proj) {
        int seq_len = k.size() / hidden_dim;
        
        // Q × K^T (1, 768) × (768, seq_len) = (1, seq_len)
        std::vector<float> scores(seq_len);
        matmul(q, k.data(), scores.data(), 1, hidden_dim, seq_len);
        
        // Softmax
        softmax(scores.data(), seq_len);
        
        // ×V (1, seq_len) × (seq_len, 768) = (1, 768)
        std::vector<float> output(hidden_dim);
        matmul(scores.data(), v.data(), output.data(), 1, seq_len, hidden_dim);
        
        // 返回第一个值作为简化指标
        return output[0];
    }
};

int main() {
    // 加载权重
    auto q_proj = load_npy("q_proj.npy");
    auto k_proj = load_npy("k_proj.npy");
    auto v_proj = load_npy("v_proj.npy");
    
    // 模拟输入隐状态 (768 维)
    std::vector<float> hidden(768);
    for (int i = 0; i < 768; ++i) hidden[i] = (i % 100) / 100.0f;  // 填充测试数据
    
    RealKVCache cache;
    
    // ========== 有 KV Cache ==========
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        // 计算 Q/K/V
        std::vector<float> q(768), new_k(768), new_v(768);
        matmul(hidden.data(), q_proj.data(), q.data(), 1, 768, 768);
        matmul(hidden.data(), k_proj.data(), new_k.data(), 1, 768, 768);
        matmul(hidden.data(), v_proj.data(), new_v.data(), 1, 768, 768);
        
        // 追加到 Cache 并计算
        cache.append(new_k.data(), new_v.data());
        float result = cache.compute_with_cache(q.data(), k_proj.data(), v_proj.data());
    }
    
    auto cache_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    // ========== 无 Cache（每次都重算）==========
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        // 每次都重新计算所有 Token 的 K/V（模拟 O(n²) 复杂度）
        int total_tokens = i + 1;
        std::vector<float> all_k(total_tokens * 768);
        std::vector<float> all_v(total_tokens * 768);
        
        // 填充数据（模拟重复计算）
        for (int j = 0; j < total_tokens; ++j) {
            matmul(hidden.data(), k_proj.data(), all_k.data() + j * 768, 1, 768, 768);
            matmul(hidden.data(), v_proj.data(), all_v.data() + j * 768, 1, 768, 768);
        }
        
        // 计算 Attention（每次都从头算）
        std::vector<float> q(768);
        matmul(hidden.data(), q_proj.data(), q.data(), 1, 768, 768);
        // ... 这里会计算 Q × all_k^T，复杂度 O(n²) ...
    }
    
    auto no_cache_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    // ========== 结果对比 ==========
    std::cout << "\n=== 真实权重性能对决 ===" << std::endl;
    std::cout << "有 KV Cache: " << cache_time << " ms" << std::endl;
    std::cout << "无 Cache:    " << no_cache_time << " ms" << std::endl;
    std::cout << "速度提升:   " << (double)no_cache_time / cache_time << " 倍" << std::endl;
    
    return 0;
}
