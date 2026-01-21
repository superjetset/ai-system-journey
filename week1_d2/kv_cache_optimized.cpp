#include <iostream>
#include <vectorr>
#include <chrono>
#include <cstring>


struct KVCacheOptimized {

	float* k_buffer;      // 使用原生指针
	float* v_buffer; 
	int max_seq_len;   // 最大容量，比如如2048
	int current_seq_len;
	int hidden_dim;   // heads * head_dim

	KVCacheOptimized( int max_len, int dim ) : max_seq_len(max_len), hidden_dim(dim), current_seq_len(0) {
		// 1.预分配，一次申请够，避免运行中 realloc
		k_buffer = new float[ max_seq_len * hidden_dim ];
		v_buffer = new float[ max_seq_len * hidden_dim ];
		
		std::cout << "[系统]预分配现存： "<<( max_seq_len * hidden_dim * 2 * 4 ) / 1024.0 / 1024.0 << " MB " <<std::endl;
	}

	~KVCacheOptimized() {
		delete[] k_buffer;
		delete[] v_buffer;
	}


     void append_and_attend(const std::vector<float>& new_k, const std::vector<float>& new_v) {
		if (current_seq_len >= max_seq_len) return;
		
		   // 2. 极速写入：使用 memcpy 替代 insert
			// 模拟 vLLM 的 Block 写入 (这里简化为连续写入)
			int offset = current_seq_len * hidden_dim;
			std::memcpy(k_buffer + offset, new_k.data(), new_k.size() * sizeof(float));
			std::memcpy(v_buffer + offset, new_v.data(), new_v.size() * sizeof(float));
			
			current_seq_len++;
			
		    // 3. 模拟 Attention 计算 (Memory Bound)
			// 随着 current_seq_len 变长，我们要遍历的数据越多
			// 这里只是象征性地读一遍内存，模拟 GPU 读显存的开销
			volatile float dummy_sum = 0; // volatile 防止编译器优化掉
			for(int i=0; i < current_seq_len * hidden_dim; ++i) {
				 dummy_sum += k_buffer[i]; // 强制 CPU 读内存
			}
	}
}


// 模拟生成 1 个 Token 的 K/V
void generate_token(KVCacheOptimized& cache, int token_id) {
    // 模拟计算：随机生成新 K/V
    std::vector<float> new_k(hidden_dim);
    std::vector<float> new_v(hidden_dim);
    
    // 填充数据（实际是用矩阵乘法计算）
    std::fill(new_k.begin(), new_k.end(), token_id * 0.01f);
    std::fill(new_v.begin(), new_v.end(), token_id * 0.02f);
    
    // 追加到 Cache
    cache.append_and_attend(new_k, new_v);
}

int main() {
    // 初始化：batch=1, heads=12, head_dim=64
    KVCacheOptimized cache(12, 64);
    
    std::cout << "=== KV Cache 性能测试 ===" << std::endl;
    
    // 模拟生成 100 个 Token
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        generate_token(cache, i);
        
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n✅ 生成 100 个 Token 完成！" << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "Seq Length: " << cache.seq_len << std::endl;
    
    return 0;
}