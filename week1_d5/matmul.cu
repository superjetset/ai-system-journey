#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>


// CPU 矩阵乘法
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N){
    for(int i = 0; i<M; ++i ){
        for( int j=0; j<N; ++j ){
            float sum =0.0f;
            for( int k=0; k<K; ++k ){
                sum += A[i*K +k] * B[k*N +j];
            }
            C[i*N +j] = sum;
        }
    }
}

// GPU kernel: 每个线程计算C的一个元素
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N){
        float sum = 0.0f;
        for (int k = 0; k < K; ++k){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// GPU矩阵乘法封装
void matmul_gpu(const float* h_A, const float* h_B, float* h_C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    std::cout << "GPU matmul_gpu enter!" << std::endl;
    // 1. 预热 (Warm-up)
    // 随便调个 API，强制 GPU 完成初始化，避免把启动时间算进测试里
    cudaFree(0); 

    // 分配 GPU 内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // 数据拷贝：CPU → GPU
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // 配置 Kernel 启动参数
    dim3 threadsPerBlock(16, 16);  // 每个 block 256 线程
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    
    // ========== GPU 核心计算阶段 (只计这一段!) ==========
    cudaDeviceSynchronize(); // 确保上面都做完了
    auto start_kernel = std::chrono::high_resolution_clock::now();

    // 启动 Kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    
    // 等待 GPU 完成
    cudaDeviceSynchronize();
    
    auto end_kernel = std::chrono::high_resolution_clock::now();

        // 计算耗时
    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel).count();
    std::cout << "GPU Kernel 纯计算耗时: " << kernel_time / 1000.0 << " ms" << std::endl;

    // 结果拷贝：GPU → CPU
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // 测试：768×768 矩阵乘法（与 opt-125m 的投影矩阵同尺寸）
    int M = 768, K = 768, N = 768;
    
    std::vector<float> A(M * K, 0.5f);
    std::vector<float> B(K * N, 0.3f);
    std::vector<float> C_cpu(M * N);
    std::vector<float> C_gpu(M * N);
    
    // CPU 计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(A.data(), B.data(), C_cpu.data(), M, K, N);
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_cpu).count();
    
    // GPU 计算
    auto start_gpu = std::chrono::high_resolution_clock::now();
    matmul_gpu(A.data(), B.data(), C_gpu.data(), M, K, N);
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_gpu).count();
    
    // 验证结果正确性
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = std::max(max_error, std::abs(C_cpu[i] - C_gpu[i]));
    }
    
    std::cout << "\n=== CUDA 首战告捷 ===" << std::endl;
    std::cout << "CPU 耗时: " << cpu_time << " ms" << std::endl;
    //std::cout << "GPU 耗时: " << gpu_time << " ms" << std::endl;
    //std::cout << "加速比: " << (double)cpu_time / gpu_time << " 倍" << std::endl;
    //std::cout << "最大误差: " << max_error << std::endl;
    
    return 0;
}
