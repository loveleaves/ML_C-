# NCNN
- ncnn is a high-performance neural network **inference** framework optimized for the **mobile platform**
- [github](https://github.com/Tencent/ncnn)

## References
- https://www.cnblogs.com/Moonjou/p/16471048.html
- https://zhuanlan.zhihu.com/p/449765328
- [NCNN部署](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch17_%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E3%80%81%E5%8A%A0%E9%80%9F%E5%8F%8A%E7%A7%BB%E5%8A%A8%E7%AB%AF%E9%83%A8%E7%BD%B2/17.8.1%20NCNN%E9%83%A8%E7%BD%B2.md)
- [docs](https://ncnn.readthedocs.io/en/latest/how-to-build/how-to-build.html#)

## 内存设计
### 内存对齐管理
```
// 存储矩阵数据按channel对齐
cstep = alignSize(w * h * sizeof(float), 16) >> 2;

// 申请内存对齐
#define MALLOC_ALIGN    16
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}
static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata; // adata指向对齐地址，adata[-1]指向malloc地址
}
```

### 引用计数
```
int refcount;
size_t totalsize = total() * sizeof(float);
data = (float*)fastMalloc(totalsize + (int)sizeof(*refcount));
refcount = (int*)(((unsigned char*)data) + totalsize);
*refcount = 1; // 数据后一位存引用计数，初始为1
NCNN_XADD(refcount, 1); // 原子操作对计数器加1，NCNN_XADD平台相关
```

## 模型部署&加速
### 量化
```
// fp16数据读取，符号位、指数、尾数，浮点数规格化和特殊值处理
unsigned short sign = (value & 0x8000) >> 15;
unsigned short exponent = (value & 0x7c00) >> 10;
unsigned short significand = value & 0x03FF;
// 1 : 8 : 23
union
{
    unsigned int u;
    float f;
} tmp; // union数据转换
```

### SIMD加速
```
// 移动端多为ARM NEON
intrinsic + assembly
```

### SIMT加速
```
// openMP
dynamic_current = omp_get_dynamic();         // 保存当前的动态线程状态
num_threads_current = omp_get_num_threads(); // 获取当前的线程数
omp_set_dynamic(0);                          // 禁用动态线程分配
omp_set_num_threads(num_threads);            // 设置为指定的线程数
    ***operation ***                         // 并行操作
omp_set_dynamic(dynamic_current);            // 恢复原来的动态线程分配状态
omp_set_num_threads(num_threads_current);    // 恢复原来的线程数
```

### 工程构建
```
// .in文件通过构建命令自动生成各平台代码，交叉编译
#cmakedefine01 NCNN_STDIO // platform.h.in文件内容
configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h) // cmakelist文件
set(NCNN_STDIO ON) // cmakelist文件
#define NCNN_STDIO 1 // 生成platform.h文件内容
```

### 模型文件加密
```
// ncnn2mem 工具对模型结构param加密
load_param_bin() // 加载加密的二进制param文件
```