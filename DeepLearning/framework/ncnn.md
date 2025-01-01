# NCNN
- ncnn is a high-performance neural network inference framework optimized for the mobile platform
- [github](https://github.com/Tencent/ncnn)

## References
- https://www.cnblogs.com/Moonjou/p/16471048.html
- https://zhuanlan.zhihu.com/p/449765328

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
```