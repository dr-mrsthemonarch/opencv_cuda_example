#include <assert.h>
#include <iostream>

inline cudaError_t cudaCheck(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


void initWith(float num, float *a, int N)
{
    for(int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

void addVectorsInto(float *result, float *a, float *b, int N)
{
    for(int i = 0; i < N; ++i)
    {
        result[i] = a[i] + b[i];
    }
}

__global__
void cudaAddVectorsInto(float *result, float *a, float *b, int N)
{
    int grid_stride = gridDim.x*blockDim.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    for(int i = idx; i < N; i+= grid_stride)
    {
        result[i] = a[i] + b[i];
    }

}


void checkElementsAre(float target, float *array, int N)
{
    for(int i = 0; i < N; i++)
    {
        if(array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2<<20;
    size_t size = N * sizeof(float);
    //cudaError_t err;
    float *a;
    float *b;
    float *c;

    cudaCheck(cudaMallocManaged(&a,size));
    cudaMallocManaged(&b,size);
    cudaMallocManaged(&c,size);


    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    int number_of_threads = 256;
    int number_of_blocks = (N+number_of_threads-1)/number_of_threads;

    cudaAddVectorsInto<<<number_of_blocks,number_of_threads>>>(c, a, b, N);
    cudaCheck(cudaGetLastError());
    //std::cout << cudaGetErrorString(err) << std::endl;
    cudaCheck(cudaDeviceSynchronize());
    checkElementsAre(7, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}