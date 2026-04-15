#include <stdio.h>
__global__ void kernel(float *out) {
    int tid = threadIdx.x;
    out[tid] = (float)tid;
}
int main() {
    float *d; cudaMalloc(&d, 1024);
    kernel<<<1,256>>>(d);
    float h[256]; cudaMemcpy(h, d, 1024, cudaMemcpyDeviceToHost);
    printf("%f %f %f %f\n", h[0], h[1], h[2], h[3]);
}
