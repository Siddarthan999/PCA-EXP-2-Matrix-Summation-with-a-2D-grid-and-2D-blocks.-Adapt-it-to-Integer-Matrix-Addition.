# PCA-EXP-2-Matrix-Summation-with-a-2D-grid-and-2D-blocks.-Adapt-it-to-Integer-Matrix-Addition.
## Aim:
To perform Matrix Summation with a 2D grid and 2D blocks. And adapt it to Integer Matrix Addition.
## Procedure:
* Define the size of the matrix by setting the values of nx and ny, and calculate the total number of elements, nxy.
* Allocate memory on both the host and device for the input and output matrices using the appropriate data types.
* Initialize the input matrices with data on the host side and transfer them to the device memory using the cudaMemcpy function.
* Define the block and grid dimensions for the kernel. The block dimensions should be a 2D array with each element representing the number of threads in each dimension. The grid dimensions should be calculated using the formula (ceil(nx/block.x), ceil(ny/block.y)).
* Launch the kernel with the input and output matrices as arguments using the <<<grid, block>>> notation. Wait for the kernel to finish executing using the cudaDeviceSynchronize function.
* Transfer the output matrix from device to host memory using the cudaMemcpy function.
* Check the correctness of the output matrix by comparing it to the expected result on the host side.
* Free the memory on both the host and device using the appropriate functions. Reset the device using the cudaDeviceReset function.
## Program:
sumMatrixOnGPU-2D-grid-2D-block.cu:

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <cuda_runtime.h>

    __global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx, int ny)
    {
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        int idx = iy * nx + ix;

        if (ix < nx && iy < ny) {
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }

    void sumMatrixOnHost(int *MatA, int *MatB, int *MatC, int nx, int ny)
    {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int idx = iy * nx + ix;
                MatC[idx] = MatA[idx] + MatB[idx];
            }
        }
    }

    void initialData(int *ip, int size)
    {
        time_t t;
        srand((unsigned int)time(&t));

        for (int i = 0; i < size; i++) {
            ip[i] = (int)(rand() & 0xFF);
        }
    }

    void checkResult(int *hostRef, int *gpuRef, const int N)
    {
        double epsilon = 1.0E-8;
        bool match = 1;

        for (int i = 0; i < N; i++) {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %d gpu %d at current %d\n", hostRef[i], gpuRef[i], i);
                break;
            }
        }

        if (match) printf("Arrays match.\n");

        return;
    }
    
    int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // set up date size of matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n",nx, ny);
    // malloc host memory
    int *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);
    // initialize data at host side
    double iStart = cpuSecond();
    initialData (h_A, nxy);
    initialData (h_B, nxy);
    double iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    // add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnHost (h_A, h_B, hostRef, nx,ny);
    iElaps = cpuSecond() - iStart;
    // malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    iStart = cpuSecond();
    sumMatrixOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    grid.y, block.x, block.y, iElaps);
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    // check device results
    checkResult(hostRef, gpuRef, nxy);
    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    // reset device
    cudaDeviceReset();
    return (0);
    }
## Output:
    $ nvcc -arch=sm_20 sumMatrixOnGPU-2D-grid-2D-block.cu -o matrix2D
    $ ./matrix2D
    
    sumMatrixOnGPU2D <<<(512,512), (32,32)>>> elapsed 0.060323 sec

    * Next, alter the block dimensions to 32 x 16 and recompile and rerun. The kernel becomes nearly two times faster:
    
    sumMatrixOnGPU2D <<<(512,1024), (32,16)>>> elapsed 0.038041 sec

    * You may wonder why the kernel performance nearly doubled just by altering the execution configuration. Intuitively, you may reason that the second configuration has twice as many blocks as
    the first configuration, so there is twice as much parallelism. The intuition is right.
    * However, if you further reduce the block size to 16 x 16, you have quadrupled the number of blocks compared to
    the first configuration.
    * The result of this configuration, as shown below, is better than the first but worse than the second.
    
    sumMatrixOnGPU2D <<< (1024,1024), (16,16) >>> elapsed 0.045535 sec
 
  ![image](https://user-images.githubusercontent.com/91734840/236863397-4f3e46b2-a519-4c3e-885b-6cdff46f485f.png)

## Result:
Thus, the Matrix Summation with a 2D grid and 2D blocks. And adapt it to Integer Matrix Addition has been successfully performed.
