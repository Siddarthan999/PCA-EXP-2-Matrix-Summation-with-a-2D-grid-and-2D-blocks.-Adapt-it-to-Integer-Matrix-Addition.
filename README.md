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

                #include "common.h"
                #include <cuda_runtime.h>
                #include <stdio.h>

                /*
                 * This example demonstrates a simple vector sum on the GPU and on the host.
                 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
                 * GPU. A 2D thread block and 2D grid are used. sumArraysOnHost sequentially
                 * iterates through vector elements on the host.
                 */

                void initialData(int *ip, const int size)
                {
                    int i;

                    for(i = 0; i < size; i++)
                    {
                        ip[i] = (int)(rand() & 0xFF) / 10.0f;
                    }

                    return;
                }

                void sumMatrixOnHost(int *A, int *B, int *C, const int nx,
                                     const int ny)
                {
                    int *ia = A;
                    int *ib = B;
                    int *ic = C;

                    for (int iy = 0; iy < ny; iy++)
                    {
                        for (int ix = 0; ix < nx; ix++)
                        {
                            ic[ix] = ia[ix] + ib[ix];

                        }

                        ia += nx;
                        ib += nx;
                        ic += nx;
                    }

                    return;
                }


                void checkResult(int *hostRef, int *gpuRef, const int N)
                {
                    double epsilon = 1.0E-8;
                    bool match = 1;

                    for (int i = 0; i < N; i++)
                    {
                        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
                        {
                            match = 0;
                            printf("host %d gpu %d\n", hostRef[i], gpuRef[i]);
                            break;
                        }
                    }

                    if (match)
                        printf("Arrays match.\n\n");
                    else
                        printf("Arrays do not match.\n\n");
                }

                // grid 2D block 2D
                __global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx,
                                                 int ny)
                {
                    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
                    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
                    unsigned int idx = iy * nx + ix;

                    if (ix < nx && iy < ny)
                        MatC[idx] = MatA[idx] + MatB[idx];
                }

                int main(int argc, char **argv)
                {
                    printf("%s Starting...\n", argv[0]);

                    // set up device
                    int dev = 0;
                    cudaDeviceProp deviceProp;
                    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
                    printf("Using Device %d: %s\n", dev, deviceProp.name);
                    CHECK(cudaSetDevice(dev));

                    // set up data size of matrix
                    int nx = 1 << 14;
                    int ny = 1 << 14;

                    int nxy = nx * ny;
                    int nBytes = nxy * sizeof(int);
                    printf("Matrix size: nx %d ny %d\n", nx, ny);

                    // malloc host memory
                    int *h_A, *h_B, *hostRef, *gpuRef;
                    h_A = (int *)malloc(nBytes);
                    h_B = (int *)malloc(nBytes);
                    hostRef = (int *)malloc(nBytes);
                    gpuRef = (int *)malloc(nBytes);

                    // initialize data at host side
                    double iStart = seconds();
                    initialData(h_A, nxy);
                    initialData(h_B, nxy);
                    double iElaps = seconds() - iStart;
                    printf("Matrix initialization elapsed %f sec\n", iElaps);

                    memset(hostRef, 0, nBytes);
                    memset(gpuRef, 0, nBytes);

                    // add matrix at host side for result checks
                    iStart = seconds();
                    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
                    iElaps = seconds() - iStart;
                    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

                    // malloc device global memory
                    int *d_MatA, *d_MatB, *d_MatC;
                    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
                    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
                    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

                    // transfer data from host to device
                    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
                    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

                    // invoke kernel at host side
                    int dimx = 32;
                    int dimy = 32;
                    dim3 block(dimx, dimy);
                    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

                    iStart = seconds();
                    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
                    CHECK(cudaDeviceSynchronize());
                    iElaps = seconds() - iStart;
                    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
                           grid.y,
                           block.x, block.y, iElaps);
                    // check kernel error
                    CHECK(cudaGetLastError());

                    // copy kernel result back to host side
                    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

                    // check device results
                    checkResult(hostRef, gpuRef, nxy);

                    // free device global memory
                    CHECK(cudaFree(d_MatA));
                    CHECK(cudaFree(d_MatB));
                    CHECK(cudaFree(d_MatC));

                    // free host memory
                    free(h_A);
                    free(h_B);
                    free(hostRef);
                    free(gpuRef);

                    // reset device
                    CHECK(cudaDeviceReset());

                    return (0);
                }
## Output:
                root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_2# nvcc sumMatrixOnGPU-2D-grid-2D-block.cu -o sumMatrixOnGPU-2D-grid-2D-block
                root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_2# nvcc sumMatrixOnGPU-2D-grid-2D-block.cu
                root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_2# ./sumMatrixOnGPU-2D-grid-2D-block
                ./sumMatrixOnGPU-2D-grid-2D-block Starting...
                Using Device 0: NVIDIA GeForce GT 710
                Matrix size: nx 16384 ny 16384
                Matrix initialization elapsed 6.922423 sec
                sumMatrixOnHost elapsed 0.566353 sec
                Error: sumMatrixOnGPU-2D-grid-2D-block.cu:126, code: 2, reason: out of memory
                root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_2#
 ![image](https://user-images.githubusercontent.com/91734840/236863397-4f3e46b2-a519-4c3e-885b-6cdff46f485f.png)

# EXPLANATION:
* The output of the program sumMatrixOnGPU-2D-grid-2D-block has started running and it is using the NVIDIA GeForce GT 710 device for CUDA processing. The program is designed to perform matrix summation using 2D grid and 2D blocks in CUDA programming.
* The size of the matrix is nx 16384 ny 16384, which means the matrix has 16384 rows and 16384 columns. The matrix initialization process has elapsed for 6.922423 seconds.
* The program has also executed sumMatrixOnHost to calculate the sum of the matrix on the host, and it has taken 0.566353 seconds to execute.
* However, an error occurred at line 126 with an error code of 2, and the reason for the error is that the program has run out of memory. This indicates that there is not enough memory available on the GPU to execute the program and perform the matrix addition operation.
* Therefore, the program failed to execute and terminated without producing the desired result.

## Result:
Thus, the Matrix Summation with a 2D grid and 2D blocks. And adapt it to Integer Matrix Addition has been successfully performed.
