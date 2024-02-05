#include <iostream>
#include <matrix.cu>
#include <chrono>
#define TIME std::chrono::steady_clock::time_point

TIME getTime(){
    return std::chrono::steady_clock::now();
}


template<typename T>
__global__ void cudaMatrixAdd(Matrix<T>* a, Matrix<T>*  b, Matrix<T>* c){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;



    if(c->isValid(x,y)){
        c->setValue(x, y, a->getValue(x,y) + b->getValue(x,y));
    }
} 
template<typename T> 
void cpuMatrixAdd(Matrix<T>* a, Matrix<T>*  b, Matrix<T>* c){
    for(int i = 0; i<c->getN();i++){
        for(int j = 0;j<c->getM();j++){
            c->setValue(i, j, a->getValue(i,j) + b->getValue(i,j));
        }
    }

}

template<typename T> 
bool verify(Matrix<T>* a, Matrix<T>*  b, Matrix<T>* c){
    for(int i = 0; i<c->getN();i++){
        for(int j = 0;j<c->getM();j++){
            if( c->getValue(i,j) !=  a->getValue(i,j) + b->getValue(i,j)){
                return false;
            }
        }
    }
    return true;
}
template<typename T>
Matrix<T>* cudaAllocateMatrix(Matrix<T> *matrix){
    Matrix<T>* d_matrix;
    cudaError_t cudaErrorMalloc = cudaMalloc((Matrix<T> **)&d_matrix, sizeof(*d_matrix));


    if (cudaErrorMalloc != cudaSuccess) {
        fprintf(stderr, "CudaMallocFailed failed!");
     
    }
    cudaError_t cudaError = cudaMemcpy(d_matrix, matrix, sizeof(*d_matrix), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CudaMemoryMove failed!");
  
    }
  
    return d_matrix;
}

void MatrixAdd(int n, int m){
    Matrix<int> *a, *b, *c;
    a = new Matrix<int>(n,m);
    b = new Matrix<int>(n,m);
    c = new Matrix<int>(n,m);

    a->fillWithRandomInt(1000);
    b->fillWithRandomInt(1000);

    Matrix<int> *d_a, *d_b, *d_c;

    a->moveToGPU();
    b->moveToGPU();
    c->moveToGPU();


    d_a = cudaAllocateMatrix(a);
    d_b = cudaAllocateMatrix(b);
    d_c = cudaAllocateMatrix(c);

    int threadsPerBlock = 32;
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(a->getN()/threadsPerBlock + 1, a->getM()/threadsPerBlock + 1);



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMatrixAdd<<<grid, block>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float GPUmilliseconds = 0;
    cudaEventElapsedTime(&GPUmilliseconds, start, stop);

    a->moveToCPU();
    b->moveToCPU();
    c->moveToCPU();

    std::cout << "GPU Completed Addition in " << GPUmilliseconds << " [ms]" << std::endl;
    if(!verify(a, b, c)){
        std::cout << "Matrix Addition Verification Has Failed" << std::endl;
    }

    TIME begin = getTime();
    cpuMatrixAdd(a, b, c);
    TIME end = getTime();
    std::cout << "CPU Completed Addition in " << (std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() )<< " [ms]" << std::endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete a;
    delete b;
    delete c;
}
int main(){
    int n, m;
    std::cin >> n >> m;
    MatrixAdd(n,m);

    return 0;
}