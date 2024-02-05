
#include <stdexcept>
#include <iostream>
#include <random>   
#include <chrono>
#include <time.h>
template <class T> class Matrix{
    private:
    long long int n;
    long long int m;
    T* data;

    bool isInitialized = false;
    static bool init;
    public:

    __host__ __device__  bool isValid(long long, long long);


    Matrix();
    Matrix(long long, long long);
    ~Matrix();
    void moveToGPU();
    void moveToCPU();
    T* getData();
     __host__ __device__ long long  getN();
     __host__ __device__ long long getM();
    __host__ __device__  void setValue(long long, long long, T);
    __host__ __device__  T getValue(long long, long long);
    void fillWithRandomInt(int);
    void print();

};

template <typename T>
__host__ __device__ bool Matrix<T>::isValid(long long int i, long long int j){
    bool validCondition = (i < n) && (j < m) && (i >= 0) && (j >=  0);
    return (validCondition);
}

template <typename T>
Matrix<T>::Matrix(){
    this->isInitialized = false;
    if(!init) {
        init = true;
        srand(time(NULL));
    }
}

template <typename T>
Matrix<T>::Matrix(long long n, long long m){
    this->n = n;
    this->m = m;
    this->isInitialized = true;
    this->data = new T[n*m];

}



template <typename T>
Matrix<T>::~Matrix(){
    delete []data;

}
template <typename T>
void Matrix<T>::moveToCPU(){
    T* data;
    data = new T[this->n*this->m];
    size_t bytes = sizeof(T)*(this->n)*(this->m);
    cudaError_t error = cudaMemcpy(data, this->data, bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr,  "CudaMempyFailed while moving to cpu! : %s", cudaGetErrorString(error) );
     
    }
    // cudaFree(this->data);
    this->data = data;

}

template <typename T>
void Matrix<T>::moveToGPU(){
    T* data;
    size_t bytes = sizeof(T)*(this->n)*(this->m);
    cudaError_t error =  cudaMalloc((T **)&data, bytes);
    if (error != cudaSuccess) {
        fprintf(stderr, "CudaMallocFailed while moving to gpu!");
     
    }
    cudaMemcpy(data, this->data, bytes, cudaMemcpyHostToDevice);
    this->data = data;
}

template <typename T>
T* Matrix<T>::getData(){
    return this->data;
}

template <typename T>
 __host__ __device__  long long Matrix<T>::getM(){
    return this->m;
}
template <typename T>
 __host__ __device__ long long Matrix<T>::getN(){
    return this->n;
} 


template <typename T>
__host__ __device__ void Matrix<T>::setValue(long long i, long long j, T value){
    this->data[j + i*m] = value;
}


template <typename T>
__host__ __device__ T Matrix<T>::getValue(long long i, long long j){

    return this->data[j + i*m];
}



template <typename T>
void Matrix<T>::fillWithRandomInt(int maxValue){

    for(long long int i = 0;i<n;i++){
        for(long long int j = 0;j<m;j++){
            
            this->data[i*m + j] =  rand()%maxValue;
        }
    }

    
}



template <typename T>
void Matrix<T>::print(){


    for(long long i = 0;i<n;i++){
        for(long long j = 0;j<m;j++){
            std::cout << data[i*m + j] << " ";
        }
        std::cout << std::endl;

    }

    std::cout << std::endl;

}

