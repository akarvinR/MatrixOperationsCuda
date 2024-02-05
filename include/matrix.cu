
#include <stdexcept>
#include <iostream>
#include <random>   
#include <chrono>
#include <time.h>
template <class T> class Matrix{
    private:
    int n;
    int m;
    T* data;

    bool isInitialized = false;
    static bool init;
    public:

    Matrix();
    Matrix(int, int);
    ~Matrix();

    __host__ __device__  bool isValid(int,int);
    T* getData();
     __host__ __device__ int getN();
     __host__ __device__ int getM();
    

    __host__ __device__  void setValue(int, int, T);
    __host__ __device__  T getValue(int, int);

    void fillWithRandomInt(int);
    void moveToGPU();
    void moveToCPU();
    void print();

};

template <typename T>
__host__ __device__ bool Matrix<T>::isValid(int i, int j){
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
Matrix<T>::Matrix(int n, int m){
    this->n = n;
    this->m = m;
    this->isInitialized = true;
    this->data = new int[n*m];

}

template <typename T>
void Matrix<T>::moveToCPU(){
    T* data;
    data = new T[this->n*this->m];

    cudaError_t error = cudaMemcpy(data, this->data, sizeof(T)*(this->n)*(this->m), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr,  "CudaMempyFailed while moving to cpu! : %s", cudaGetErrorString(error) );
     
    }
    // cudaFree(this->data);
    this->data = data;

}

template <typename T>
void Matrix<T>::moveToGPU(){
    T* data;
    
    cudaError_t error =  cudaMalloc((T **)&data, sizeof(T)*(this->n)*(this->m));
    if (error != cudaSuccess) {
        fprintf(stderr, "CudaMallocFailed while moving to gpu!");
     
    }
    cudaMemcpy(data, this->data, sizeof(T)*(this->n)*(this->m), cudaMemcpyHostToDevice);
    this->data = data;
}

template <typename T>
T* Matrix<T>::getData(){
    return this->data;
}

template <typename T>
 __host__ __device__  int Matrix<T>::getM(){
    return this->m;
}
template <typename T>
 __host__ __device__ int Matrix<T>::getN(){
    return this->n;
} 


template <typename T>
__host__ __device__ void Matrix<T>::setValue(int i, int j, T value){
    this->data[j + i*m] = value;
}


template <typename T>
__host__ __device__ T Matrix<T>::getValue(int i, int j){

    return this->data[j + i*m];
}



template <typename T>
void Matrix<T>::fillWithRandomInt(int maxValue){

    for(int i = 0;i<n;i++){
        for(int j = 0;j<m;j++){
            this->data[i*m + j] =  rand()%maxValue;
        }
    }

    
}

template <typename T>
Matrix<T>::~Matrix(){
    delete data;

}


template <typename T>
void Matrix<T>::print(){


    for(int i = 0;i<n;i++){
        for(int j = 0;j<m;j++){
            std::cout << data[i*m + j] << " ";
        }
        std::cout << std::endl;

    }

    std::cout << std::endl;

}

