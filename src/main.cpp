#include "tensor.h"
#include <iostream>
#include <chrono>

void main_tensor_multiply()
{
    std::cout << "Testing out tensor multiplication\n";
    std::cout << "*********************************\n";
    NeuralNetwork::Tensor t(100, 50, 70);
    NeuralNetwork::Tensor t2(100, 16, 50);
    NeuralNetwork::Tensor t3(100, 16, 70);

    t.initialize_value(1.0f);
    t2.initialize_value(7.0f);
    // Timing the execution
    auto start = std::chrono::high_resolution_clock::now();
    bool success = NeuralNetwork::Tensor::matrix_multiply(t, t2, t3);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Multiplication result status: " << success << std::endl;
    std::cout << "Time to run multiplication: " << duration.count() << "ms" << std::endl;

    std::cout << "The result at index [77,4,25]: " << t3.get(77, 4, 25) << std::endl;
}

void main_tensor_basics()
{
    std::cout << "Testing out some tensor operations\n";
    std::cout << "**********************************\n";
    std::cout << "1) Creating a few tensors of size [2,5,6]" << std::endl;
    NeuralNetwork::Tensor t(2, 5, 6);
    NeuralNetwork::Tensor t2(2, 5, 6);
    NeuralNetwork::Tensor t3(2, 5, 6);
    NeuralNetwork::Tensor t4(2, 4, 6);
    std::cout << "\tTensor t value at index [1,2,3]: " << t.get(1, 2, 3) << std::endl;
    std::cout << "2) Initializing t to 7.0f" << std::endl;
    t.initialize_value(7.0f);
    t2.initialize_value(9.0f);
    t3.initialize_value(25.0f);
    t4.initialize_value(-7.0f);
    std::cout << "\tTensor t value at index [1,2,3]: " << t.get(1, 2, 3) << std::endl;
    t.set(16.0f, 1, 2, 3);
    std::cout << "3) Setting value of t [1,2,3] to 16.0f" << std::endl;
    std::cout << "\tTensor value t at index [1,2,3]: " << t.get(1, 2, 3) << std::endl;

    std::cout << "4) Can we operate on t and t2? \n\t" << NeuralNetwork::Tensor::check_valid_two_operants_elementwise(t, t2) << std::endl;
    std::cout << "5) Can we operate on t and t4? \n\t" << NeuralNetwork::Tensor::check_valid_two_operants_elementwise(t, t4) << std::endl;

    std::cout << "6) Adding up tensors t and t2 and storing the result in t3:" << std::endl;
    NeuralNetwork::Tensor::add(t, t2, t3);
    std::cout << "\tTensor value t3 at index [1,2,3]: " << t3.get(1, 2, 3) << std::endl;
}

int main(int, char **)
{
    std::cout << "Hello Deep Neural Networks!!!\n\n";
    main_tensor_multiply();
}