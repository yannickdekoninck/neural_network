#include "tensor.h"
#include <iostream>

int main(int, char **)
{
    std::cout << "Hello Deep Neural Networks!!!\n\n";
    std::cout << "1) Creating a new tensor of size [2,5,6]" << std::endl;
    NeuralNetwork::Tensor t(2, 5, 6);
    std::cout << "\tTensor value at index [1,2,3]: " << t.get(1, 2, 3) << std::endl;
    std::cout << "2) Initializing to 7.0f" << std::endl;
    t.initialize_value(7.0f);
    std::cout << "\tTensor value at index [1,2,3]: " << t.get(1, 2, 3) << std::endl;
    t.set(16.0f, 1, 2, 3);
    std::cout << "3) Setting value of [1,2,3] to 16.0f" << std::endl;
    std::cout << "\tTensor value at index [1,2,3]: " << t.get(1, 2, 3) << std::endl;
}
