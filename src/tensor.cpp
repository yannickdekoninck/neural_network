#include <algorithm>
#include "tensor.h"

void NeuralNetwork::Tensor::set_dimensions_and_indexers(unsigned int n_k, unsigned int n_j, unsigned int n_i)
{
    // Set dimensions and safeguard against 0 dimensions

    dimensions[0] = std::max(1, (int)n_i); // This is the deepest dimension
    dimensions[1] = std::max(1, (int)n_j); // This is the second dimension
    dimensions[2] = std::max(1, (int)n_k); // This is the outer dimension

    // Calculate total size
    total_size = 1;
    for (int i = 0; i < 3; i++)
    {
        total_size *= dimensions[i];
    }

    // Set indexers
    indexers[0] = 0;
    indexers[1] = dimensions[0];
    indexers[2] = dimensions[0] * dimensions[1];
}

// Main constructor
NeuralNetwork::Tensor::Tensor(const float *initial_data, unsigned int n_k, unsigned int n_j, unsigned int n_i)
{

    // Update indexing data
    set_dimensions_and_indexers(n_k, n_j, n_i);

    // allocate data
    values = new float[total_size];

    // copy if data is available

    if (initial_data != nullptr)
    {
        size_t total_bytes_to_copy = total_size * sizeof(float);
        memcpy(values, initial_data, total_bytes_to_copy);
    }
}

NeuralNetwork::Tensor::~Tensor()
{
    // Clean up values
    delete[] values;
}

// initializers

void NeuralNetwork::Tensor::initialize_value(float value)
{
    for (unsigned int i = 0; i < total_size; i++)
    {
        values[i] = value;
    }
}

// accessors

float NeuralNetwork::Tensor::get(unsigned int k, unsigned int j, unsigned int i)
{
    size_t index = i + indexers[1] * j + indexers[2] * k;
    if (index < total_size)
    {
        return values[index];
    }
    return 0.0f;
}

void NeuralNetwork::Tensor::set(float value, unsigned int k, unsigned int j, unsigned int i)
{
    size_t index = i + indexers[1] * j + indexers[2] * k;
    if (index < total_size)
    {
        values[index] = value;
    }
}