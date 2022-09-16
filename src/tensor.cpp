#include <algorithm>
#include "tensor.h"

// Helpers

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

bool NeuralNetwork::Tensor::check_valid_two_operants_elementwise(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2)
{
    // dimension 0
    if (tensor_operant_1.dimensions[0] != tensor_operant_2.dimensions[0])
    {
        return false;
    }
    // dimension 1
    if (tensor_operant_1.dimensions[1] != tensor_operant_2.dimensions[1])
    {
        return false;
    }
    // dimension 2
    if (tensor_operant_1.dimensions[2] != tensor_operant_2.dimensions[2])
    {
        return false;
    }

    return true;
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

// Mathematical operators

void NeuralNetwork::Tensor::add(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2, Tensor &result)
{
    // Check if the involved tensor shapes match
    if (check_valid_two_operants_elementwise(tensor_operant_1, tensor_operant_2) && check_valid_two_operants_elementwise(tensor_operant_1, result))
    {
        // Add up the two tensors
        unsafe_add(tensor_operant_1, tensor_operant_2, result);
    }
}

void NeuralNetwork::Tensor::scale(float scaling_factor, const Tensor &tensor_operant, Tensor &result)
{
    if (check_valid_two_operants_elementwise(tensor_operant, result))
    {
        unsafe_scale(scaling_factor, tensor_operant, result);
    }
}

void NeuralNetwork::Tensor::unsafe_add(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2, Tensor &result)
{
    // Add up two tensors without doing any shape checking
    // We rely on the code to have checked all of this before
    for (unsigned int i = 0; i < result.total_size; i++)
    {
        result.values[i] = tensor_operant_1.values[i] + tensor_operant_2.values[i];
    }
}

void NeuralNetwork::Tensor::unsafe_scale(float scaling_factor, const Tensor &tensor_operant, Tensor &result)
{
    // Add up two tensors without doing any shape checking
    // We rely on the code to have checked all of this before
    for (unsigned int i = 0; i < result.total_size; i++)
    {
        result.values[i] = scaling_factor * tensor_operant.values[i];
    }
}
