#pragma once

namespace NeuralNetwork
{
    class Tensor
    {
    public:
        // Default constructor
        Tensor() : Tensor(nullptr, 1, 1, 1){};
        // Constructors for 1, 2 and 3D tensors without data initializers
        Tensor(unsigned int n_i) : Tensor(nullptr, 1, 1, n_i){};
        Tensor(unsigned int n_j, unsigned int n_i) : Tensor(nullptr, 1, n_j, n_i){};
        Tensor(unsigned int n_k, unsigned int n_j, unsigned int n_i) : Tensor(nullptr, n_k, n_j, n_i){};

        // Constructors for 1, 2 and 3D tensors with data initializers
        Tensor(const float *initial_data, unsigned int n_i) : Tensor(initial_data, 1, 1, n_i){};
        Tensor(const float *initial_data, unsigned int n_j, unsigned int n_i) : Tensor(initial_data, 1, n_j, n_i){};
        Tensor(const float *initial_data, unsigned int n_k, unsigned int n_j, unsigned int n_i);

        // Destructor
        ~Tensor();

        // Initializers
        void initialize_random_uniform(float min, float max);
        void initialize_random_gaussian(float mean, float std);
        void initialize_value(float value);
        inline void initilize_zero() { initialize_value(0.0f); }

        // Accessors

        // Getters
        inline float get(unsigned int i) { return get(0, 0, i); };
        inline float get(unsigned int j, unsigned int i) { return get(0, j, i); };
        float get(unsigned int k, unsigned int j, unsigned int i);

        // Setters
        inline void set(float value, unsigned int i) { return set(value, 0, 0, i); };
        inline void set(float value, unsigned int j, unsigned int i) { return set(value, 0, j, i); };
        void set(float value, unsigned int k, unsigned int j, unsigned int i);

        // Mathematical operators

        static void add(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2, Tensor &result);
        static void scale(float scaling_factor, const Tensor &tensor_operant, Tensor &result);
        inline static void scale(float scaling_factor, Tensor &tensor_operant) { return unsafe_scale(scaling_factor, tensor_operant, tensor_operant); }; // scale in place
        static bool matrix_multiply(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2, Tensor &result);

        // utilities
        static bool check_valid_two_operants_elementwise(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2);

    private:
        float *values;              // The actual values
        unsigned int total_size;    // Keeping track of the total size of the tensor
        unsigned int dimensions[3]; // An array storing the shape of the tensor
        unsigned int indexers[3];   // An array storing offsets to index into the different dimensions

        // helper methods

        void set_dimensions_and_indexers(unsigned int n_k, unsigned int n_j, unsigned int n_i);
        inline size_t expand_array_index(unsigned int k, unsigned int j, unsigned int i) const
        {
            return i + indexers[1] * j + indexers[2] * k;
        }

        // unsafe math -> doesn't check tensor sizes
        static void unsafe_add(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2, Tensor &result);
        static void unsafe_scale(float scaling_factor, const Tensor &tensor_operant, Tensor &result);
        static void unsafe_matrix_multiply(const Tensor &tensor_operant_1, const Tensor &tensor_operant_2, Tensor &result);
    };
} // namespace NeuralNetwork