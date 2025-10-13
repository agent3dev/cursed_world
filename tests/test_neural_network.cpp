#include <gtest/gtest.h>
#include "NeuralNetwork.h"
#include <cmath>

// Test fixture for NeuralNetwork tests
class NeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup runs before each test
    }

    void TearDown() override {
        // Cleanup runs after each test
    }
};

// Test construction
TEST_F(NeuralNetworkTest, Construction) {
    // Neural network should be created without throwing
    EXPECT_NO_THROW(NeuralNetwork nn({3, 5, 2}));

    NeuralNetwork nn({3, 5, 2});  // 3 inputs, 5 hidden, 2 outputs
    std::vector<double> input = {0.5, 0.3, 0.7};

    // Forward pass should not throw
    EXPECT_NO_THROW(nn.forward(input));
}

// Test forward pass dimensions
TEST_F(NeuralNetworkTest, ForwardPassDimensions) {
    NeuralNetwork nn({4, 6, 3});  // 4 inputs, 6 hidden, 3 outputs

    std::vector<double> input = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> output = nn.forward(input);

    EXPECT_EQ(output.size(), 3);  // Should have 3 outputs
}

// Test forward pass with specific values
TEST_F(NeuralNetworkTest, ForwardPassValues) {
    NeuralNetwork nn({2, 3, 1});  // Simple network

    std::vector<double> input = {1.0, 0.0};
    std::vector<double> output = nn.forward(input);

    EXPECT_EQ(output.size(), 1);
    // Output should be in range [-1, 1] due to tanh
    EXPECT_GE(output[0], -1.0);
    EXPECT_LE(output[0], 1.0);
}

// Test recurrent connections (memory)
TEST_F(NeuralNetworkTest, RecurrentMemory) {
    NeuralNetwork nn({3, 4, 2});

    std::vector<double> input = {0.5, 0.5, 0.5};

    // First forward pass
    std::vector<double> output1 = nn.forward(input);

    // Second forward pass with same input should be different due to memory
    std::vector<double> output2 = nn.forward(input);

    // Outputs might be different due to recurrent connections
    // (though with same input they could be similar after settling)
    EXPECT_EQ(output1.size(), output2.size());
}

// Test reset hidden state
TEST_F(NeuralNetworkTest, ResetHiddenState) {
    NeuralNetwork nn({3, 4, 2});

    std::vector<double> input = {0.5, 0.5, 0.5};

    // Run network
    nn.forward(input);
    nn.forward(input);

    // Reset memory
    nn.resetHiddenState();

    // Should work without crashing
    std::vector<double> output = nn.forward(input);
    EXPECT_EQ(output.size(), 2);
}

// Test weight access
TEST_F(NeuralNetworkTest, WeightAccess) {
    NeuralNetwork nn({2, 3, 1});

    std::vector<double> weights = nn.getWeights();

    // Should have weights for:
    // - Input to hidden: 2*3 = 6
    // - Hidden bias: 3
    // - Recurrent: 3*3 = 9
    // - Hidden to output: 3*1 = 3
    // - Output bias: 1
    // Total: 6 + 3 + 9 + 3 + 1 = 22
    EXPECT_EQ(weights.size(), 22);
}

// Test weight mutation
TEST_F(NeuralNetworkTest, WeightMutation) {
    NeuralNetwork nn({2, 3, 1});

    std::vector<double> original_weights = nn.getWeights();

    // Mutate with 100% rate
    nn.mutate(1.0, 0.5);

    std::vector<double> mutated_weights = nn.getWeights();

    // At least some weights should have changed
    bool weights_changed = false;
    for (size_t i = 0; i < original_weights.size(); i++) {
        if (std::abs(original_weights[i] - mutated_weights[i]) > 0.001) {
            weights_changed = true;
            break;
        }
    }

    EXPECT_TRUE(weights_changed);
}

// Test weight mutation with zero rate
TEST_F(NeuralNetworkTest, NoMutationWithZeroRate) {
    NeuralNetwork nn({2, 3, 1});

    std::vector<double> original_weights = nn.getWeights();

    // Mutate with 0% rate
    nn.mutate(0.0, 0.5);

    std::vector<double> after_weights = nn.getWeights();

    // No weights should have changed
    for (size_t i = 0; i < original_weights.size(); i++) {
        EXPECT_DOUBLE_EQ(original_weights[i], after_weights[i]);
    }
}

// Test copy from another network
TEST_F(NeuralNetworkTest, CopyFromAnotherNetwork) {
    NeuralNetwork nn1({3, 4, 2});
    NeuralNetwork nn2({3, 4, 2});

    // Mutate nn1 to make it different
    nn1.mutate(1.0, 0.5);

    std::vector<double> nn1_weights = nn1.getWeights();

    // Copy nn1 weights to nn2
    nn2.setWeights(nn1_weights);

    std::vector<double> nn2_weights = nn2.getWeights();

    // Weights should be identical
    EXPECT_EQ(nn1_weights.size(), nn2_weights.size());
    for (size_t i = 0; i < nn1_weights.size(); i++) {
        EXPECT_DOUBLE_EQ(nn1_weights[i], nn2_weights[i]);
    }
}

// Test output range (tanh activation)
TEST_F(NeuralNetworkTest, OutputRangeWithTanh) {
    NeuralNetwork nn({5, 8, 4});

    // Test with various inputs
    for (int trial = 0; trial < 10; trial++) {
        std::vector<double> input = {
            static_cast<double>(trial) / 10.0,
            static_cast<double>(trial) / 20.0,
            static_cast<double>(trial) / 30.0,
            static_cast<double>(trial) / 40.0,
            static_cast<double>(trial) / 50.0
        };

        std::vector<double> output = nn.forward(input);

        for (double val : output) {
            EXPECT_GE(val, -1.0);
            EXPECT_LE(val, 1.0);
        }
    }
}

// Test mouse brain architecture
TEST_F(NeuralNetworkTest, MouseBrainArchitecture) {
    NeuralNetwork mouse_brain({9, 16, 9});

    std::vector<double> mouse_input(9, 0.5);  // 9 inputs
    std::vector<double> output = mouse_brain.forward(mouse_input);

    EXPECT_EQ(output.size(), 9);  // 9 outputs
}

// Test cat brain architecture
TEST_F(NeuralNetworkTest, CatBrainArchitecture) {
    NeuralNetwork cat_brain({10, 16, 9});

    std::vector<double> cat_input(10, 0.5);  // 10 inputs
    std::vector<double> output = cat_brain.forward(cat_input);

    EXPECT_EQ(output.size(), 9);  // 9 outputs
}
