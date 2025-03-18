#include <iostream>
#include <vector>
#include "Neural-Network/Neural_Network.hpp"
#include "MnistDataLoader.hpp"

#define LOAD_NN

template<typename T>
Matrix<T> oneHotEncode(const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& Y_in) {
    int m = Y_in.size();
    int numClasses = Y_in.maxCoeff() + 1;
    Matrix<T> oneHot = Matrix<T>::Zero(m, numClasses);
    for (int i = 0; i < m; ++i) {
        oneHot(i, Y_in(i)) = 1.;
    }
    return oneHot.transpose();
}


int main() {

    #ifdef EIGEN_VECTORIZE_NEON
    std::cout << "NEON is enabled in Eigen!" << std::endl;
    #else
    std::cout << "NEON is not enabled in Eigen." << std::endl;
    #endif
    
    using test_type = float;
    #ifndef LOAD_NN
    MnistData mnistData(GET_TRAIN_SET | GET_TEST_SET);


    NeuralNetwork<test_type, true> nn({ 784, 256, 128, 64, 10 }, 
        { ActivationFunction::Tanh, ActivationFunction::ReLU, ActivationFunction::ReLU, ActivationFunction::Softmax }, 
        LossFunction::CrossEntropy, 
        Optimizer::Adam,
        { WeightInitializer::RecommendedNormal,  WeightInitializer::RecommendedNormal, WeightInitializer::RecommendedNormal, WeightInitializer::XavierNormal }, 
        3e-4, 
        32
    );
    nn.printNetworkInfo();
    
    auto X_train = mnistData.train_images.cast<test_type>() / 255.f;
    auto Y_train = oneHotEncode<test_type>(mnistData.train_labels);

    nn.train(X_train, Y_train, 10);
    nn.saveModel("./trained_model/library_model.bin");  
    #else
    MnistData mnistData(GET_TEST_SET);

    NeuralNetwork<test_type, true> nn;
    nn.loadModel("./trained_model/library_model.bin");
    std::cout << "Neural network information is loaded.\n";
    std::cout << "Printing network information...\n";
    nn.printNetworkInfo();

    #endif

    auto X_test = mnistData.test_images.cast<test_type>() / 255.f;
    auto Y_test = oneHotEncode<test_type>(mnistData.test_labels);

    int matches = nn.test(X_test, Y_test);
    std::cout << "Accuracy: " << (static_cast<float>(matches) / Y_test.cols()) * 100 << "%\n";
}