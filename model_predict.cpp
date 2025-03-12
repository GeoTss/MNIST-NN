#include <iostream>
#include "Eigen/Core"
#include <fstream>

#include "MnistDataLoader.hpp"

inline Eigen::MatrixXf ReLU(const Eigen::MatrixXf& Z) {
    return Z.cwiseMax(0.0f);
}

inline Eigen::MatrixXf softmax(const Eigen::MatrixXf& Z) {
    Eigen::MatrixXf Z_stable = Z.rowwise() - Z.colwise().maxCoeff();
    auto expZ = Z_stable.array().exp();
    Eigen::RowVectorXf sumExp = expZ.colwise().sum();
    return (expZ.array().rowwise() / sumExp.array()).matrix();
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf,
           Eigen::MatrixXf, Eigen::MatrixXf,
           Eigen::MatrixXf, Eigen::MatrixXf>
forward_prop(const Eigen::MatrixXf& w1, const Eigen::MatrixXf& b1,
                      const Eigen::MatrixXf& w2, const Eigen::MatrixXf& b2,
                      const Eigen::MatrixXf& w3, const Eigen::MatrixXf& b3,
                      const Eigen::MatrixXf& X) {
    
    Eigen::MatrixXf z1 = w1 * X + b1 * Eigen::RowVectorXf::Ones(X.cols());
    Eigen::MatrixXf a1 = ReLU(z1);
    
    Eigen::MatrixXf z2 = w2 * a1 + b2 * Eigen::RowVectorXf::Ones(a1.cols());
    Eigen::MatrixXf a2 = ReLU(z2);
    
    Eigen::MatrixXf z3 = w3 * a2 + b3 * Eigen::RowVectorXf::Ones(a2.cols());
    Eigen::MatrixXf a3 = softmax(z3);
    
    return std::make_tuple(z1, a1, z2, a2, z3, a3);
}

int predict_digit(
    const Eigen::MatrixXf& w1, const Eigen::MatrixXf& b1,
    const Eigen::MatrixXf& w2, const Eigen::MatrixXf& b2,
    const Eigen::MatrixXf& w3, const Eigen::MatrixXf& b3,
    const Eigen::MatrixXf& test_case)
{
    auto propagation_result = forward_prop(w1, b1, w2, b2, w3, b3, test_case);

    Eigen::MatrixXf softmax_result = std::get<5>(propagation_result);

    Eigen::Index index;
    softmax_result.col(0).maxCoeff(&index);
    
    return (int)index;
}

Eigen::MatrixXf readMatrixFromBinaryFile(std::ifstream& inFile) {
    int rows, cols;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(int));
    Eigen::MatrixXf matrix(rows, cols);
    inFile.read(reinterpret_cast<char*>(matrix.data()), sizeof(float) * rows * cols);
    return matrix;
}

int main(int argc, char** argv){
    std::ifstream inFile("./trained_model/model.bin", std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: model.bin\n";
        return 1;
    }
    auto w1 = readMatrixFromBinaryFile(inFile);
    auto b1 = readMatrixFromBinaryFile(inFile);
    auto w2 = readMatrixFromBinaryFile(inFile);
    auto b2 = readMatrixFromBinaryFile(inFile);
    auto w3 = readMatrixFromBinaryFile(inFile);
    auto b3 = readMatrixFromBinaryFile(inFile);

    inFile.close();

    MnistData mnistDataLoader(GET_TEST_SET);
    
    for(int i{1}; i < argc; ++i){
        Eigen::MatrixXf test_case = mnistDataLoader.test_images.col(std::stoi(argv[i])).cast<float>();
        test_case /= 255.f;

        std::cout << predict_digit(w1, b1, w2, b2, w3, b3, test_case) << ' ';
    }
    return 0;
}