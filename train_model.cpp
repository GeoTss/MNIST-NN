#include <iostream>
#include <cstdint>
#include <fstream>
#include <vector>
#include <tuple>
#include <Eigen/Core>
#include <random>

#include "./MnistDataLoader.hpp"

inline Eigen::MatrixXf ReLU(const Eigen::MatrixXf& Z) {
    return Z.cwiseMax(0.0f);
}

Eigen::MatrixXf softmax(const Eigen::MatrixXf& Z) {
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
    const Eigen::MatrixXf& X
    ) {

    Eigen::MatrixXf z1 = w1 * X + b1 * Eigen::RowVectorXf::Ones(X.cols());
    Eigen::MatrixXf a1 = ReLU(z1);

    Eigen::MatrixXf z2 = w2 * a1 + b2 * Eigen::RowVectorXf::Ones(a1.cols());
    Eigen::MatrixXf a2 = ReLU(z2);

    Eigen::MatrixXf z3 = w3 * a2 + b3 * Eigen::RowVectorXf::Ones(a2.cols());
    Eigen::MatrixXf a3 = softmax(z3);

    return std::make_tuple(z1, a1, z2, a2, z3, a3);
}

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf,
    Eigen::MatrixXf, Eigen::MatrixXf,
    Eigen::MatrixXf, Eigen::MatrixXf>
    backward_prop(const Eigen::MatrixXf& w1, const Eigen::MatrixXf& w2, const Eigen::MatrixXf& w3,
    const Eigen::MatrixXf& z1, const Eigen::MatrixXf& a1,
    const Eigen::MatrixXf& z2, const Eigen::MatrixXf& a2,
    const Eigen::MatrixXf& a3, const Eigen::MatrixXf& X,
    const Eigen::MatrixXf& Y
    ) {

    int m = X.cols();

    Eigen::MatrixXf dz3 = a3 - Y;
    Eigen::MatrixXf dw3 = (dz3 * a2.transpose()) / m;
    Eigen::MatrixXf db3 = dz3.rowwise().sum() / m;

    Eigen::MatrixXf relu_deriv2 = (z2.array() > 0).cast<float>();
    Eigen::MatrixXf dz2 = (w3.transpose() * dz3).array() * relu_deriv2.array();
    Eigen::MatrixXf dw2 = (dz2 * a1.transpose()) / m;
    Eigen::MatrixXf db2 = dz2.rowwise().sum() / m;

    Eigen::MatrixXf relu_deriv1 = (z1.array() > 0).cast<float>();
    Eigen::MatrixXf dz1 = (w2.transpose() * dz2).array() * relu_deriv1.array();
    Eigen::MatrixXf dw1 = (dz1 * X.transpose()) / m;
    Eigen::MatrixXf db1 = dz1.rowwise().sum() / m;

    return std::make_tuple(dw1, db1, dw2, db2, dw3, db3);
}

Eigen::MatrixXf oneHotEncode(const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& Y_in) {
    int m = Y_in.size();
    int numClasses = Y_in.maxCoeff() + 1;
    Eigen::MatrixXf oneHot = Eigen::MatrixXf::Zero(m, numClasses);
    for (int i = 0; i < m; ++i) {
        oneHot(i, Y_in(i)) = 1.0f;
    }
    return oneHot.transpose();
}

std::random_device rd;
std::mt19937 gen(rd());

Eigen::MatrixXf heInit(int rows, int cols) {
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / cols));
    Eigen::MatrixXf W(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            W(i, j) = dist(gen);
    return W;
}

void shuffle_data(Eigen::MatrixXf& X, Eigen::MatrixXf& Y) {
    int m = X.cols();
    std::vector<int> indices(m);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    Eigen::MatrixXf X_shuffled = X(Eigen::all, Eigen::Map<Eigen::VectorXi>(indices.data(), indices.size()));
    Eigen::MatrixXf Y_shuffled = Y(Eigen::all, Eigen::Map<Eigen::VectorXi>(indices.data(), indices.size()));
    X = X_shuffled;
    Y = Y_shuffled;
}

void writeMatrixToBinaryFile(std::ofstream& outFile, const Eigen::MatrixXf& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    outFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
    outFile.write(reinterpret_cast<char*>(&cols), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(matrix.data()), sizeof(float) * rows * cols);
}

Eigen::MatrixXf readMatrixFromBinaryFile(std::ifstream& inFile) {
    int rows, cols;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(int));
    Eigen::MatrixXf matrix(rows, cols);
    inFile.read(reinterpret_cast<char*>(matrix.data()), sizeof(float) * rows * cols);
    return matrix;
}

std::vector<uint8_t> get_predictions(const Eigen::MatrixXf& Y) {
    std::vector<uint8_t> predictions;
    for (int j = 0; j < Y.cols(); ++j) {
        Eigen::Index index;
        Y.col(j).maxCoeff(&index);
        predictions.push_back(static_cast<uint8_t>(index));
    }
    return predictions;
}

size_t countMatchingLabels(const std::vector<uint8_t>& predictions,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& trueLabels) {
    assert(predictions.size() == static_cast<size_t>(trueLabels.size()));
    size_t correctCount = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == trueLabels(i)) {
            ++correctCount;
        }
    }
    return correctCount;
}


std::tuple<Eigen::MatrixXf, Eigen::MatrixXf,
Eigen::MatrixXf, Eigen::MatrixXf,
Eigen::MatrixXf, Eigen::MatrixXf>
train(const MnistData& mnistData) {
    
    constexpr size_t BATCH_SIZE = 64;

    constexpr int input_size = 784;
    constexpr int hidden1_size = 326;
    constexpr int hidden2_size = 128;
    constexpr int output_size = 10;

    Eigen::MatrixXf w1 = heInit(hidden1_size, input_size);
    Eigen::MatrixXf b1 = Eigen::MatrixXf::Zero(hidden1_size, 1);
    Eigen::MatrixXf w2 = heInit(hidden2_size, hidden1_size);
    Eigen::MatrixXf b2 = Eigen::MatrixXf::Zero(hidden2_size, 1);
    Eigen::MatrixXf w3 = heInit(output_size, hidden2_size);
    Eigen::MatrixXf b3 = Eigen::MatrixXf::Zero(output_size, 1);

    Eigen::MatrixXf X_train = mnistData.train_images.cast<float>() / 255.0f;
    Eigen::MatrixXf X_dev = mnistData.dev_images.cast<float>() / 255.0f;

    Eigen::MatrixXf Y_train = oneHotEncode(mnistData.train_labels);

    constexpr float learning_rate = 3e-4f;
    constexpr int num_iterations = 20;
    std::cout << "Hyperparameters:\nLearning rate: " << learning_rate
        << "\nIterations: " << num_iterations << "\n";


    int num_batches = X_train.cols() / BATCH_SIZE;

    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-8f;
    int t = 0;

    Eigen::MatrixXf m_w1 = Eigen::MatrixXf::Zero(w1.rows(), w1.cols());
    Eigen::MatrixXf v_w1 = Eigen::MatrixXf::Zero(w1.rows(), w1.cols());
    Eigen::MatrixXf m_b1 = Eigen::MatrixXf::Zero(b1.rows(), b1.cols());
    Eigen::MatrixXf v_b1 = Eigen::MatrixXf::Zero(b1.rows(), b1.cols());

    Eigen::MatrixXf m_w2 = Eigen::MatrixXf::Zero(w2.rows(), w2.cols());
    Eigen::MatrixXf v_w2 = Eigen::MatrixXf::Zero(w2.rows(), w2.cols());
    Eigen::MatrixXf m_b2 = Eigen::MatrixXf::Zero(b2.rows(), b2.cols());
    Eigen::MatrixXf v_b2 = Eigen::MatrixXf::Zero(b2.rows(), b2.cols());

    Eigen::MatrixXf m_w3 = Eigen::MatrixXf::Zero(w3.rows(), w3.cols());
    Eigen::MatrixXf v_w3 = Eigen::MatrixXf::Zero(w3.rows(), w3.cols());
    Eigen::MatrixXf m_b3 = Eigen::MatrixXf::Zero(b3.rows(), b3.cols());
    Eigen::MatrixXf v_b3 = Eigen::MatrixXf::Zero(b3.rows(), b3.cols());

    for (int iter = 1; iter <= num_iterations; ++iter) {
        std::cout << "Iteration " << iter << "\n";
        shuffle_data(X_train, Y_train);
        double total_cost = 0.0;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * BATCH_SIZE;
            Eigen::MatrixXf X_batch = X_train.block(0, start_idx, X_train.rows(), BATCH_SIZE);
            Eigen::MatrixXf Y_batch = Y_train.block(0, start_idx, Y_train.rows(), BATCH_SIZE);

            auto [z1, a1, z2, a2, z3, a3] = forward_prop(w1, b1, w2, b2, w3, b3, X_batch);
            auto [dw1, db1, dw2, db2, dw3, db3] = backward_prop(w1, w2, w3, z1, a1, z2, a2, a3, X_batch, Y_batch);

            double batch_cost = -(Y_batch.array() * a3.array().log()).sum() / BATCH_SIZE;
            total_cost += batch_cost;

            // w1 -= learning_rate * dw1;
            // b1 -= learning_rate * db1;
            // w2 -= learning_rate * dw2;
            // b2 -= learning_rate * db2;
            // w3 -= learning_rate * dw3;
            // b3 -= learning_rate * db3;

            t += 1;
            m_w1 = beta1 * m_w1 + (1 - beta1) * dw1;
            v_w1 = beta2 * v_w1 + (1 - beta2) * dw1.cwiseProduct(dw1);
            m_b1 = beta1 * m_b1 + (1 - beta1) * db1;
            v_b1 = beta2 * v_b1 + (1 - beta2) * db1.cwiseProduct(db1);

            m_w2 = beta1 * m_w2 + (1 - beta1) * dw2;
            v_w2 = beta2 * v_w2 + (1 - beta2) * dw2.cwiseProduct(dw2);
            m_b2 = beta1 * m_b2 + (1 - beta1) * db2;
            v_b2 = beta2 * v_b2 + (1 - beta2) * db2.cwiseProduct(db2);

            m_w3 = beta1 * m_w3 + (1 - beta1) * dw3;
            v_w3 = beta2 * v_w3 + (1 - beta2) * dw3.cwiseProduct(dw3);
            m_b3 = beta1 * m_b3 + (1 - beta1) * db3;
            v_b3 = beta2 * v_b3 + (1 - beta2) * db3.cwiseProduct(db3);

            Eigen::MatrixXf m_w1_hat = m_w1 / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_w1_hat = v_w1 / (1 - std::pow(beta2, t));
            Eigen::MatrixXf m_b1_hat = m_b1 / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_b1_hat = v_b1 / (1 - std::pow(beta2, t));

            Eigen::MatrixXf m_w2_hat = m_w2 / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_w2_hat = v_w2 / (1 - std::pow(beta2, t));
            Eigen::MatrixXf m_b2_hat = m_b2 / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_b2_hat = v_b2 / (1 - std::pow(beta2, t));

            Eigen::MatrixXf m_w3_hat = m_w3 / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_w3_hat = v_w3 / (1 - std::pow(beta2, t));
            Eigen::MatrixXf m_b3_hat = m_b3 / (1 - std::pow(beta1, t));
            Eigen::MatrixXf v_b3_hat = v_b3 / (1 - std::pow(beta2, t));

            w1 -= (learning_rate * m_w1_hat.array() / (v_w1_hat.array().sqrt() + epsilon)).matrix();
            b1 -= (learning_rate * m_b1_hat.array() / (v_b1_hat.array().sqrt() + epsilon)).matrix();
            w2 -= (learning_rate * m_w2_hat.array() / (v_w2_hat.array().sqrt() + epsilon)).matrix();
            b2 -= (learning_rate * m_b2_hat.array() / (v_b2_hat.array().sqrt() + epsilon)).matrix();
            w3 -= (learning_rate * m_w3_hat.array() / (v_w3_hat.array().sqrt() + epsilon)).matrix();
            b3 -= (learning_rate * m_b3_hat.array() / (v_b3_hat.array().sqrt() + epsilon)).matrix();
        }

        std::cout << "Cost: " << total_cost / num_batches << "\n";

        if (mnistData.uses_dev_set == true) {
            if (iter % 10 == 0) {
                std::cout << "Evaluating on dev set...\n";
                auto [dz1_dev, da1_dev, dz2_dev, da2_dev, dz3_dev, a3_dev] = forward_prop(w1, b1, w2, b2, w3, b3, X_dev);
                auto predictions = get_predictions(a3_dev);
                size_t correct_predictions = countMatchingLabels(predictions, mnistData.dev_labels);
                std::cout << "Accuracy: "
                    << (static_cast<float>(correct_predictions) / predictions.size()) * 100.0f << "%\n";
            }
        }
    }
    return { w1, b1, w2, b2, w3, b3 };
}

float precision(const std::vector<uint8_t>& predictions, const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& trueLabels) {
    size_t true_positive = 0;
    size_t false_positive = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == trueLabels(i)) {
            true_positive++;
        } else {
            false_positive++;
        }
    }
    return static_cast<float>(true_positive) / (true_positive + false_positive);
}

float recall(const std::vector<uint8_t>& predictions, const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& trueLabels) {
    size_t true_positive = 0;
    size_t false_negative = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == trueLabels(i)) {
            true_positive++;
        } else {
            false_negative++;
        }
    }
    return static_cast<float>(true_positive) / (true_positive + false_negative);
}

float f1_score(const std::vector<uint8_t>& predictions, const Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>& trueLabels) {
    float prec = precision(predictions, trueLabels);
    float rec = recall(predictions, trueLabels);
    return 2 * (prec * rec) / (prec + rec);
}

void test_metrics(const MnistData& mnistData) {
    Eigen::MatrixXf w1, b1, w2, b2, w3, b3;
    std::ifstream inFile("./trained_model/model.bin", std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: model.bin\n";
        return;
    }
    w1 = readMatrixFromBinaryFile(inFile);
    b1 = readMatrixFromBinaryFile(inFile);
    w2 = readMatrixFromBinaryFile(inFile);
    b2 = readMatrixFromBinaryFile(inFile);
    w3 = readMatrixFromBinaryFile(inFile);
    b3 = readMatrixFromBinaryFile(inFile);

    Eigen::MatrixXf X_test = mnistData.test_images.cast<float>() / 255.0f;
    Eigen::MatrixXf Y_test = oneHotEncode(mnistData.test_labels);

    auto [z1_test, a1_test, z2_test, a2_test, z3_test, a3_test] = forward_prop(w1, b1, w2, b2, w3, b3, X_test);
    auto predictions = get_predictions(a3_test);
    size_t correct_predictions = countMatchingLabels(predictions, mnistData.test_labels);
    std::cout << "Accuracy on test set: "
        << (static_cast<float>(correct_predictions) / predictions.size()) * 100.0f << "%\n";
    
    std::cout << "F1 Score: " << f1_score(predictions, mnistData.test_labels) << "\n";
    std::cout << "Precision: " << precision(predictions, mnistData.test_labels) << "\n";
    std::cout << "Recall: " << recall(predictions, mnistData.test_labels) << "\n";
}

int main() {

    Eigen::MatrixXf w1, b1, w2, b2, w3, b3;
    std::cout << "Training model...\n";

    MnistData mnistDataLoader(GET_TRAIN_SET | GET_DEV_SET | GET_TEST_SET, 2000);
    auto res_tuple = train(mnistDataLoader);
    w1 = std::get<0>(res_tuple);
    b1 = std::get<1>(res_tuple);
    w2 = std::get<2>(res_tuple);
    b2 = std::get<3>(res_tuple);
    w3 = std::get<4>(res_tuple);
    b3 = std::get<5>(res_tuple);


    std::ofstream outFile("./trained_model/model.bin", std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: model.bin\n";
        return 1;
    }
    outFile.clear();
    writeMatrixToBinaryFile(outFile, w1);
    writeMatrixToBinaryFile(outFile, b1);
    writeMatrixToBinaryFile(outFile, w2);
    writeMatrixToBinaryFile(outFile, b2);
    writeMatrixToBinaryFile(outFile, w3);
    writeMatrixToBinaryFile(outFile, b3);
    outFile.close();

    test_metrics(mnistDataLoader);

    return 0;
}
