#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <tuple>
#include <vector>

const std::filesystem::path MNIST_INPUT_DIR = "./mnist-input";
const std::filesystem::path TRAIN_IMAGES_PATH = MNIST_INPUT_DIR / "train-images.idx3-ubyte";
const std::filesystem::path TRAIN_LABELS_PATH = MNIST_INPUT_DIR / "train-labels.idx1-ubyte";
const std::filesystem::path TEST_IMAGES_PATH = MNIST_INPUT_DIR / "t10k-images.idx3-ubyte";
const std::filesystem::path TEST_LABELS_PATH = MNIST_INPUT_DIR / "t10k-labels.idx1-ubyte";

constexpr uint8_t GET_TRAIN_SET = 0x01;
constexpr uint8_t GET_TEST_SET = 0x02;
constexpr uint8_t GET_DEV_SET = 0x04;

struct MnistData {
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> train_images;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> dev_images;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> test_images;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> train_labels;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> dev_labels;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> test_labels;

    bool uses_dev_set = false;

    MnistData(uint8_t get_flag, const size_t dev_size = 0) {
        if(get_flag & GET_TRAIN_SET) {
            loadTrainFiles();
            if(get_flag & GET_DEV_SET && dev_size > 0) {
                splitTrainDevData(dev_size);
                uses_dev_set = true;
            }
        }
        if(get_flag & GET_TEST_SET)
            loadTestFiles();
    }

    void loadTrainFiles(){
        try {
            auto [train_images, train_labels] = read_images_labels(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
            this->train_images = train_images;
            this->train_labels = train_labels;
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            exit(1);
        }
    }

    void loadTestFiles(){
        try {
            auto [test_images, test_labels] = read_images_labels(TEST_IMAGES_PATH, TEST_LABELS_PATH);
            this->test_images = test_images;
            this->test_labels = test_labels;
        }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            exit(1);
        }
    }

    void splitTrainDevData(const size_t dev_size) {
        dev_images = train_images.rightCols(dev_size);
        dev_labels = train_labels.bottomRows(dev_size);

        train_images.conservativeResize(Eigen::NoChange, train_images.cols() - dev_size);
        train_labels.conservativeResize(Eigen::NoChange, train_labels.rows() - dev_size);
    }

    void showData() {
        std::cout << "Train images: " << train_images.rows() << "x" << train_images.cols() << "\n";
        std::cout << "Train labels: " << train_labels.rows() << "x" << train_labels.cols() << "\n";
        std::cout << "Dev images: " << dev_images.rows() << "x" << dev_images.cols() << "\n";
        std::cout << "Dev labels: " << dev_labels.rows() << "x" << dev_labels.cols() << "\n";
        std::cout << "Test images: " << test_images.rows() << "x" << test_images.cols() << "\n";
        std::cout << "Test labels: " << test_labels.rows() << "x" << test_labels.cols() << "\n";
    }

private:

    static uint32_t read_uint32(std::ifstream& file) {
        uint32_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        return __builtin_bswap32(value); // Convert from big-endian to little-endian
    }

    static std::tuple<Eigen::MatrixX<uint8_t>, Eigen::MatrixX<uint8_t>> read_images_labels(const std::filesystem::path image_filepath, const std::filesystem::path label_filepath) {
        std::ifstream
            image_file(image_filepath, std::ios::binary),
            label_file(label_filepath, std::ios::binary);

        if (!image_file.is_open()) {
            throw std::runtime_error("Error: Could not open file for images " + image_filepath.string());
        }
        if (!label_file.is_open()) {
            throw std::runtime_error("Error: Could not open file for labels" + label_filepath.string());
        }

        uint32_t magic_number = read_uint32(image_file);
        uint32_t num_images = read_uint32(image_file);
        uint32_t num_rows = read_uint32(image_file);
        uint32_t num_cols = read_uint32(image_file);

        // std::cout << "magic_number: " << magic_number << "\n";
        // std::cout << "num_images: " << num_images << "\n";
        // std::cout << "num_rows: " << num_rows << "\n";
        // std::cout << "num_cols: " << num_cols << "\n";

        std::vector<uint8_t> images_data(num_rows * num_cols * num_images);
        image_file.read(reinterpret_cast<char*>(images_data.data()), images_data.size());
        image_file.close();

        Eigen::MatrixX<uint8_t> images(num_rows * num_cols, num_images);

        for (int i = 0; i < num_images; ++i) {
            for (int j = 0; j < num_rows * num_cols; ++j) {
                uint8_t pixel;
                pixel = images_data[i * num_rows * num_cols + j];
                images(j, i) = pixel;
            }
        }

        magic_number = read_uint32(label_file);
        num_images = read_uint32(label_file);

        std::vector<uint8_t> labels_data(num_images);
        label_file.read(reinterpret_cast<char*>(labels_data.data()), labels_data.size());
        label_file.close();

        Eigen::MatrixX<uint8_t> labels(num_images, 1);
        for (int i = 0; i < num_images; ++i) {
            uint8_t label;
            label = labels_data[i];
            labels(i, 0) = label;
        }

        return { images, labels };
    }

};

#endif