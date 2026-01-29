#include "NeuralNetworks/Local/LocalNeuralNetwork.hpp"
#include <chrono>


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>

int XORTest(std::shared_ptr<Utils::SharedResources> p_sharedResources, NeuralNetworks::Local::LocalNeuralNetwork& net){
    size_t batchSize = 1;
    DataLoaders::CSVNumericalLoader csvLoader(p_sharedResources, batchSize, {"bit1", "bit2"}, {"outputbit"});
    csvLoader.loadData("data/XOR/xor_data.csv");
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csvLoader.splitData(1.0f, 0.0f, seed);
    csvLoader.activateTrainPartition();
    csvLoader.shuffleCurrentPartition(seed);

    std::cout << "\nTesting:\n";
    for (const Utils::Batch& batch : csvLoader){
        std::vector<float> inputsVec = batch.getInputsVector();
        std::vector<float> targetsVec = batch.getTargetsVector();
        std::vector<float> prediction = net.predict(batch.getInputs(), batch.getSize());

        std::cout << "Input: (" << inputsVec[0] << ", " << inputsVec[1] << ")"
          << " | Predicted: " << prediction[0]
          << " | Target: " << targetsVec[0] << std::endl;
    }
    return 0;
}

int makeXORModel(Utils::OpenCLResources oclResources, const std::string p_fileName) {
    size_t batchSize = 3;
    float learningRate = 0.001f;
    float weightDecayRate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int epochs = 3000;
    bool lossReporting = true;

    DataLoaders::CSVNumericalLoader csvLoader(oclResources.getSharedResources(), batchSize, {"bit1", "bit2"}, {"outputbit"});
    csvLoader.loadData("data/XOR/xor_data.csv");
    size_t seed;
    seed = static_cast<size_t>(std::chrono::system_clock::now().time_since_epoch().count());
    csvLoader.splitData(1.0f, 0.0f, seed);
    const Utils::Dimensions initialInputDimensions = Utils::Dimensions({csvLoader.getInputSize()});
    const size_t flatInputSize = initialInputDimensions.getTotalElements();
    const size_t flatOutputSize = csvLoader.getTargetSize();

    NeuralNetworks::Local::LocalNeuralNetwork loadedNet;
    NeuralNetworks::Local::LocalNeuralNetwork net;
    if (!std::filesystem::exists(p_fileName)){
        auto lossFunctionArgs = Utils::makeBinaryCrossEntropyLossFunctionArgs();

        auto optimizerArgs = Utils::makeAdamWArgs(
            learningRate,
            weightDecayRate,
            beta1,
            beta2,
            epsilon
        );
        net = NeuralNetworks::Local::LocalNeuralNetwork(std::move(oclResources), Utils::createNetworkArgs(
            initialInputDimensions,
            {},
            std::move(optimizerArgs),
            std::move(lossFunctionArgs)
        ), seed, batchSize);

        net.addDense(32)
           .addTanh()
           .addConvolutional(Utils::FilterDimensions(1,1,32,24),
                             Utils::StrideDimensions(1,1),
                             Utils::PaddingType::Same)
           .addReLU()
           .addConvolutional(Utils::FilterDimensions(1,1,24,16),
                             Utils::StrideDimensions(1,1),
                             Utils::PaddingType::Same)
           .addReLU()
           .addConvolutional(Utils::FilterDimensions(1,1,16,8),
                             Utils::StrideDimensions(1,1),
                             Utils::PaddingType::Same)
           .addReLU()
           .addDense(16)
           .addTanh()
           .addDense(flatOutputSize)
           .addSigmoid();
           
        net.train(
            csvLoader,
            epochs,
            lossReporting
        );

        XORTest(net.getSharedResources(), net);

        std::cout << "\nSaving network to file\n";
        net.save(p_fileName);

        loadedNet = NeuralNetworks::Local::LocalNeuralNetwork::load(net.getSharedResources(), p_fileName, batchSize);

        if (net.equals(loadedNet)) { 
            std::cout << "Loaded network is equivalent to initial network\n";
        }
        else {
            std::cerr << "Loaded network not equivalent to initial network\n";
            return -1;
        }
    }
    else {
        loadedNet = NeuralNetworks::Local::LocalNeuralNetwork::load(oclResources.getSharedResources(), p_fileName, batchSize);
    }
    
    while (true){
        std::cout << "\nTesting loaded network:\n";
        XORTest(loadedNet.getSharedResources(), loadedNet);
        loadedNet.train(
            csvLoader,
            epochs,
            lossReporting
        );

        std::cout << "\nTesting after retraining:\n";
        XORTest(loadedNet.getSharedResources(), loadedNet);
        std::cout << "\nCheckpointing again...\n";
        loadedNet.save(p_fileName);
    }
   return 0;
}

int makeCIFARModel(Utils::OpenCLResources oclResources, const std::string p_fileName) {
    size_t batchSize = 10;
    float learningRate = 1e-3f;
    float weightDecayRate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int epochs = 2;
    bool lossReporting = true;

    DataLoaders::BinImageDataLoader cifarLoader(
        oclResources.getSharedResources(),
        batchSize,
        32,
        32,
        3,
        true,
        DataLoaders::BinImageDataLoader::DataOrder::CHW,
        DataLoaders::BinImageDataLoader::DataOrder::CHW,
        10
    );

    cifarLoader.loadData("data/CIFAR-10/data_batch_1.bin");

    size_t seed =
        static_cast<size_t>(std::chrono::system_clock::now().time_since_epoch().count());

    cifarLoader.splitData(0.8f, 0.1f, seed);
    cifarLoader.activateTrainPartition();
    cifarLoader.shuffleCurrentPartition(seed);

    const Utils::Dimensions inputDims({3, 32, 32});
    const size_t outputSize = 10;

    NeuralNetworks::Local::LocalNeuralNetwork net;

    if (!std::filesystem::exists(p_fileName)) {
        auto lossFunctionArgs = Utils::makeMeanSquaredErrorLossFunctionArgs();

        auto optimizerArgs = Utils::makeAdamWArgs(
            learningRate,
            weightDecayRate,
            beta1,
            beta2,
            epsilon
        );
        net = NeuralNetworks::Local::LocalNeuralNetwork(
            std::move(oclResources),
            Utils::createNetworkArgs(
                inputDims,
                {},
                std::move(optimizerArgs),
                std::move(lossFunctionArgs)
            ),
            seed,
            batchSize
        );

        net.addConvolutional(
                Utils::FilterDimensions(3, 3, 3, 32),
                Utils::StrideDimensions(1, 1),
                Utils::PaddingType::Same
            )
            .addReLU()
            .addConvolutional(
                Utils::FilterDimensions(3, 3, 32, 64),
                Utils::StrideDimensions(2, 2),
                Utils::PaddingType::Same
            )
            .addReLU()
            .addConvolutional(
                Utils::FilterDimensions(3, 3, 64, 128),
                Utils::StrideDimensions(2, 2),
                Utils::PaddingType::Same
            )
            .addReLU()
            .addDense(256)
            .addReLU()
            .addDense(outputSize);
        net.train(cifarLoader, epochs, lossReporting);
        net.save(p_fileName);
    } else {
        net = NeuralNetworks::Local::LocalNeuralNetwork::load(
            oclResources.getSharedResources(),
            p_fileName,
            batchSize
        );
    }

    return 0;
}


int main(void) {
    Utils::OpenCLResources oclResources = Utils::OpenCLResources::createOpenCLResources();
    makeXORModel(std::move(oclResources), "xor_network.h5");
    return 0;
}