#include "NeuralNetwork/NeuralNetwork.hpp"
#include <chrono>

int XORTest(std::shared_ptr<Utils::SharedResources> p_sharedResources, NeuralNetwork& net){
    size_t batchSize = 1;
    CSVNumericalLoader csvLoader(p_sharedResources, batchSize);
    csvLoader.loadData("data/XOR/xor_data.csv", {"bit1", "bit2"}, {"outputbit"});
    unsigned int seed;
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    csvLoader.splitData(1.0f, 0.0f, seed);
    csvLoader.activateTrainPartition();
    csvLoader.shuffleCurrentPartition();

    std::cout << "\nTesting:\n";
    for (const Batch& batch : csvLoader){
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
    size_t batchSize = 1;
    float learningRate = 0.001f;
    float weightDecayRate = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int epochs = 10000;
    bool lossReporting = true;

    CSVNumericalLoader csvLoader(oclResources.getSharedResources(), batchSize);
    csvLoader.loadData("data/XOR/xor_data.csv", {"bit1", "bit2"}, {"outputbit"});
    size_t seed;
    seed = static_cast<size_t>(std::chrono::system_clock::now().time_since_epoch().count());
    csvLoader.splitData(1.0f, 0.0f, seed);
    const Utils::Dimensions initialInputDimensions = Utils::Dimensions({csvLoader.getInputSize()});
    const size_t flatInputSize = initialInputDimensions.getTotalElements();
    const size_t flatOutputSize = csvLoader.getTargetSize();

    NeuralNetwork loadedNet;
    NeuralNetwork net;
    if (!std::filesystem::exists(p_fileName)){
        Utils::LossFunctionType lossFunctionType = Utils::LossFunctionType::BinaryCrossEntropy;

        auto optimizerArgs = Utils::makeAdamWArgs(
            learningRate,
            weightDecayRate,
            beta1,
            beta2,
            epsilon
        );

        net = NeuralNetwork(std::move(oclResources), Utils::createNetworkArgs(
            initialInputDimensions,    
            optimizerArgs,
            batchSize,
            lossFunctionType
        ), seed);

        net.addDense(32, Utils::ActivationType::Tanh)
            .addConvolutional(Utils::FilterDimensions(1,1,32,24),
                                Utils::StrideDimensions(1,1),
                                Utils::PaddingType::Same,
                                Utils::ActivationType::ReLU)
            .addConvolutional(Utils::FilterDimensions(1,1,24,16),
                                Utils::StrideDimensions(1,1),
                                Utils::PaddingType::Same,
                                Utils::ActivationType::ReLU)
            .addConvolutional(Utils::FilterDimensions(1,1,16,8),
                                Utils::StrideDimensions(1,1),
                                Utils::PaddingType::Same,
                                Utils::ActivationType::ReLU)
            .addDense(16, Utils::ActivationType::Tanh)
           .addDense(flatOutputSize, Utils::ActivationType::Sigmoid);
        
        net.train(
            csvLoader,
            epochs,
            lossReporting
        );
        

        XORTest(net.getSharedResources(), net);

        std::cout << "\nSaving network to file\n";
        net.saveNetwork(p_fileName);

        loadedNet = NeuralNetwork::loadNetwork(net.getSharedResources(), p_fileName);

        if (net.equals(loadedNet)) { 
            std::cout << "Loaded network is equivalent to initial network\n";
        }
        else {
            std::cerr << "Loaded network not equivalent to initial network\n";
            return -1;
        }


        

    }
    else {
        loadedNet = NeuralNetwork::loadNetwork(oclResources.getSharedResources(), p_fileName);
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
        loadedNet.saveNetwork(p_fileName);
    }
   return 0;
}

int main() {
    Utils::OpenCLResources oclResources = Utils::OpenCLResources::createOpenCLResources();

    if (oclResources.getErrorCode() != 0) {
        std::cerr << "Failed to initialize OpenCL resources. Error code: " << oclResources.getErrorCode() << std::endl;
        return -1;
    }
    makeXORModel(std::move(oclResources), "xor_network.h5");

}