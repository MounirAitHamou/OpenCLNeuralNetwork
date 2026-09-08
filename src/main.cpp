#include "clnn/clnn.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string_view>

int train_and_save_cifar() {
    using namespace clnn;

    if (!opencl_available()) {
        std::cerr << "No OpenCL GPU is available. Install a vendor OpenCL driver.\n";
        return 1;
    }
    const auto device = Device::opencl();
    std::cout << "Training on " << device.name() << '\n';

    constexpr std::size_t batch_size = 64;
    constexpr std::size_t epochs = 2;
    const std::filesystem::path data_path = "data/CIFAR-10/data_batch_1.bin";
    const std::filesystem::path checkpoint_path = "cifar10.clnn";
    auto optimizer_path = checkpoint_path;
    optimizer_path += ".optimizer";

    std::cout << "Loading CIFAR-10 from " << data_path << '\n';
    auto dataset = data::load_cifar10_batch(data_path);
    auto partitions = data::split(dataset, 0.9F, 0.1F, 42);
    data::DataLoader training(partitions.training, batch_size, device, true, 42);
    data::DataLoader validation(partitions.validation, batch_size, device);

    std::mt19937 generator(42);
    std::unique_ptr<nn::Sequential> model;
    if (std::filesystem::exists(checkpoint_path)) {
        model = nn::load_checkpoint(checkpoint_path, device);
        std::cout << "Resumed model from " << checkpoint_path << '\n';
    } else {
        model = std::make_unique<nn::Sequential>();
        model
            ->add(std::make_unique<nn::Conv2d>(3, 32, std::array<std::size_t, 2>{3, 3}, generator,
                                               std::array<std::size_t, 2>{1, 1},
                                               nn::PaddingMode::same, true, device))
            .add(std::make_unique<nn::ReLU>())
            .add(std::make_unique<nn::Conv2d>(32, 64, std::array<std::size_t, 2>{3, 3}, generator,
                                              std::array<std::size_t, 2>{2, 2},
                                              nn::PaddingMode::same, true, device))
            .add(std::make_unique<nn::ReLU>())
            .add(std::make_unique<nn::Conv2d>(64, 128, std::array<std::size_t, 2>{3, 3}, generator,
                                              std::array<std::size_t, 2>{2, 2},
                                              nn::PaddingMode::same, true, device))
            .add(std::make_unique<nn::ReLU>())
            .add(std::make_unique<nn::Flatten>())
            .add(std::make_unique<nn::Linear>(128 * 8 * 8, 256, generator, true, device))
            .add(std::make_unique<nn::ReLU>())
            .add(std::make_unique<nn::Linear>(256, 10, generator, true, device));
    }

    optim::AdamW optimizer(model->parameters(), 1.0e-3F, 0.9F, 0.999F, 1.0e-8F, 1.0e-4F);
    if (std::filesystem::exists(optimizer_path)) {
        optim::load_state_dict(optimizer, optimizer_path);
        std::cout << "Resumed optimizer state from " << optimizer_path << '\n';
    }

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        training.reset(static_cast<std::uint32_t>(42 + epoch));
        double accumulated_loss = 0.0;
        std::size_t trained_samples = 0;
        while (auto batch = training.next()) {
            optimizer.zero_grad();
            auto loss = cross_entropy((*model)(batch->inputs), batch->class_labels);
            accumulated_loss += static_cast<double>(loss.item()) * batch->size;
            trained_samples += batch->size;
            loss.backward();
            optimizer.step();
        }

        validation.reset();
        std::size_t correct = 0;
        std::size_t validated_samples = 0;
        {
            NoGradGuard no_grad;
            while (auto batch = validation.next()) {
                const auto logits = (*model)(batch->inputs).data();
                for (std::size_t row = 0; row < batch->size; ++row) {
                    const auto first = logits.begin() + static_cast<std::ptrdiff_t>(row * 10);
                    const auto prediction = static_cast<std::size_t>(
                        std::distance(first, std::max_element(first, first + 10)));
                    if (prediction == batch->class_labels[row])
                        ++correct;
                }
                validated_samples += batch->size;
            }
        }

        const auto mean_loss = accumulated_loss / static_cast<double>(trained_samples);
        const auto accuracy = validated_samples == 0 ? 0.0
                                                     : static_cast<double>(correct) /
                                                           static_cast<double>(validated_samples);
        std::cout << "epoch " << (epoch + 1) << '/' << epochs << "  loss " << mean_loss
                  << "  validation accuracy " << std::fixed << std::setprecision(2)
                  << accuracy * 100.0 << "%\n";

        nn::save_checkpoint(*model, checkpoint_path);
        optim::save_state_dict(optimizer, optimizer_path);
    }

    validation.reset();
    if (auto batch = validation.next()) {
        std::size_t predicted_class = 0;
        {
            NoGradGuard no_grad;
            const auto logits = (*model)(batch->inputs).data();
            predicted_class = static_cast<std::size_t>(
                std::distance(logits.begin(),
                              std::max_element(logits.begin(), logits.begin() + 10)));
        }
        const auto baseline = Tensor::zeros(batch->inputs.shape(), false, {}, device);
        const auto attribution = explain::integrated_gradients(
            *model, batch->inputs, {predicted_class, 0}, baseline, 32);
        std::cout << "Explained validation class " << predicted_class << " with attribution shape "
                  << shape_string(attribution.shape()) << " on " << attribution.device().name()
                  << '\n';
    }

    std::cout << "Saved CIFAR-10 checkpoint to " << checkpoint_path << '\n';
    return 0;
}

int train_and_save_xor() {
    using namespace clnn;

    if (!opencl_available()) {
        std::cerr << "No OpenCL GPU is available. Install a vendor OpenCL driver.\n";
        return 1;
    }
    const auto device = Device::opencl();
    std::cout << "Training on " << device.name() << '\n';

    std::mt19937 generator(42);
    nn::Sequential model;
    model.add(std::make_unique<nn::Linear>(2, 8, generator, true, device))
        .add(std::make_unique<nn::Tanh>())
        .add(std::make_unique<nn::Linear>(8, 1, generator, true, device));

    const Tensor inputs({0, 0, 0, 1, 1, 0, 1, 1}, {4, 2}, false, {}, device);
    const Tensor targets({0, 1, 1, 0}, {4, 1}, false, {}, device);
    optim::Adam optimizer(model.parameters(), 0.03F);

    for (int epoch = 0; epoch < 1'000; ++epoch) {
        optimizer.zero_grad();
        auto loss = binary_cross_entropy_with_logits(model(inputs), targets);
        loss.backward();
        optimizer.step();
        if (epoch % 200 == 0) {
            std::cout << "epoch " << std::setw(4) << epoch << "  loss " << loss.item() << '\n';
        }
    }

    NoGradGuard no_grad;
    const auto predictions = sigmoid(model(inputs));
    std::cout << "\nXOR probabilities:\n";
    for (std::size_t row = 0; row < 4; ++row) {
        std::cout << static_cast<int>(inputs.data()[row * 2]) << " xor "
                  << static_cast<int>(inputs.data()[row * 2 + 1]) << " = " << std::fixed
                  << std::setprecision(4) << predictions.data()[row] << '\n';
    }
    clnn::nn::save_state_dict(model, "xor.clnn");
    return 0;
}

int main(const int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--cifar") {
        return train_and_save_cifar();
    }
    return train_and_save_xor();
}
