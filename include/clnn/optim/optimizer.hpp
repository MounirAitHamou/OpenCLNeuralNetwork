#pragma once

#include "clnn/autograd/tensor.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace clnn::optim {

struct ParameterGroup final {
    std::vector<Tensor*> parameters;
    float learning_rate;
    float weight_decay = 0.0F;
};

struct ParameterGroups final {
    explicit ParameterGroups(std::vector<ParameterGroup> values) : groups(std::move(values)) {}
    std::vector<ParameterGroup> groups;
};

class Optimizer {
  public:
    explicit Optimizer(std::vector<Tensor*> parameters);
    explicit Optimizer(std::vector<ParameterGroup> parameter_groups);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void zero_grad();
    [[nodiscard]] std::size_t parameter_group_count() const noexcept;
    [[nodiscard]] float learning_rate(std::size_t group = 0) const;
    virtual void set_learning_rate(float learning_rate, std::size_t group = 0);

  protected:
    std::vector<Tensor*> parameters_;
    std::vector<ParameterGroup> parameter_groups_;
    std::vector<std::size_t> parameter_group_indices_;
    friend void save_state_dict(const Optimizer&, const std::filesystem::path&);
    friend void load_state_dict(Optimizer&, const std::filesystem::path&);
};

class SGD final : public Optimizer {
  public:
    SGD(std::vector<Tensor*> parameters, float learning_rate, float momentum = 0.0F,
        float weight_decay = 0.0F);
    explicit SGD(ParameterGroups parameter_groups, float momentum = 0.0F);
    void step() override;

  private:
    float learning_rate_;
    float momentum_;
    float weight_decay_;
    std::vector<std::vector<float>> velocity_;
    std::vector<std::shared_ptr<opencl::Buffer>> velocity_buffers_;
    friend void save_state_dict(const Optimizer&, const std::filesystem::path&);
    friend void load_state_dict(Optimizer&, const std::filesystem::path&);
};

class Adam final : public Optimizer {
  public:
    Adam(std::vector<Tensor*> parameters, float learning_rate = 1.0e-3F, float beta1 = 0.9F,
         float beta2 = 0.999F, float epsilon = 1.0e-8F, float weight_decay = 0.0F,
         bool decoupled_weight_decay = false);
    explicit Adam(ParameterGroups parameter_groups, float beta1 = 0.9F, float beta2 = 0.999F,
                  float epsilon = 1.0e-8F, bool decoupled_weight_decay = false);
    void step() override;

  private:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    bool decoupled_weight_decay_;
    std::size_t step_ = 0;
    std::vector<std::vector<float>> first_moment_;
    std::vector<std::vector<float>> second_moment_;
    std::vector<std::shared_ptr<opencl::Buffer>> first_moment_buffers_;
    std::vector<std::shared_ptr<opencl::Buffer>> second_moment_buffers_;
    friend void save_state_dict(const Optimizer&, const std::filesystem::path&);
    friend void load_state_dict(Optimizer&, const std::filesystem::path&);
};

class AdamW final : public Optimizer {
  public:
    AdamW(std::vector<Tensor*> parameters, float learning_rate = 1.0e-3F, float beta1 = 0.9F,
          float beta2 = 0.999F, float epsilon = 1.0e-8F, float weight_decay = 1.0e-2F);
    explicit AdamW(ParameterGroups parameter_groups, float beta1 = 0.9F, float beta2 = 0.999F,
                   float epsilon = 1.0e-8F);
    void step() override;
    void set_learning_rate(float learning_rate, std::size_t group = 0) override;

  private:
    Adam implementation_;
    friend void save_state_dict(const Optimizer&, const std::filesystem::path&);
    friend void load_state_dict(Optimizer&, const std::filesystem::path&);
};

class LRScheduler {
  public:
    explicit LRScheduler(Optimizer& optimizer);
    virtual ~LRScheduler() = default;
    virtual void step() = 0;
    [[nodiscard]] std::size_t steps() const noexcept;

  protected:
    Optimizer* optimizer_;
    std::vector<float> initial_learning_rates_;
    std::size_t steps_ = 0;
};

class ExponentialLR final : public LRScheduler {
  public:
    ExponentialLR(Optimizer& optimizer, float gamma);
    void step() override;

  private:
    float gamma_;
};

class StepLR final : public LRScheduler {
  public:
    StepLR(Optimizer& optimizer, std::size_t step_size, float gamma = 0.1F);
    void step() override;

  private:
    std::size_t step_size_;
    float gamma_;
};

class CosineAnnealingLR final : public LRScheduler {
  public:
    CosineAnnealingLR(Optimizer& optimizer, std::size_t maximum_steps,
                      float minimum_learning_rate = 0.0F);
    void step() override;

  private:
    std::size_t maximum_steps_;
    float minimum_learning_rate_;
};

void save_state_dict(const Optimizer& optimizer, const std::filesystem::path& path);
void load_state_dict(Optimizer& optimizer, const std::filesystem::path& path);

} // namespace clnn::optim
