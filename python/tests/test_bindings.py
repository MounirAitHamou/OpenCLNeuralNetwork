import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

import clnn
from clnn import data, explain, nn, optim


class TensorTests(unittest.TestCase):
    def test_numpy_round_trip_and_broadcast_gradient(self) -> None:
        x = clnn.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                        requires_grad=True, name="x")
        bias = clnn.Tensor(np.array([0.5, -0.5], dtype=np.float32), requires_grad=True)

        loss = ((x + bias) * (x + bias)).mean()
        loss.backward()

        np.testing.assert_allclose(loss.item(), 7.25, rtol=1e-6)
        np.testing.assert_allclose(x.grad, [[0.75, 0.75], [1.75, 1.75]], rtol=1e-6)
        np.testing.assert_allclose(bias.grad, [2.5, 2.5], rtol=1e-6)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.numpy().dtype, np.float32)

    def test_no_grad_context(self) -> None:
        x = clnn.Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        self.assertTrue(clnn.grad_enabled())
        with clnn.no_grad():
            self.assertFalse(clnn.grad_enabled())
            result = x * 3.0
        self.assertTrue(clnn.grad_enabled())
        self.assertFalse(result.requires_grad)

    def test_explicit_gradient(self) -> None:
        x = clnn.Tensor(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
        y = x * x
        y.backward(clnn.Tensor(np.array([1.0, 0.5], dtype=np.float32)))
        np.testing.assert_allclose(x.grad, [4.0, 3.0], rtol=1e-6)

    def test_axis_reductions_and_pooling(self) -> None:
        matrix = clnn.Tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3),
                             requires_grad=True)
        reduced = matrix.sum(0, keep_dimensions=True)
        self.assertEqual(reduced.shape, (1, 3))
        np.testing.assert_allclose(reduced.numpy(), [[5.0, 7.0, 9.0]])

        image = clnn.Tensor(np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3),
                            requires_grad=True)
        pooled = clnn.max_pool2d(image, (2, 2), (1, 1))
        np.testing.assert_allclose(pooled.numpy(), [[[[5.0, 6.0], [8.0, 9.0]]]])
        pooled.sum().backward()
        np.testing.assert_allclose(image.grad,
                                   [[[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]])


class ModuleAndOptimizerTests(unittest.TestCase):
    def test_sequential_owns_layers_and_trains(self) -> None:
        model = nn.Sequential()
        model.add(nn.Linear(1, 1, seed=7))
        self.assertEqual(len(model), 1)
        self.assertIsInstance(model[0], nn.Linear)
        self.assertEqual(model[0].type, nn.ModuleType.LINEAR)
        self.assertEqual(set(model.named_parameters()), {"0.weight", "0.bias"})
        gc.collect()

        inputs = clnn.Tensor(np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32))
        targets = clnn.Tensor(np.array([[1.0], [3.0], [5.0], [7.0]], dtype=np.float32))
        optimizer = optim.Adam(model, learning_rate=0.08)

        initial_loss = clnn.mse_loss(model(inputs), targets).item()
        for _ in range(160):
            optimizer.zero_grad()
            loss = clnn.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        final_loss = clnn.mse_loss(model(inputs), targets).item()

        self.assertLess(final_loss, initial_loss * 0.01)

    def test_state_dict_round_trip(self) -> None:
        source = nn.Linear(2, 1, seed=11)
        destination = nn.Linear(2, 1, seed=99)
        inputs = clnn.Tensor(np.array([[2.0, -1.0]], dtype=np.float32))

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "weights.clnn"
            nn.save_state_dict(source, path)
            nn.load_state_dict(destination, path)

        np.testing.assert_allclose(source(inputs).numpy(), destination(inputs).numpy())

    def test_train_eval_dropout_and_batch_normalization(self) -> None:
        model = nn.Sequential()
        model.add(nn.BatchNorm(2))
        model.add(nn.Dropout(0.5, seed=4))
        values = clnn.Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                      dtype=np.float32))

        training = model(values).numpy()
        self.assertTrue(model.training)
        self.assertIn(0.0, training)
        self.assertEqual(set(model.named_buffers()),
                         {"0.running_mean", "0.running_variance"})
        model.eval()
        self.assertFalse(model.training)
        evaluated = model(values).numpy()
        self.assertTrue(np.isfinite(evaluated).all())
        self.assertFalse(np.array_equal(training, evaluated))

    def test_layer_norm_global_pool_residual_and_schedulers(self) -> None:
        values = clnn.Tensor(np.array([[[1.0, 2.0], [4.0, 8.0]]], dtype=np.float32))
        normalized = nn.LayerNorm([2, 2], affine=False)(values).numpy()
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(np.square(normalized).mean()), 1.0, places=4)

        image = clnn.Tensor(np.arange(1, 9, dtype=np.float32).reshape(1, 2, 2, 2))
        np.testing.assert_allclose(nn.GlobalAvgPool2d()(image).numpy(), [[2.5, 6.5]])
        residual = nn.Residual(nn.ReLU())
        residual_input = clnn.Tensor(np.array([[-2.0, 3.0]], dtype=np.float32))
        np.testing.assert_allclose(residual(residual_input).numpy(), [[-2.0, 6.0]])

        first = nn.Linear(1, 1, seed=1)
        second = nn.Linear(1, 1, seed=2)
        optimizer = optim.SGD([
            optim.ParameterGroup(first, learning_rate=0.1),
            optim.ParameterGroup(second, learning_rate=0.02),
        ])
        scheduler = optim.StepLR(optimizer, step_size=1, gamma=0.5)
        scheduler.step()
        self.assertAlmostEqual(optimizer.learning_rate(0), 0.05)
        self.assertAlmostEqual(optimizer.learning_rate(1), 0.01)


class ExplainabilityTests(unittest.TestCase):
    def make_linear(self) -> nn.Linear:
        model = nn.Linear(2, 2, seed=1)
        model.weight.set_data(np.array([[2.0, -1.0], [-3.0, 4.0]], dtype=np.float32))
        model.bias.set_data(np.array([0.5, -0.25], dtype=np.float32))
        return model

    def test_attributions_and_parameter_gradients(self) -> None:
        model = self.make_linear()
        training_input = clnn.Tensor(np.array([[1.0, 2.0]], dtype=np.float32),
                                     requires_grad=True)
        model(training_input).sum().backward()
        weight_gradient = model.weight.grad.copy()
        input = clnn.Tensor(np.array([[1.5, -2.0]], dtype=np.float32))
        baseline = clnn.Tensor.zeros(input.shape)

        np.testing.assert_allclose(explain.saliency(model, input, target=0).numpy(),
                                   np.array([[2.0, 3.0]], dtype=np.float32))
        np.testing.assert_allclose(
            explain.saliency(model, input, target=explain.Target(0)).numpy(),
            np.array([[2.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(explain.input_x_gradient(model, input, target=0).numpy(),
                                   np.array([[3.0, 6.0]], dtype=np.float32))
        np.testing.assert_allclose(
            explain.integrated_gradients(model, input, target=0, baseline=baseline,
                                         steps=8).numpy(),
            np.array([[3.0, 6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(model.weight.grad, weight_gradient)

    def test_smoothgrad_batch_selection_and_forward_hook(self) -> None:
        model = self.make_linear()
        input = clnn.Tensor(np.array([[1.0, 2.0], [-1.0, 3.0]], dtype=np.float32))
        first = explain.smoothgrad(model, input, target=1, samples=5,
                                   noise_stddev=0.1, seed=42, batch_index=1)
        second = explain.smoothgrad(model, input, target=1, samples=5,
                                    noise_stddev=0.1, seed=42, batch_index=1)
        np.testing.assert_array_equal(first.numpy(), second.numpy())
        np.testing.assert_allclose(first.numpy()[0], 0.0)

        observed: list[clnn.Tensor] = []
        handle = model.register_forward_hook(
            lambda _module, _input, output: observed.append(output)
        )
        model(input)
        self.assertEqual(len(observed), 1)
        self.assertTrue(handle.active)
        handle.remove()
        model(input)
        self.assertEqual(len(observed), 1)

    def test_validation(self) -> None:
        model = self.make_linear()
        input = clnn.Tensor(np.ones((1, 2), dtype=np.float32))
        with self.assertRaises(IndexError):
            explain.saliency(model, input, target=2)
        with self.assertRaises(ValueError):
            explain.integrated_gradients(model, input, target=0,
                                         baseline=clnn.Tensor.zeros((2,)), steps=4)
        with self.assertRaises(ValueError):
            explain.smoothgrad(model, input, target=0, samples=0)


class DataTests(unittest.TestCase):
    def test_dataset_and_batches(self) -> None:
        dataset = data.TensorDataset(
            np.arange(10, dtype=np.float32).reshape(5, 2),
            np.arange(5, dtype=np.float32),
            [0, 1, 0, 1, 0],
        )
        loader = data.DataLoader(dataset, batch_size=2, shuffle=False)
        batches = list(loader)

        self.assertEqual(len(dataset), 5)
        self.assertEqual(loader.batch_count, 3)
        self.assertEqual([batch.size for batch in batches], [2, 2, 1])
        self.assertEqual(batches[0].inputs.shape, (2, 2))
        self.assertEqual(batches[0].targets.shape, (2, 1))
        self.assertEqual(batches[0].class_labels, [0, 1])

    def test_prefetched_loader_matches_normal_iteration(self) -> None:
        dataset = data.TensorDataset(
            np.arange(12, dtype=np.float32).reshape(6, 2),
            np.arange(6, dtype=np.float32),
        )
        normal = data.DataLoader(dataset, 2, shuffle=True, seed=13)
        prefetched = data.DataLoader(dataset, 2, shuffle=True, seed=13, prefetch=True)
        self.assertTrue(prefetched.prefetch_enabled)
        for expected, actual in zip(normal, prefetched):
            np.testing.assert_array_equal(actual.inputs.numpy(), expected.inputs.numpy())
            np.testing.assert_array_equal(actual.targets.numpy(), expected.targets.numpy())


@unittest.skipUnless(clnn.opencl_available(), "OpenCL GPU is not available")
class OpenCLTests(unittest.TestCase):
    def test_gpu_forward_and_backward_match_cpu(self) -> None:
        values = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)

        cpu = clnn.Tensor(values, requires_grad=True)
        cpu_loss = clnn.mean(clnn.relu(cpu * cpu + 0.25))
        cpu_loss.backward()

        gpu = clnn.Tensor(values, requires_grad=True, device=clnn.Device.opencl())
        gpu_loss = clnn.mean(clnn.relu(gpu * gpu + 0.25))
        gpu_loss.backward()
        gpu.device.synchronize()

        self.assertEqual(gpu.device.type, clnn.DeviceType.OPENCL)
        np.testing.assert_allclose(gpu_loss.item(), cpu_loss.item(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gpu.grad, cpu.grad, rtol=1e-5, atol=1e-6)

    def test_runtime_profiling_and_statistics(self) -> None:
        gpu = clnn.Device.opencl()
        gpu.set_profiling(True)
        gpu.profile()
        tensor = clnn.Tensor(np.ones((32, 32), dtype=np.float32), device=gpu)
        for _ in range(3):
            result = clnn.relu(tensor + 1.0)
            gpu.synchronize()
            result.numpy()
        profiles = gpu.profile()
        statistics = gpu.runtime_statistics()
        gpu.set_profiling(False)
        self.assertTrue(any(profile.operation == "binary_op" for profile in profiles))
        self.assertGreater(statistics.cached_kernels, 0)


if __name__ == "__main__":
    unittest.main()
