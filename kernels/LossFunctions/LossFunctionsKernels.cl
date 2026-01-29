__kernel void meanSquaredErrorComputeGradients(
    __global const float* p_predictions,
    __global const float* p_targets,
    __global float* p_gradients,
    const unsigned int p_outputElements
) {
    int idx = get_global_id(0) * p_outputElements + get_global_id(1);
    p_gradients[idx] = 2.0f * (p_predictions[idx] - p_targets[idx]) / p_outputElements;
}

__kernel void binaryCrossEntropyComputeGradients(
    __global const float* p_predictions,
    __global const float* p_targets,
    __global float* p_gradients,
    const unsigned int p_outputElements
) {
    int idx = get_global_id(0) * p_outputElements + get_global_id(1);
    float pred = p_predictions[idx];
    float target = p_targets[idx];
    pred = fmax(fmin(pred, 1.0f - 1e-7f), 1e-7f);
    p_gradients[idx] = -(target / pred) + ((1.0f - target) / (1.0f - pred));
    p_gradients[idx] /= p_outputElements;
}

__kernel void categoricalCrossEntropyComputeGradients(
    __global const float* p_logits,
    __global const float* p_targets,
    __global float* p_gradients,
    const unsigned int p_numClasses,
    const unsigned int p_batchSize
) {
    const uint batchIdx = get_global_id(0);
    const uint classIdx = get_global_id(1);

    const uint idx = batchIdx * p_numClasses + classIdx;

    const float eps = 1e-8f;
    p_gradients[idx] =
        -p_targets[idx] / fmax(p_logits[idx], eps)
        / (float)p_batchSize;
}


__kernel void softmaxCrossEntropyComputeGradients(
    __global const float* p_logits,
    __global const float* p_targets,
    __global float* p_gradients,
    const unsigned int p_numClasses
) {
    const uint batchIdx = get_global_id(0);
    const uint classIdx = get_global_id(1);

    const uint base = batchIdx * p_numClasses;

    float maxLogit = -FLT_MAX;
    for (uint c = 0; c < p_numClasses; ++c)
        maxLogit = fmax(maxLogit, p_logits[base + c]);

    float sumExp = 0.0f;
    for (uint c = 0; c < p_numClasses; ++c)
        sumExp += exp(p_logits[base + c] - maxLogit);

    float softmax =
        exp(p_logits[base + classIdx] - maxLogit) / sumExp;

    p_gradients[base + classIdx] =
        softmax - p_targets[base + classIdx];
}