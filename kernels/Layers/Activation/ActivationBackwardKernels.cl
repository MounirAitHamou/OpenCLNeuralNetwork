__kernel void leakyReLUBackward(
    __global float* p_previousDeltas,
    __global const float* p_deltas,
    __global const float* p_preActivations,
    const float p_alpha
    )
{
    const unsigned int idx = get_global_id(0);
    const float x = p_preActivations[idx];
    const float delta = p_deltas[idx];
    p_previousDeltas[idx] = (x > 0.0f) * delta + (x <= 0.0f) * (p_alpha * delta);
}

__kernel void reLUBackward(
    __global float* p_previousDeltas,
    __global const float* p_deltas,
    __global const float* p_preActivations
    )
{
    const unsigned int idx = get_global_id(0);
    const float x = p_preActivations[idx];
    const float delta = p_deltas[idx];
    p_previousDeltas[idx] = (x > 0.0f) ? delta : 0.0f;
}

__kernel void sigmoidBackward(
    __global float* p_previousDeltas,
    __global const float* p_deltas,
    __global const float* p_outputs
    )
{
    const unsigned int idx = get_global_id(0);
    const float y = p_outputs[idx];
    p_previousDeltas[idx] = p_deltas[idx] * y * (1.0f - y);
}

__kernel void tanhBackward(
    __global float* p_previousDeltas,
    __global const float* p_deltas,
    __global const float* p_outputs
    )
{
    const unsigned int idx = get_global_id(0);
    const float y = p_outputs[idx];
    p_previousDeltas[idx] = p_deltas[idx] * (1.0f - y * y);
}

__kernel void softmaxBackward(
    __global float* p_previousDeltas,
    __global const float* p_deltas,
    __global const float* p_outputs,
    const unsigned int p_numClasses
    )
{
    const unsigned int batchIdx = get_global_id(0);
    const unsigned int offset = batchIdx * p_numClasses;

    float dot = 0.0f;
    for (unsigned int i = 0; i < p_numClasses; ++i)
        dot += p_outputs[offset + i] * p_deltas[offset + i];

    for (unsigned int i = 0; i < p_numClasses; ++i) {
        const float y = p_outputs[offset + i];
        p_previousDeltas[offset + i] = y * (p_deltas[offset + i] - dot);
    }
}
