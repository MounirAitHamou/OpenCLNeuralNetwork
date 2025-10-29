__kernel void leakyReLUForward(
    __global const float* p_inputs,
    __global float* p_outputs,
    __global float* p_preActivations,
    const float p_alpha
    )
{
    const unsigned int idx = get_global_id(0);
    const float x = p_inputs[idx];
    p_preActivations[idx] = x;
    p_outputs[idx] = (x > 0.0f) * x + (x <= 0.0f) * (p_alpha * x);
}

__kernel void reLUForward(
    __global const float* p_inputs,
    __global float* p_outputs,
    __global float* p_preActivations
    )
{
    const unsigned int idx = get_global_id(0);
    const float x = p_inputs[idx];
    p_preActivations[idx] = x;
    p_outputs[idx] = fmax(x, 0.0f);
}

__kernel void sigmoidForward(
    __global const float* p_inputs,
    __global float* p_outputs)
{
    const unsigned int idx = get_global_id(0);
    p_outputs[idx] = 1.0f / (1.0f + exp(-p_inputs[idx]));
}

__kernel void tanhForward(
    __global const float* p_inputs,
    __global float* p_outputs)
{
    const unsigned int idx = get_global_id(0);
    p_outputs[idx] = tanh(p_inputs[idx]);
}

__kernel void softmaxForward(
    __global const float* p_inputs,
    __global float* p_outputs,
    const unsigned int p_numClasses)
{
    const unsigned int batchIdx = get_global_id(0);

    const unsigned int offset = batchIdx * p_numClasses;

    float maxVal = -FLT_MAX;
    for (unsigned int i = 0; i < p_numClasses; ++i)
        maxVal = fmax(maxVal, p_inputs[offset + i]);

    float sumExp = 0.0f;
    for (unsigned int i = 0; i < p_numClasses; ++i)
        sumExp += exp(p_inputs[offset + i] - maxVal);

    for (unsigned int i = 0; i < p_numClasses; ++i)
        p_outputs[offset + i] = exp(p_inputs[offset + i] - maxVal) / sumExp;
}

