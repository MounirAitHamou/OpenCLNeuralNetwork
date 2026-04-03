__kernel void convolutionalBias(
    __global const float* p_biases,
    __global float* p_outputs,
    const int p_OH,
    const int p_OW,
    const int p_OC)
{   
    const int oc = get_global_id(0);      
    const int spatialIdx = get_global_id(1); 
    const int b = get_global_id(2);       

    int outputIndex = b * (p_OC * p_OH * p_OW) 
                    + oc * (p_OH * p_OW) 
                    + spatialIdx;

    p_outputs[outputIndex] += p_biases[oc];
}

__kernel void convolutionalBackpropDeltas(
    __global const float* p_weights,
    __global const float* p_deltas,
    const int p_IH, const int p_IW,
    const int p_OH, const int p_OW,
    const int p_FH, const int p_FW,
    const int p_strideH, const int p_strideW,
    const int p_padH, const int p_padW,
    const int p_IC, const int p_OC,
    __global float* p_prevDeltas
) {
    const int gid0 = get_global_id(0);
    const int iw0 = gid0 * 2;
    const int iw1 = iw0 + 1;
    const int ih = get_global_id(1);
    const int icBatch = get_global_id(2);

    if (ih >= p_IH || iw0 >= p_IW) return;

    const int ic = icBatch % p_IC;
    const int b  = icBatch / p_IC;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int filterSize = p_FH * p_FW;
    const int icWeightOffset = ic * filterSize;
    const int weightStride = p_IC * filterSize;
    const int deltaStride = p_OH * p_OW;
    const int batchDeltaOffset = b * p_OC * deltaStride;

    const int fhStart = max(0, ih + p_padH - (p_OH - 1) * p_strideH);
    const int fhEnd = min(p_FH, ih + p_padH + 1);

    for (int fh = fhStart; fh < fhEnd; fh++) {
        int ohIdx = ih + p_padH - fh;
        if (ohIdx % p_strideH != 0) continue;
        int oh = ohIdx / p_strideH;

        for (int fw = 0; fw < p_FW; fw++) {
            int owIdx0 = iw0 + p_padW - fw;
            int owIdx1 = iw1 + p_padW - fw;

            int ow0 = (owIdx0 >= 0 && owIdx0 % p_strideW == 0) ? owIdx0 / p_strideW : -1;
            int ow1 = (owIdx1 >= 0 && owIdx1 % p_strideW == 0) ? owIdx1 / p_strideW : -1;
            
            if (ow0 >= p_OW) ow0 = -1;
            if (ow1 >= p_OW) ow1 = -1;

            if (ow0 == -1 && ow1 == -1) continue;

            int weightIdx = icWeightOffset + (fh * p_FW + fw);
            int deltaBase = batchDeltaOffset + (oh * p_OW);
            
            int oc = 0;
            for (; oc <= p_OC - 4; oc += 4) {
                float4 w4 = (float4)(
                    p_weights[(oc + 0) * weightStride + weightIdx],
                    p_weights[(oc + 1) * weightStride + weightIdx],
                    p_weights[(oc + 2) * weightStride + weightIdx],
                    p_weights[(oc + 3) * weightStride + weightIdx]
                );

                if (ow0 != -1) {
                    float4 d4_0 = (float4)(
                        p_deltas[(oc + 0) * deltaStride + deltaBase + ow0],
                        p_deltas[(oc + 1) * deltaStride + deltaBase + ow0],
                        p_deltas[(oc + 2) * deltaStride + deltaBase + ow0],
                        p_deltas[(oc + 3) * deltaStride + deltaBase + ow0]
                    );
                    acc0 += dot(w4, d4_0);
                }

                if (ow1 != -1) {
                    float4 d4_1 = (float4)(
                        p_deltas[(oc + 0) * deltaStride + deltaBase + ow1],
                        p_deltas[(oc + 1) * deltaStride + deltaBase + ow1],
                        p_deltas[(oc + 2) * deltaStride + deltaBase + ow1],
                        p_deltas[(oc + 3) * deltaStride + deltaBase + ow1]
                    );
                    acc1 += dot(w4, d4_1);
                }
            }
            
            for (; oc < p_OC; oc++) {
                float w = p_weights[oc * weightStride + weightIdx];
                if (ow0 != -1) acc0 += w * p_deltas[oc * deltaStride + deltaBase + ow0];
                if (ow1 != -1) acc1 += w * p_deltas[oc * deltaStride + deltaBase + ow1];
            }
        }
    }

    const int outputIdx = icBatch * (p_IH * p_IW) + ih * p_IW;
    p_prevDeltas[outputIdx + iw0] = acc0;
    if (iw1 < p_IW) {
        p_prevDeltas[outputIdx + iw1] = acc1;
    }
}

__kernel void convolutionalComputeWeightsGradients(
    __global const float* p_deltas,
    __global float* p_weightGradients,
    const int p_IC, const int p_IH, const int p_IW,
    const int p_OC, const int p_OH, const int p_OW,
    const int p_FH, const int p_FW,
    const int p_strideH, const int p_strideW,
    const int p_padH, const int p_padW,
    __global const float* p_inputs,
    const int p_B
) {
    const int fw = get_global_id(0);
    const int fh = get_global_id(1);
    const int icOc = get_global_id(2);
    
    const int ic = icOc % p_IC;
    const int oc = icOc / p_IC;

    if (fw >= p_FW || fh >= p_FH || oc >= p_OC) return;

    float gradientSum = 0.0f;

    for (int b = 0; b < p_B; b++) {
        for (int oh = 0; oh < p_OH; oh++) {
            for (int ow = 0; ow < p_OW; ow++) {
                int ih = (oh * p_strideH) - p_padH + fh;
                int iw = (ow * p_strideW) - p_padW + fw;

                if (ih >= 0 && ih < p_IH && iw >= 0 && iw < p_IW) {
                    int inputIdx = b * (p_IC * p_IH * p_IW) + ic * (p_IH * p_IW) + ih * p_IW + iw;
                    int deltaIdx = b * (p_OC * p_OH * p_OW) + oc * (p_OH * p_OW) + oh * p_OW + ow;
                    
                    gradientSum += p_inputs[inputIdx] * p_deltas[deltaIdx];
                }
            }
        }
    }

    const int weightIdx = oc * (p_IC * p_FH * p_FW) + ic * (p_FH * p_FW) + fh * p_FW + fw;
    p_weightGradients[weightIdx] = gradientSum / (float)p_B;
}

__kernel void convolutionalComputeBiasesGradients(
    __global const float* p_deltas,
    __global float* p_biasGradients,
    const int p_OC,
    const int p_OH,
    const int p_OW,
    const int p_B
) {
    const int oc = get_global_id(0);
    if (oc >= p_OC) return;

    float sum = 0.0f;
    const int spatialSize = p_OH * p_OW;
    const int batchStride = p_OC * spatialSize;

    for (int b = 0; b < p_B; b++) {
        int batchOffset = b * batchStride + oc * spatialSize;
        for (int i = 0; i < spatialSize; i++) {
            sum += p_deltas[batchOffset + i];
        }
    }

    p_biasGradients[oc] = sum / (float)p_B;
}