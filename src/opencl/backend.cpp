#include "backend.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace clnn::opencl {
namespace {

constexpr const char* kernel_source = R"CLC(
__kernel void copy_op(__global const float* input, __global float* output, ulong count) {
    ulong i = get_global_id(0); if (i < count) output[i] = input[i];
}
__kernel void accumulate_op(__global float* destination, __global const float* contribution,
                            ulong count) {
    ulong i = get_global_id(0); if (i < count) destination[i] += contribution[i];
}
__kernel void binary_op(uint op, __global const float* a, __global const float* b,
                        __global const ulong* ai, __global const ulong* bi,
                        __global float* out, ulong count) {
    ulong i = get_global_id(0); if (i >= count) return;
    float x = a[ai[i]], y = b[bi[i]];
    if (op == 0) out[i] = x + y;
    else if (op == 1) out[i] = x - y;
    else if (op == 2) out[i] = x * y;
    else out[i] = x / y;
}
__kernel void unary_op(uint op, __global const float* in, __global float* out,
                       float argument, ulong count) {
    ulong i = get_global_id(0); if (i >= count) return; float x = in[i];
    if (op == 0) out[i] = -x;
    else if (op == 1) out[i] = pow(x, argument);
    else if (op == 2) out[i] = exp(x);
    else if (op == 3) out[i] = log(x);
    else if (op == 4) out[i] = fmax(0.0f, x);
    else if (op == 5) out[i] = 1.0f / (1.0f + exp(-x));
    else out[i] = tanh(x);
}
__kernel void matmul_op(__global const float* a, __global const float* b,
                        __global float* out, ulong rows, ulong inner, ulong columns) {
    const ulong row = get_global_id(0), col = get_global_id(1);
    const ulong local_row = get_local_id(0), local_col = get_local_id(1);
    __local float a_tile[16][16];
    __local float b_tile[16][16];
    float value = 0.0f;
    const ulong tile_count = (inner + 15) / 16;
    for (ulong tile = 0; tile < tile_count; ++tile) {
        const ulong a_column = tile * 16 + local_col;
        const ulong b_row = tile * 16 + local_row;
        a_tile[local_row][local_col] = row < rows && a_column < inner
                                           ? a[row * inner + a_column] : 0.0f;
        b_tile[local_row][local_col] = b_row < inner && col < columns
                                           ? b[b_row * columns + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < 16; ++k) value += a_tile[local_row][k] * b_tile[k][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows && col < columns) out[row * columns + col] = value;
}
__kernel void sum_op(__global const float* input, __global float* output, ulong count,
                     __local float* scratch) {
    ulong local_i = get_local_id(0), group = get_group_id(0);
    ulong first = group * get_local_size(0) * 2 + local_i;
    float value = first < count ? input[first] : 0.0f;
    ulong second = first + get_local_size(0); if (second < count) value += input[second];
    scratch[local_i] = value; barrier(CLK_LOCAL_MEM_FENCE);
    for (ulong offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
        if (local_i < offset) scratch[local_i] += scratch[local_i + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_i == 0) output[group] = scratch[0];
}
__kernel void axis_sum_op(__global const float* input, __global float* output,
                          ulong outer, ulong axis, ulong inner, __local float* scratch) {
    ulong group = get_group_id(0), local_i = get_local_id(0);
    if (group >= outer * inner) return;
    ulong outer_i = group / inner, inner_i = group % inner; float value = 0.0f;
    for (ulong a = local_i; a < axis; a += get_local_size(0))
        value += input[(outer_i * axis + a) * inner + inner_i];
    scratch[local_i] = value; barrier(CLK_LOCAL_MEM_FENCE);
    for (ulong offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
        if (local_i < offset) scratch[local_i] += scratch[local_i + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_i == 0) output[group] = scratch[0];
}
__kernel void axis_sum_grad(__global const float* upstream, __global float* gradient,
                            ulong outer, ulong axis, ulong inner) {
    ulong i = get_global_id(0); if (i >= outer * axis * inner) return;
    ulong outer_i = i / (axis * inner), inner_i = i % inner;
    gradient[i] = upstream[outer_i * inner + inner_i];
}
__kernel void sum_grad(__global const float* upstream, __global float* gradient, ulong count) {
    ulong i = get_global_id(0); if (i < count) gradient[i] = upstream[0];
}
__kernel void transpose_op(__global const float* in, __global float* out,
                           ulong rows, ulong columns) {
    ulong row = get_global_id(0), col = get_global_id(1);
    if (row < rows && col < columns) out[col * rows + row] = in[row * columns + col];
}
__kernel void softmax_op(__global const float* in, __global float* out,
                         ulong outer, ulong axis, ulong inner) {
    ulong group = get_global_id(0); if (group >= outer * inner) return;
    ulong outer_i = group / inner, inner_i = group % inner;
    float maximum = -INFINITY;
    for (ulong a = 0; a < axis; ++a) maximum = fmax(maximum, in[(outer_i * axis + a) * inner + inner_i]);
    float denominator = 0.0f;
    for (ulong a = 0; a < axis; ++a) {
        ulong i = (outer_i * axis + a) * inner + inner_i;
        out[i] = exp(in[i] - maximum); denominator += out[i];
    }
    for (ulong a = 0; a < axis; ++a) out[(outer_i * axis + a) * inner + inner_i] /= denominator;
}
__kernel void conv2d_op(__global const float* input, __global const float* weight,
                        __global const float* bias, uint has_bias, __global float* output,
                        ulong batch, ulong in_channels, ulong in_h, ulong in_w,
                        ulong out_channels, ulong kernel_h, ulong kernel_w,
                        ulong out_h, ulong out_w, ulong stride_y, ulong stride_x,
                        ulong padding_y, ulong padding_x) {
    ulong linear = get_global_id(0); ulong total = batch * out_channels * out_h * out_w;
    if (linear >= total) return;
    ulong out_x = linear % out_w; linear /= out_w;
    ulong out_y = linear % out_h; linear /= out_h;
    ulong oc = linear % out_channels; ulong n = linear / out_channels;
    float value = has_bias ? bias[oc] : 0.0f;
    for (ulong ic = 0; ic < in_channels; ++ic) for (ulong ky = 0; ky < kernel_h; ++ky) {
        long iy = (long)(out_y * stride_y + ky) - (long)padding_y;
        if (iy < 0 || iy >= (long)in_h) continue;
        for (ulong kx = 0; kx < kernel_w; ++kx) {
            long ix = (long)(out_x * stride_x + kx) - (long)padding_x;
            if (ix < 0 || ix >= (long)in_w) continue;
            ulong ii = ((n * in_channels + ic) * in_h + (ulong)iy) * in_w + (ulong)ix;
            ulong wi = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;
            value += input[ii] * weight[wi];
        }
    }
    output[((n * out_channels + oc) * out_h + out_y) * out_w + out_x] = value;
}
__kernel void conv2d_tiled(__global const float* input, __global const float* weight,
                           __global const float* bias, uint has_bias,
                           __global float* output, ulong batch, ulong in_channels,
                           ulong in_h, ulong in_w, ulong out_channels, ulong kernel_h,
                           ulong kernel_w, ulong out_h, ulong out_w, ulong stride_y,
                           ulong stride_x, ulong padding_y, ulong padding_x,
                           __local float* input_tile, __local float* weight_tile) {
    ulong n_oc = get_group_id(0); if (n_oc >= batch * out_channels) return;
    ulong n = n_oc / out_channels, oc = n_oc % out_channels;
    ulong local_y = get_local_id(1), local_x = get_local_id(2);
    ulong local_h = get_local_size(1), local_w = get_local_size(2);
    ulong lid = local_y * local_w + local_x, local_count = local_h * local_w;
    ulong tile_h = (local_h - 1) * stride_y + kernel_h;
    ulong tile_w = (local_w - 1) * stride_x + kernel_w;
    ulong input_count = in_channels * tile_h * tile_w;
    long origin_y = (long)(get_group_id(1) * local_h * stride_y) - (long)padding_y;
    long origin_x = (long)(get_group_id(2) * local_w * stride_x) - (long)padding_x;
    for (ulong index = lid; index < input_count; index += local_count) {
        ulong tx = index % tile_w, rest = index / tile_w;
        ulong ty = rest % tile_h, ic = rest / tile_h;
        long iy = origin_y + (long)ty, ix = origin_x + (long)tx;
        input_tile[index] = iy >= 0 && iy < (long)in_h && ix >= 0 && ix < (long)in_w
            ? input[((n * in_channels + ic) * in_h + (ulong)iy) * in_w + (ulong)ix]
            : 0.0f;
    }
    ulong weight_count = in_channels * kernel_h * kernel_w;
    for (ulong index = lid; index < weight_count; index += local_count)
        weight_tile[index] = weight[oc * weight_count + index];
    barrier(CLK_LOCAL_MEM_FENCE);
    ulong oy = get_global_id(1), ox = get_global_id(2);
    if (oy >= out_h || ox >= out_w) return;
    float value = has_bias ? bias[oc] : 0.0f;
    ulong tile_y = local_y * stride_y, tile_x = local_x * stride_x;
    for (ulong ic = 0; ic < in_channels; ++ic)
        for (ulong ky = 0; ky < kernel_h; ++ky)
            for (ulong kx = 0; kx < kernel_w; ++kx) {
                ulong ii = (ic * tile_h + tile_y + ky) * tile_w + tile_x + kx;
                ulong wi = (ic * kernel_h + ky) * kernel_w + kx;
                value += input_tile[ii] * weight_tile[wi];
            }
    output[((n * out_channels + oc) * out_h + oy) * out_w + ox] = value;
}
)CLC"
                                      R"CLC(
__kernel void pool2d_op(uint operation, __global const float* input, __global float* output,
                        ulong batch, ulong channels, ulong in_h, ulong in_w,
                        ulong kernel_h, ulong kernel_w, ulong out_h, ulong out_w,
                        ulong stride_y, ulong stride_x, ulong padding_y, ulong padding_x) {
    ulong linear = get_global_id(0), total = batch * channels * out_h * out_w;
    if (linear >= total) return;
    ulong out_x = linear % out_w; linear /= out_w;
    ulong out_y = linear % out_h; linear /= out_h;
    ulong channel = linear % channels; ulong n = linear / channels;
    float value = operation == 0 ? -INFINITY : 0.0f; ulong count = 0;
    for (ulong ky = 0; ky < kernel_h; ++ky) {
        long iy = (long)(out_y * stride_y + ky) - (long)padding_y;
        if (iy < 0 || iy >= (long)in_h) continue;
        for (ulong kx = 0; kx < kernel_w; ++kx) {
            long ix = (long)(out_x * stride_x + kx) - (long)padding_x;
            if (ix < 0 || ix >= (long)in_w) continue;
            float candidate = input[((n * channels + channel) * in_h + (ulong)iy) * in_w + (ulong)ix];
            value = operation == 0 ? fmax(value, candidate) : value + candidate; ++count;
        }
    }
    output[((n * channels + channel) * out_h + out_y) * out_w + out_x] =
        operation == 1 ? value / (float)count : value;
}
__kernel void pool2d_grad(uint operation, __global const float* input,
                          __global const float* upstream, __global float* gradient,
                          ulong batch, ulong channels, ulong in_h, ulong in_w,
                          ulong kernel_h, ulong kernel_w, ulong out_h, ulong out_w,
                          ulong stride_y, ulong stride_x, ulong padding_y, ulong padding_x) {
    ulong linear = get_global_id(0), total = batch * channels * in_h * in_w;
    if (linear >= total) return;
    ulong ix = linear % in_w; ulong rest = linear / in_w;
    ulong iy = rest % in_h; rest /= in_h;
    ulong channel = rest % channels; ulong n = rest / channels; float value = 0.0f;
    for (ulong oy = 0; oy < out_h; ++oy) for (ulong ox = 0; ox < out_w; ++ox) {
        long start_y = (long)(oy * stride_y) - (long)padding_y;
        long start_x = (long)(ox * stride_x) - (long)padding_x;
        if ((long)iy < start_y || (long)iy >= start_y + (long)kernel_h ||
            (long)ix < start_x || (long)ix >= start_x + (long)kernel_w) continue;
        ulong upstream_i = ((n * channels + channel) * out_h + oy) * out_w + ox;
        if (operation == 1) {
            ulong count = 0;
            for (ulong ky = 0; ky < kernel_h; ++ky) for (ulong kx = 0; kx < kernel_w; ++kx) {
                long py = start_y + (long)ky, px = start_x + (long)kx;
                if (py >= 0 && py < (long)in_h && px >= 0 && px < (long)in_w) ++count;
            }
            value += upstream[upstream_i] / (float)count;
        } else {
            float maximum = -INFINITY; ulong winner = 0;
            for (ulong ky = 0; ky < kernel_h; ++ky) for (ulong kx = 0; kx < kernel_w; ++kx) {
                long py = start_y + (long)ky, px = start_x + (long)kx;
                if (py < 0 || py >= (long)in_h || px < 0 || px >= (long)in_w) continue;
                ulong candidate_i = ((n * channels + channel) * in_h + (ulong)py) * in_w + (ulong)px;
                float candidate = input[candidate_i];
                if (candidate > maximum) { maximum = candidate; winner = candidate_i; }
            }
            if (winner == ((n * channels + channel) * in_h + iy) * in_w + ix)
                value += upstream[upstream_i];
        }
    }
    gradient[((n * channels + channel) * in_h + iy) * in_w + ix] = value;
}
__kernel void binary_grad(uint op, uint lhs_grad, __global const float* a,
                          __global const float* b, __global const ulong* ai,
                          __global const ulong* bi, __global const float* upstream,
                          __global float* gradient, ulong output_count, ulong parent_count) {
    ulong p = get_global_id(0); if (p >= parent_count) return; float value = 0.0f;
    for (ulong i = 0; i < output_count; ++i) {
        ulong mapped = lhs_grad ? ai[i] : bi[i]; if (mapped != p) continue;
        float x = a[ai[i]], y = b[bi[i]], derivative;
        if (op == 0) derivative = 1.0f;
        else if (op == 1) derivative = lhs_grad ? 1.0f : -1.0f;
        else if (op == 2) derivative = lhs_grad ? y : x;
        else derivative = lhs_grad ? 1.0f / y : -x / (y * y);
        value += upstream[i] * derivative;
    }
    gradient[p] = value;
}
__kernel void unary_grad(uint op, __global const float* input,
                         __global const float* upstream, __global float* gradient,
                         float argument, ulong count) {
    ulong i = get_global_id(0); if (i >= count) return; float x = input[i], derivative;
    if (op == 0) derivative = -1.0f;
    else if (op == 1) derivative = argument * pow(x, argument - 1.0f);
    else if (op == 2) derivative = exp(x);
    else if (op == 3) derivative = 1.0f / x;
    else if (op == 4) derivative = x > 0.0f ? 1.0f : 0.0f;
    else if (op == 5) { float y = 1.0f / (1.0f + exp(-x)); derivative = y * (1.0f - y); }
    else { float y = tanh(x); derivative = 1.0f - y * y; }
    gradient[i] = upstream[i] * derivative;
}
__kernel void softmax_grad(__global const float* output, __global const float* upstream,
                           __global float* gradient, ulong outer, ulong axis, ulong inner) {
    ulong group = get_global_id(0); if (group >= outer * inner) return;
    ulong outer_i = group / inner, inner_i = group % inner; float dot = 0.0f;
    for (ulong a = 0; a < axis; ++a) { ulong i = (outer_i * axis + a) * inner + inner_i; dot += upstream[i] * output[i]; }
    for (ulong a = 0; a < axis; ++a) { ulong i = (outer_i * axis + a) * inner + inner_i; gradient[i] = output[i] * (upstream[i] - dot); }
}
__kernel void conv_input_grad(__global const float* weight, __global const float* upstream,
                              __global float* gradient, ulong batch, ulong in_channels,
                              ulong in_h, ulong in_w, ulong out_channels, ulong kernel_h,
                              ulong kernel_w, ulong out_h, ulong out_w, ulong stride_y,
                              ulong stride_x, ulong padding_y, ulong padding_x) {
    ulong linear=get_global_id(0),total=batch*in_channels*in_h*in_w;if(linear>=total)return;
    ulong ix=linear%in_w;linear/=in_w;ulong iy=linear%in_h;linear/=in_h;
    ulong ic=linear%in_channels;ulong n=linear/in_channels;float value=0.0f;
    for(ulong oc=0;oc<out_channels;++oc)for(ulong ky=0;ky<kernel_h;++ky){
        long oy_num=(long)iy+(long)padding_y-(long)ky;if(oy_num<0||oy_num%(long)stride_y!=0)continue;
        ulong oy=(ulong)(oy_num/(long)stride_y);if(oy>=out_h)continue;
        for(ulong kx=0;kx<kernel_w;++kx){long ox_num=(long)ix+(long)padding_x-(long)kx;
            if(ox_num<0||ox_num%(long)stride_x!=0)continue;ulong ox=(ulong)(ox_num/(long)stride_x);if(ox>=out_w)continue;
            ulong ui=((n*out_channels+oc)*out_h+oy)*out_w+ox;
            ulong wi=((oc*in_channels+ic)*kernel_h+ky)*kernel_w+kx;value+=upstream[ui]*weight[wi];}}
    gradient[((n*in_channels+ic)*in_h+iy)*in_w+ix]=value;
}
__kernel void conv_weight_grad(__global const float* input, __global const float* upstream,
                               __global float* gradient, ulong batch, ulong in_channels,
                               ulong in_h, ulong in_w, ulong out_channels, ulong kernel_h,
                               ulong kernel_w, ulong out_h, ulong out_w, ulong stride_y,
                               ulong stride_x, ulong padding_y, ulong padding_x) {
    ulong linear=get_global_id(0),total=out_channels*in_channels*kernel_h*kernel_w;if(linear>=total)return;
    ulong kx=linear%kernel_w;linear/=kernel_w;ulong ky=linear%kernel_h;linear/=kernel_h;
    ulong ic=linear%in_channels;ulong oc=linear/in_channels;float value=0.0f;
    for(ulong n=0;n<batch;++n)for(ulong oy=0;oy<out_h;++oy){long iy=(long)(oy*stride_y+ky)-(long)padding_y;if(iy<0||iy>=(long)in_h)continue;
        for(ulong ox=0;ox<out_w;++ox){long ix=(long)(ox*stride_x+kx)-(long)padding_x;if(ix<0||ix>=(long)in_w)continue;
            ulong ii=((n*in_channels+ic)*in_h+(ulong)iy)*in_w+(ulong)ix;ulong ui=((n*out_channels+oc)*out_h+oy)*out_w+ox;
            value+=input[ii]*upstream[ui];}}
    gradient[((oc*in_channels+ic)*kernel_h+ky)*kernel_w+kx]=value;
}
__kernel void conv_bias_grad(__global const float* upstream,__global float* gradient,
                             ulong batch,ulong out_channels,ulong out_h,ulong out_w){
    ulong oc=get_global_id(0);if(oc>=out_channels)return;float value=0.0f;
    for(ulong n=0;n<batch;++n)for(ulong y=0;y<out_h;++y)for(ulong x=0;x<out_w;++x)value+=upstream[((n*out_channels+oc)*out_h+y)*out_w+x];
    gradient[oc]=value;
}
__kernel void binary_loss(uint with_logits,__global const float* prediction,
                          __global const float* target,__global float* output,
                          float epsilon,ulong count){
    if(get_global_id(0)!=0)return;float value=0.0f;
    for(ulong i=0;i<count;++i){float x=prediction[i],y=target[i];
        if(with_logits)value+=fmax(x,0.0f)-x*y+log(1.0f+exp(-fabs(x)));
        else{float p=clamp(x,epsilon,1.0f-epsilon);value-=y*log(p)+(1.0f-y)*log(1.0f-p);}}
    output[0]=value/(float)count;
}
__kernel void binary_loss_grad(uint with_logits,uint target_grad,__global const float* prediction,
                               __global const float* target,__global float* gradient,
                               __global const float* upstream,float epsilon,ulong count){
    ulong i=get_global_id(0);if(i>=count)return;float x=prediction[i],y=target[i],value;
    if(with_logits){if(target_grad)value=-x;else value=1.0f/(1.0f+exp(-x))-y;}
    else{float p=clamp(x,epsilon,1.0f-epsilon);if(target_grad)value=log(1.0f-p)-log(p);
        else value=(p-y)/(p*(1.0f-p));}
    gradient[i]=upstream[0]*value/(float)count;
}
__kernel void cross_entropy_loss_op(__global const float* logits,__global const ulong* labels,
                                    __global float* output,ulong batch,ulong classes){
    if(get_global_id(0)!=0)return;float loss=0.0f;
    for(ulong row=0;row<batch;++row){float maximum=-INFINITY;for(ulong c=0;c<classes;++c)maximum=fmax(maximum,logits[row*classes+c]);
        float denominator=0.0f;for(ulong c=0;c<classes;++c)denominator+=exp(logits[row*classes+c]-maximum);
        loss+=log(denominator)-(logits[row*classes+labels[row]]-maximum);}output[0]=loss/(float)batch;
}
__kernel void cross_entropy_grad_op(__global const float* logits,__global const ulong* labels,
                                    __global float* gradient,__global const float* upstream,
                                    ulong batch,ulong classes){
    ulong row=get_global_id(0);if(row>=batch)return;float maximum=-INFINITY;
    for(ulong c=0;c<classes;++c)maximum=fmax(maximum,logits[row*classes+c]);float denominator=0.0f;
    for(ulong c=0;c<classes;++c)denominator+=exp(logits[row*classes+c]-maximum);
    for(ulong c=0;c<classes;++c){float value=exp(logits[row*classes+c]-maximum)/denominator;
        if(c==labels[row])value-=1.0f;gradient[row*classes+c]=upstream[0]*value/(float)batch;}
}
__kernel void sgd_update_op(__global float* parameter,__global const float* gradient,
                            __global float* velocity,float learning_rate,float momentum,
                            float weight_decay,uint use_momentum,ulong count){
    ulong i=get_global_id(0);if(i>=count)return;float direction=gradient[i]+weight_decay*parameter[i];
    if(use_momentum){velocity[i]=momentum*velocity[i]+direction;direction=velocity[i];}
    parameter[i]-=learning_rate*direction;
}
__kernel void adam_update_op(__global float* parameter,__global const float* gradient,
                             __global float* first,__global float* second,float learning_rate,
                             float beta1,float beta2,float epsilon,float weight_decay,
                             float first_correction,float second_correction,uint decoupled,ulong count){
    ulong i=get_global_id(0);if(i>=count)return;float direction=gradient[i];
    if(decoupled)parameter[i]*=1.0f-learning_rate*weight_decay;else direction+=weight_decay*parameter[i];
    first[i]=beta1*first[i]+(1.0f-beta1)*direction;second[i]=beta2*second[i]+(1.0f-beta2)*direction*direction;
    parameter[i]-=learning_rate*(first[i]/first_correction)/(sqrt(second[i]/second_correction)+epsilon);
}
)CLC";

void check(const api::Int status, const char* operation) {
    if (status != api::success) {
        throw std::runtime_error(std::string(operation) + " failed with OpenCL error " +
                                 std::to_string(status));
    }
}

api::Functions& functions() {
    static auto* instance = new api::Functions();
    return *instance;
}

std::string info_string(api::DeviceHandle device, api::DeviceInfo key) {
    std::size_t size = 0;
    check(functions().get_device_info(device, key, 0, nullptr, &size), "clGetDeviceInfo");
    std::string value(size, '\0');
    check(functions().get_device_info(device, key, value.size(), value.data(), nullptr),
          "clGetDeviceInfo");
    while (!value.empty() && value.back() == '\0')
        value.pop_back();
    return value;
}

class Event final {
  public:
    explicit Event(const api::EventHandle handle) : handle_(handle) {}
    ~Event() {
        if (handle_ != nullptr)
            (void)functions().release_event(handle_);
    }
    [[nodiscard]] api::EventHandle handle() const noexcept {
        return handle_;
    }
    [[nodiscard]] bool complete() const {
        api::Int status = api::success;
        check(functions().get_event_info(handle_, api::event_command_execution_status,
                                         sizeof(status), &status, nullptr),
              "clGetEventInfo");
        return status <= api::complete;
    }

  private:
    api::EventHandle handle_ = nullptr;
};

class Runtime final {
  public:
    explicit Runtime(const Device requested) : requested_(requested) {
        api::UInt platform_count = 0;
        check(functions().get_platform_ids(0, nullptr, &platform_count), "clGetPlatformIDs");
        std::vector<api::PlatformHandle> platforms(platform_count);
        check(functions().get_platform_ids(platform_count, platforms.data(), nullptr),
              "clGetPlatformIDs");
        if (requested.platform_index() >= platforms.size())
            throw std::out_of_range("OpenCL platform index is out of range");
        api::UInt device_count = 0;
        check(functions().get_device_ids(platforms[requested.platform_index()],
                                         api::device_type_gpu, 0, nullptr, &device_count),
              "clGetDeviceIDs(GPU)");
        std::vector<api::DeviceHandle> devices(device_count);
        check(functions().get_device_ids(platforms[requested.platform_index()],
                                         api::device_type_gpu, device_count, devices.data(),
                                         nullptr),
              "clGetDeviceIDs(GPU)");
        if (requested.device_index() >= devices.size())
            throw std::out_of_range("OpenCL GPU index is out of range");
        device_ = devices[requested.device_index()];
        api::Int status = api::success;
        context_ = functions().create_context(nullptr, 1, &device_, nullptr, nullptr, &status);
        check(status, "clCreateContext");
        queue_ = functions().create_command_queue(context_, device_, api::queue_profiling_enable,
                                                  &status);
        check(status, "clCreateCommandQueue");
        check(functions().get_device_info(device_, api::device_local_memory_size,
                                          sizeof(local_memory_size_), &local_memory_size_, nullptr),
              "clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE)");
        const std::size_t source_length = std::char_traits<char>::length(kernel_source);
        const char* source = kernel_source;
        program_ =
            functions().create_program_with_source(context_, 1, &source, &source_length, &status);
        check(status, "clCreateProgramWithSource");
        status =
            functions().build_program(program_, 1, &device_, "-cl-std=CL1.2", nullptr, nullptr);
        if (status != api::success) {
            std::size_t log_size = 0;
            functions().get_program_build_info(program_, device_, api::program_build_log, 0,
                                               nullptr, &log_size);
            std::string log(log_size, '\0');
            functions().get_program_build_info(program_, device_, api::program_build_log,
                                               log.size(), log.data(), nullptr);
            throw std::runtime_error("OpenCL autograd kernel build failed:\n" + log);
        }
    }

    ~Runtime() {
        if (queue_ != nullptr)
            (void)functions().finish(queue_);
        profiles_.clear();
        last_event_.reset();
        for (const auto& entry : memory_pool_)
            (void)functions().release_memory_object(entry.handle);
        for (const auto& [name, kernel] : kernels_) {
            static_cast<void>(name);
            (void)functions().release_kernel(kernel);
        }
        if (program_ != nullptr)
            functions().release_program(program_);
        if (queue_ != nullptr)
            functions().release_command_queue(queue_);
        if (context_ != nullptr)
            functions().release_context(context_);
    }

    [[nodiscard]] api::ContextHandle context() const noexcept {
        return context_;
    }
    [[nodiscard]] api::QueueHandle queue() const noexcept {
        return queue_;
    }
    [[nodiscard]] std::string name() const {
        return info_string(device_, api::device_name);
    }
    [[nodiscard]] std::mutex& mutex() noexcept {
        return mutex_;
    }

    [[nodiscard]] std::size_t local_memory_size() const noexcept {
        return static_cast<std::size_t>(local_memory_size_);
    }

    api::KernelHandle kernel(const char* name) {
        if (const auto found = kernels_.find(name); found != kernels_.end())
            return found->second;
        api::Int status = api::success;
        auto result = functions().create_kernel(program_, name, &status);
        check(status, "clCreateKernel");
        kernels_.emplace(name, result);
        return result;
    }

    void record(const char* name, const api::EventHandle handle) {
        auto event = std::make_shared<Event>(handle);
        {
            std::scoped_lock pool_lock(pool_mutex_);
            last_event_ = event;
        }
        if (profiling_)
            profiles_.push_back({name, std::move(event)});
    }

    [[nodiscard]] api::MemoryHandle acquire_memory(const std::size_t elements,
                                                   const float* initial_data) {
        const auto bytes = elements * sizeof(float);
        if (initial_data == nullptr) {
            std::scoped_lock pool_lock(pool_mutex_);
            for (auto iterator = memory_pool_.begin(); iterator != memory_pool_.end(); ++iterator) {
                if (iterator->bytes == bytes &&
                    (iterator->completion == nullptr || iterator->completion->complete())) {
                    const auto handle = iterator->handle;
                    memory_pool_.erase(iterator);
                    ++buffer_reuses_;
                    return handle;
                }
            }
        }
        api::Int status = api::success;
        auto flags = api::memory_read_write;
        void* data = nullptr;
        if (initial_data != nullptr) {
            flags |= api::memory_copy_host_pointer;
            data = const_cast<float*>(initial_data);
        }
        const auto handle = functions().create_buffer(context_, flags, bytes, data, &status);
        check(status, "clCreateBuffer");
        {
            std::scoped_lock pool_lock(pool_mutex_);
            ++buffer_allocations_;
        }
        return handle;
    }

    void recycle_memory(const api::MemoryHandle handle, const std::size_t elements) noexcept {
        if (handle == nullptr)
            return;
        try {
            std::scoped_lock pool_lock(pool_mutex_);
            if (memory_pool_.size() >= 128) {
                (void)functions().release_memory_object(handle);
                return;
            }
            memory_pool_.push_back({handle, elements * sizeof(float), last_event_});
        } catch (...) {
            (void)functions().release_memory_object(handle);
        }
    }

    void clear_pool() {
        std::scoped_lock pool_lock(pool_mutex_);
        for (const auto& entry : memory_pool_)
            check(functions().release_memory_object(entry.handle), "clReleaseMemObject");
        memory_pool_.clear();
    }

    void set_profiling(const bool enabled) {
        profiling_ = enabled;
        if (!enabled)
            profiles_.clear();
    }

    [[nodiscard]] std::vector<KernelProfile> profile(const bool reset) {
        std::map<std::string, KernelProfile> aggregated;
        for (const auto& record : profiles_) {
            api::ULong start = 0, end = 0;
            check(functions().get_event_profiling_info(record.event->handle(),
                                                       api::profiling_command_start, sizeof(start),
                                                       &start, nullptr),
                  "clGetEventProfilingInfo(start)");
            check(functions().get_event_profiling_info(record.event->handle(),
                                                       api::profiling_command_end, sizeof(end),
                                                       &end, nullptr),
                  "clGetEventProfilingInfo(end)");
            const auto milliseconds = static_cast<double>(end - start) / 1.0e6;
            auto& result = aggregated[record.name];
            result.operation = record.name;
            if (result.calls == 0) {
                result.minimum_milliseconds = milliseconds;
                result.maximum_milliseconds = milliseconds;
            } else {
                result.minimum_milliseconds = std::min(result.minimum_milliseconds, milliseconds);
                result.maximum_milliseconds = std::max(result.maximum_milliseconds, milliseconds);
            }
            ++result.calls;
            result.total_milliseconds += milliseconds;
        }
        std::vector<KernelProfile> result;
        result.reserve(aggregated.size());
        for (auto& [name, entry] : aggregated) {
            static_cast<void>(name);
            result.push_back(std::move(entry));
        }
        if (reset)
            profiles_.clear();
        return result;
    }

    [[nodiscard]] OpenCLRuntimeStatistics statistics() {
        std::scoped_lock pool_lock(pool_mutex_);
        return {buffer_allocations_, buffer_reuses_, memory_pool_.size(), kernels_.size()};
    }

  private:
    struct PoolEntry final {
        api::MemoryHandle handle;
        std::size_t bytes;
        std::shared_ptr<Event> completion;
    };
    struct ProfileRecord final {
        std::string name;
        std::shared_ptr<Event> event;
    };
    Device requested_;
    api::DeviceHandle device_ = nullptr;
    api::ContextHandle context_ = nullptr;
    api::QueueHandle queue_ = nullptr;
    api::ProgramHandle program_ = nullptr;
    api::ULong local_memory_size_ = 0;
    std::mutex mutex_;
    std::mutex pool_mutex_;
    std::unordered_map<std::string, api::KernelHandle> kernels_;
    std::vector<PoolEntry> memory_pool_;
    std::shared_ptr<Event> last_event_;
    std::vector<ProfileRecord> profiles_;
    bool profiling_ = false;
    std::size_t buffer_allocations_ = 0;
    std::size_t buffer_reuses_ = 0;
};

Runtime& runtime(const Device device) {
    if (device.type() != DeviceType::opencl)
        throw std::invalid_argument("an OpenCL device is required");
    static std::mutex registry_mutex;
    static std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<Runtime>> runtimes;
    std::scoped_lock lock(registry_mutex);
    const auto key = std::make_pair(device.platform_index(), device.device_index());
    auto& instance = runtimes[key];
    if (!instance)
        instance = std::make_unique<Runtime>(device);
    return *instance;
}

class Kernel final {
  public:
    Kernel(Runtime& runtime, const char* name)
        : runtime_(runtime), name_(name), handle_(runtime.kernel(name)) {}
    template <typename Value> void argument(api::UInt index, const Value& value) {
        check(functions().set_kernel_argument(handle_, index, sizeof(Value), &value),
              "clSetKernelArg");
    }
    void local_argument(const api::UInt index, const std::size_t bytes) {
        check(functions().set_kernel_argument(handle_, index, bytes, nullptr), "clSetKernelArg");
    }
    void run(const std::vector<std::size_t>& global, const std::vector<std::size_t>& local = {}) {
        const auto* local_sizes = local.empty() ? nullptr : local.data();
        api::EventHandle event = nullptr;
        check(functions().enqueue_nd_range_kernel(runtime_.queue(), handle_,
                                                  static_cast<api::UInt>(global.size()), nullptr,
                                                  global.data(), local_sizes, 0, nullptr, &event),
              "clEnqueueNDRangeKernel");
        runtime_.record(name_, event);
        check(functions().flush(runtime_.queue()), "clFlush");
    }

  private:
    Runtime& runtime_;
    const char* name_;
    api::KernelHandle handle_;
};

class RawMemory final {
  public:
    RawMemory(Runtime& runtime, std::size_t bytes, const void* data) : runtime_(runtime) {
        api::Int status = api::success;
        auto flags = api::memory_read_write;
        void* mutable_data = nullptr;
        if (data != nullptr) {
            flags |= api::memory_copy_host_pointer;
            mutable_data = const_cast<void*>(data);
        }
        handle_ = functions().create_buffer(runtime.context(), flags, bytes, mutable_data, &status);
        check(status, "clCreateBuffer");
    }
    ~RawMemory() {
        if (handle_ != nullptr)
            functions().release_memory_object(handle_);
    }
    [[nodiscard]] api::MemoryHandle handle() const noexcept {
        return handle_;
    }

  private:
    Runtime& runtime_;
    api::MemoryHandle handle_ = nullptr;
};

api::ULong to_ulong(std::size_t value) {
    return static_cast<api::ULong>(value);
}

} // namespace

Buffer::Buffer(const Device device, const std::size_t elements, const float* initial_data)
    : device_(device), elements_(elements) {
    auto& selected = runtime(device);
    handle_ = selected.acquire_memory(elements, initial_data);
}

Buffer::~Buffer() {
    if (handle_ != nullptr)
        runtime(device_).recycle_memory(handle_, elements_);
}
Device Buffer::device() const noexcept {
    return device_;
}
std::size_t Buffer::size() const noexcept {
    return elements_;
}
api::MemoryHandle Buffer::handle() const noexcept {
    return handle_;
}

std::vector<float> Buffer::read() const {
    std::vector<float> values(elements_);
    auto& selected = runtime(device_);
    std::scoped_lock lock(selected.mutex());
    check(functions().enqueue_read_buffer(selected.queue(), handle_, api::true_value, 0,
                                          values.size() * sizeof(float), values.data(), 0, nullptr,
                                          nullptr),
          "clEnqueueReadBuffer");
    return values;
}

void Buffer::write(const std::vector<float>& values) {
    if (values.size() != elements_)
        throw std::invalid_argument("OpenCL buffer write size mismatch");
    auto& selected = runtime(device_);
    std::scoped_lock lock(selected.mutex());
    check(functions().enqueue_write_buffer(selected.queue(), handle_, api::true_value, 0,
                                           values.size() * sizeof(float), values.data(), 0, nullptr,
                                           nullptr),
          "clEnqueueWriteBuffer");
}

std::shared_ptr<Buffer> clone(const Buffer& input) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(input.device(), input.size());
    Kernel kernel(selected, "copy_op");
    auto in = input.handle(), out = output->handle();
    auto count = to_ulong(input.size());
    kernel.argument(0, in);
    kernel.argument(1, out);
    kernel.argument(2, count);
    kernel.run({input.size()});
    return output;
}

void accumulate(Buffer& destination, const Buffer& contribution) {
    if (destination.device() != contribution.device() || destination.size() != contribution.size())
        throw std::invalid_argument("OpenCL gradient accumulation mismatch");
    auto& selected = runtime(destination.device());
    std::scoped_lock lock(selected.mutex());
    Kernel kernel(selected, "accumulate_op");
    auto destination_handle = destination.handle(), contribution_handle = contribution.handle();
    auto count = to_ulong(destination.size());
    kernel.argument(0, destination_handle);
    kernel.argument(1, contribution_handle);
    kernel.argument(2, count);
    kernel.run({destination.size()});
}

bool available() noexcept {
    try {
        (void)runtime(Device::opencl(0, 0));
        return true;
    } catch (...) {
        return false;
    }
}

std::string device_name(const Device device) {
    return runtime(device).name();
}

void synchronize(const Device device) {
    auto& selected = runtime(device);
    std::scoped_lock lock(selected.mutex());
    check(functions().finish(selected.queue()), "clFinish");
}

void set_profiling(const Device device, const bool enabled) {
    auto& selected = runtime(device);
    std::scoped_lock lock(selected.mutex());
    selected.set_profiling(enabled);
}

std::vector<KernelProfile> profile(const Device device, const bool reset) {
    auto& selected = runtime(device);
    std::scoped_lock lock(selected.mutex());
    check(functions().finish(selected.queue()), "clFinish");
    return selected.profile(reset);
}

OpenCLRuntimeStatistics runtime_statistics(const Device device) {
    auto& selected = runtime(device);
    std::scoped_lock lock(selected.mutex());
    return selected.statistics();
}

void clear_memory_pool(const Device device) {
    auto& selected = runtime(device);
    std::scoped_lock lock(selected.mutex());
    check(functions().finish(selected.queue()), "clFinish");
    selected.clear_pool();
}

std::shared_ptr<Buffer> binary(const BinaryOperation operation, const Buffer& lhs,
                               const Buffer& rhs, const std::vector<std::size_t>& lhs_indices,
                               const std::vector<std::size_t>& rhs_indices) {
    if (lhs.device() != rhs.device() || lhs_indices.size() != rhs_indices.size())
        throw std::invalid_argument("invalid OpenCL binary operation inputs");
    auto& selected = runtime(lhs.device());
    std::scoped_lock lock(selected.mutex());
    RawMemory lhs_map(selected, lhs_indices.size() * sizeof(std::size_t), lhs_indices.data());
    RawMemory rhs_map(selected, rhs_indices.size() * sizeof(std::size_t), rhs_indices.data());
    auto output = std::make_shared<Buffer>(lhs.device(), lhs_indices.size());
    Kernel kernel(selected, "binary_op");
    const auto op = static_cast<api::UInt>(operation);
    const auto count = to_ulong(lhs_indices.size());
    auto a = lhs.handle(), b = rhs.handle(), ai = lhs_map.handle(), bi = rhs_map.handle(),
         out = output->handle();
    kernel.argument(0, op);
    kernel.argument(1, a);
    kernel.argument(2, b);
    kernel.argument(3, ai);
    kernel.argument(4, bi);
    kernel.argument(5, out);
    kernel.argument(6, count);
    kernel.run({lhs_indices.size()});
    return output;
}

std::shared_ptr<Buffer> unary(const UnaryOperation operation, const Buffer& input,
                              const float argument) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(input.device(), input.size());
    Kernel kernel(selected, "unary_op");
    const auto op = static_cast<api::UInt>(operation);
    const auto count = to_ulong(input.size());
    auto in = input.handle(), out = output->handle();
    kernel.argument(0, op);
    kernel.argument(1, in);
    kernel.argument(2, out);
    kernel.argument(3, argument);
    kernel.argument(4, count);
    kernel.run({input.size()});
    return output;
}

std::shared_ptr<Buffer> matrix_multiply(const Buffer& lhs, const Buffer& rhs,
                                        const std::size_t rows, const std::size_t inner,
                                        const std::size_t columns) {
    auto& selected = runtime(lhs.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(lhs.device(), rows * columns);
    Kernel kernel(selected, "matmul_op");
    auto a = lhs.handle(), b = rhs.handle(), out = output->handle();
    auto r = to_ulong(rows), i = to_ulong(inner), c = to_ulong(columns);
    kernel.argument(0, a);
    kernel.argument(1, b);
    kernel.argument(2, out);
    kernel.argument(3, r);
    kernel.argument(4, i);
    kernel.argument(5, c);
    constexpr std::size_t tile = 16;
    const auto global_rows = ((rows + tile - 1) / tile) * tile;
    const auto global_columns = ((columns + tile - 1) / tile) * tile;
    kernel.run({global_rows, global_columns}, {tile, tile});
    return output;
}

std::shared_ptr<Buffer> reduce_sum(const Buffer& input) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    constexpr std::size_t local_size = 128;
    const Buffer* current = &input;
    std::shared_ptr<Buffer> current_owner;
    auto count = input.size();
    do {
        const auto groups = (count + local_size * 2 - 1) / (local_size * 2);
        auto output = std::make_shared<Buffer>(input.device(), groups);
        Kernel kernel(selected, "sum_op");
        auto in = current->handle(), out = output->handle();
        auto elements = to_ulong(count);
        kernel.argument(0, in);
        kernel.argument(1, out);
        kernel.argument(2, elements);
        kernel.local_argument(3, local_size * sizeof(float));
        kernel.run({groups * local_size}, {local_size});
        current_owner = std::move(output);
        current = current_owner.get();
        count = groups;
    } while (count > 1);
    return current_owner;
}

std::shared_ptr<Buffer> reduce_axis_sum(const Buffer& input, const std::size_t outer,
                                        const std::size_t axis, const std::size_t inner) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(input.device(), outer * inner);
    Kernel kernel(selected, "axis_sum_op");
    auto in = input.handle(), out = output->handle();
    auto o = to_ulong(outer), a = to_ulong(axis), i = to_ulong(inner);
    kernel.argument(0, in);
    kernel.argument(1, out);
    kernel.argument(2, o);
    kernel.argument(3, a);
    kernel.argument(4, i);
    constexpr std::size_t local_size = 128;
    kernel.local_argument(5, local_size * sizeof(float));
    kernel.run({outer * inner * local_size}, {local_size});
    return output;
}

std::shared_ptr<Buffer> axis_sum_gradient(const Buffer& upstream, const std::size_t outer,
                                          const std::size_t axis, const std::size_t inner) {
    auto& selected = runtime(upstream.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(upstream.device(), outer * axis * inner);
    Kernel kernel(selected, "axis_sum_grad");
    auto in = upstream.handle(), out = output->handle();
    auto o = to_ulong(outer), a = to_ulong(axis), i = to_ulong(inner);
    kernel.argument(0, in);
    kernel.argument(1, out);
    kernel.argument(2, o);
    kernel.argument(3, a);
    kernel.argument(4, i);
    kernel.run({outer * axis * inner});
    return output;
}

std::shared_ptr<Buffer> transpose(const Buffer& input, std::size_t rows, std::size_t columns) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(input.device(), input.size());
    Kernel kernel(selected, "transpose_op");
    auto in = input.handle(), out = output->handle();
    auto r = to_ulong(rows), c = to_ulong(columns);
    kernel.argument(0, in);
    kernel.argument(1, out);
    kernel.argument(2, r);
    kernel.argument(3, c);
    kernel.run({rows, columns});
    return output;
}

std::shared_ptr<Buffer> softmax(const Buffer& input, std::size_t outer, std::size_t axis,
                                std::size_t inner) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(input.device(), input.size());
    Kernel kernel(selected, "softmax_op");
    auto in = input.handle(), out = output->handle();
    auto o = to_ulong(outer), a = to_ulong(axis), i = to_ulong(inner);
    kernel.argument(0, in);
    kernel.argument(1, out);
    kernel.argument(2, o);
    kernel.argument(3, a);
    kernel.argument(4, i);
    kernel.run({outer * inner});
    return output;
}

std::shared_ptr<Buffer>
convolution_2d(const Buffer& input, const Buffer& weight, const Buffer* bias, std::size_t batch,
               std::size_t input_channels, std::size_t input_height, std::size_t input_width,
               std::size_t output_channels, std::size_t kernel_height, std::size_t kernel_width,
               std::size_t output_height, std::size_t output_width, std::size_t stride_y,
               std::size_t stride_x, std::size_t padding_y, std::size_t padding_x) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(input.device(),
                                           batch * output_channels * output_height * output_width);
    constexpr std::size_t tile_height = 8;
    constexpr std::size_t tile_width = 8;
    const auto input_tile_height = (tile_height - 1) * stride_y + kernel_height;
    const auto input_tile_width = (tile_width - 1) * stride_x + kernel_width;
    const auto input_tile_bytes =
        input_channels * input_tile_height * input_tile_width * sizeof(float);
    const auto weight_tile_bytes = input_channels * kernel_height * kernel_width * sizeof(float);
    const auto use_tiled = input_tile_bytes + weight_tile_bytes <= selected.local_memory_size();
    Kernel kernel(selected, use_tiled ? "conv2d_tiled" : "conv2d_op");
    auto in = input.handle(), w = weight.handle();
    auto unused = bias ? bias->handle() : weight.handle(), out = output->handle();
    api::UInt has_bias = bias ? 1U : 0U;
    kernel.argument(0, in);
    kernel.argument(1, w);
    kernel.argument(2, unused);
    kernel.argument(3, has_bias);
    kernel.argument(4, out);
    const std::size_t values[] = {batch,           input_channels, input_height, input_width,
                                  output_channels, kernel_height,  kernel_width, output_height,
                                  output_width,    stride_y,       stride_x,     padding_y,
                                  padding_x};
    for (api::UInt index = 0; index < 13; ++index) {
        auto value = to_ulong(values[index]);
        kernel.argument(5 + index, value);
    }
    if (use_tiled) {
        kernel.local_argument(18, input_tile_bytes);
        kernel.local_argument(19, weight_tile_bytes);
        const auto global_height = ((output_height + tile_height - 1) / tile_height) * tile_height;
        const auto global_width = ((output_width + tile_width - 1) / tile_width) * tile_width;
        kernel.run({batch * output_channels, global_height, global_width},
                   {1, tile_height, tile_width});
    } else {
        kernel.run({output->size()});
    }
    return output;
}

std::shared_ptr<Buffer> pool_2d(const PoolingOperation operation, const Buffer& input,
                                const std::size_t batch, const std::size_t channels,
                                const std::size_t input_height, const std::size_t input_width,
                                const std::size_t kernel_height, const std::size_t kernel_width,
                                const std::size_t output_height, const std::size_t output_width,
                                const std::size_t stride_y, const std::size_t stride_x,
                                const std::size_t padding_y, const std::size_t padding_x) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto output =
        std::make_shared<Buffer>(input.device(), batch * channels * output_height * output_width);
    Kernel kernel(selected, "pool2d_op");
    auto op = static_cast<api::UInt>(operation);
    auto in = input.handle(), out = output->handle();
    kernel.argument(0, op);
    kernel.argument(1, in);
    kernel.argument(2, out);
    const std::size_t values[] = {batch,         channels,     input_height,  input_width,
                                  kernel_height, kernel_width, output_height, output_width,
                                  stride_y,      stride_x,     padding_y,     padding_x};
    for (api::UInt index = 0; index < 12; ++index) {
        auto value = to_ulong(values[index]);
        kernel.argument(3 + index, value);
    }
    kernel.run({output->size()});
    return output;
}

std::shared_ptr<Buffer> pool_2d_gradient(
    const PoolingOperation operation, const Buffer& input, const Buffer& upstream,
    const std::size_t batch, const std::size_t channels, const std::size_t input_height,
    const std::size_t input_width, const std::size_t kernel_height, const std::size_t kernel_width,
    const std::size_t output_height, const std::size_t output_width, const std::size_t stride_y,
    const std::size_t stride_x, const std::size_t padding_y, const std::size_t padding_x) {
    auto& selected = runtime(input.device());
    std::scoped_lock lock(selected.mutex());
    auto gradient = std::make_shared<Buffer>(input.device(), input.size());
    Kernel kernel(selected, "pool2d_grad");
    auto op = static_cast<api::UInt>(operation);
    auto in = input.handle(), up = upstream.handle(), out = gradient->handle();
    kernel.argument(0, op);
    kernel.argument(1, in);
    kernel.argument(2, up);
    kernel.argument(3, out);
    const std::size_t values[] = {batch,         channels,     input_height,  input_width,
                                  kernel_height, kernel_width, output_height, output_width,
                                  stride_y,      stride_x,     padding_y,     padding_x};
    for (api::UInt index = 0; index < 12; ++index) {
        auto value = to_ulong(values[index]);
        kernel.argument(4 + index, value);
    }
    kernel.run({input.size()});
    return gradient;
}

std::shared_ptr<Buffer> binary_gradient(const BinaryOperation operation, const bool lhs_gradient,
                                        const Buffer& lhs, const Buffer& rhs,
                                        const std::vector<std::size_t>& lhs_indices,
                                        const std::vector<std::size_t>& rhs_indices,
                                        const Buffer& upstream, const std::size_t parent_size) {
    auto& selected = runtime(lhs.device());
    auto gradient = std::make_shared<Buffer>(lhs.device(), parent_size);
    {
        std::scoped_lock lock(selected.mutex());
        RawMemory lhs_map(selected, lhs_indices.size() * sizeof(std::size_t), lhs_indices.data());
        RawMemory rhs_map(selected, rhs_indices.size() * sizeof(std::size_t), rhs_indices.data());
        Kernel kernel(selected, "binary_grad");
        auto op = static_cast<api::UInt>(operation);
        api::UInt lhs_flag = lhs_gradient ? 1U : 0U;
        auto a = lhs.handle(), b = rhs.handle(), ai = lhs_map.handle(), bi = rhs_map.handle(),
             up = upstream.handle(), out = gradient->handle();
        auto output_count = to_ulong(upstream.size()), parents = to_ulong(parent_size);
        kernel.argument(0, op);
        kernel.argument(1, lhs_flag);
        kernel.argument(2, a);
        kernel.argument(3, b);
        kernel.argument(4, ai);
        kernel.argument(5, bi);
        kernel.argument(6, up);
        kernel.argument(7, out);
        kernel.argument(8, output_count);
        kernel.argument(9, parents);
        kernel.run({parent_size});
    }
    return gradient;
}

std::shared_ptr<Buffer> unary_gradient(const UnaryOperation operation, const Buffer& input,
                                       const Buffer& upstream, const float argument) {
    auto& selected = runtime(input.device());
    auto gradient = std::make_shared<Buffer>(input.device(), input.size());
    {
        std::scoped_lock lock(selected.mutex());
        Kernel kernel(selected, "unary_grad");
        auto op = static_cast<api::UInt>(operation);
        auto in = input.handle(), up = upstream.handle(), out = gradient->handle();
        auto count = to_ulong(input.size());
        kernel.argument(0, op);
        kernel.argument(1, in);
        kernel.argument(2, up);
        kernel.argument(3, out);
        kernel.argument(4, argument);
        kernel.argument(5, count);
        kernel.run({input.size()});
    }
    return gradient;
}

std::shared_ptr<Buffer> sum_gradient(const Buffer& upstream, const std::size_t count) {
    auto& selected = runtime(upstream.device());
    std::scoped_lock lock(selected.mutex());
    auto gradient = std::make_shared<Buffer>(upstream.device(), count);
    Kernel kernel(selected, "sum_grad");
    auto up = upstream.handle(), out = gradient->handle();
    auto elements = to_ulong(count);
    kernel.argument(0, up);
    kernel.argument(1, out);
    kernel.argument(2, elements);
    kernel.run({count});
    return gradient;
}

std::shared_ptr<Buffer> matmul_lhs_gradient(const Buffer& rhs, const Buffer& upstream,
                                            const std::size_t rows, const std::size_t inner,
                                            const std::size_t columns) {
    auto rhs_transposed = transpose(rhs, inner, columns);
    return matrix_multiply(upstream, *rhs_transposed, rows, columns, inner);
}

std::shared_ptr<Buffer> matmul_rhs_gradient(const Buffer& lhs, const Buffer& upstream,
                                            const std::size_t rows, const std::size_t inner,
                                            const std::size_t columns) {
    auto lhs_transposed = transpose(lhs, rows, inner);
    return matrix_multiply(*lhs_transposed, upstream, inner, rows, columns);
}

std::shared_ptr<Buffer> softmax_gradient(const Buffer& output, const Buffer& upstream,
                                         const std::size_t outer, const std::size_t axis,
                                         const std::size_t inner) {
    auto& selected = runtime(output.device());
    auto gradient = std::make_shared<Buffer>(output.device(), output.size());
    {
        std::scoped_lock lock(selected.mutex());
        Kernel kernel(selected, "softmax_grad");
        auto out = output.handle(), up = upstream.handle(), grad = gradient->handle();
        auto o = to_ulong(outer), a = to_ulong(axis), i = to_ulong(inner);
        kernel.argument(0, out);
        kernel.argument(1, up);
        kernel.argument(2, grad);
        kernel.argument(3, o);
        kernel.argument(4, a);
        kernel.argument(5, i);
        kernel.run({outer * inner});
    }
    return gradient;
}

ConvolutionGradients convolution_2d_gradients(
    const Buffer& input, const Buffer& weight, const Buffer& upstream, const bool input_required,
    const bool weight_required, const bool bias_required, const std::size_t batch,
    const std::size_t input_channels, const std::size_t input_height, const std::size_t input_width,
    const std::size_t output_channels, const std::size_t kernel_height,
    const std::size_t kernel_width, const std::size_t output_height, const std::size_t output_width,
    const std::size_t stride_y, const std::size_t stride_x, const std::size_t padding_y,
    const std::size_t padding_x) {
    ConvolutionGradients result;
    auto& selected = runtime(input.device());
    std::shared_ptr<Buffer> input_result, weight_result, bias_result;
    {
        std::scoped_lock lock(selected.mutex());
        const std::size_t values[] = {batch,           input_channels, input_height, input_width,
                                      output_channels, kernel_height,  kernel_width, output_height,
                                      output_width,    stride_y,       stride_x,     padding_y,
                                      padding_x};
        if (input_required) {
            input_result = std::make_shared<Buffer>(input.device(), input.size());
            Kernel kernel(selected, "conv_input_grad");
            auto w = weight.handle(), up = upstream.handle(), out = input_result->handle();
            kernel.argument(0, w);
            kernel.argument(1, up);
            kernel.argument(2, out);
            for (api::UInt index = 0; index < 13; ++index) {
                auto value = to_ulong(values[index]);
                kernel.argument(3 + index, value);
            }
            kernel.run({input.size()});
        }
        if (weight_required) {
            weight_result = std::make_shared<Buffer>(input.device(), weight.size());
            Kernel kernel(selected, "conv_weight_grad");
            auto in = input.handle(), up = upstream.handle(), out = weight_result->handle();
            kernel.argument(0, in);
            kernel.argument(1, up);
            kernel.argument(2, out);
            for (api::UInt index = 0; index < 13; ++index) {
                auto value = to_ulong(values[index]);
                kernel.argument(3 + index, value);
            }
            kernel.run({weight.size()});
        }
        if (bias_required) {
            bias_result = std::make_shared<Buffer>(input.device(), output_channels);
            Kernel kernel(selected, "conv_bias_grad");
            auto up = upstream.handle(), out = bias_result->handle();
            auto b = to_ulong(batch), oc = to_ulong(output_channels), oh = to_ulong(output_height),
                 ow = to_ulong(output_width);
            kernel.argument(0, up);
            kernel.argument(1, out);
            kernel.argument(2, b);
            kernel.argument(3, oc);
            kernel.argument(4, oh);
            kernel.argument(5, ow);
            kernel.run({output_channels});
        }
    }
    result.input = std::move(input_result);
    result.weight = std::move(weight_result);
    result.bias = std::move(bias_result);
    return result;
}

std::shared_ptr<Buffer> binary_cross_entropy_loss(const Buffer& prediction, const Buffer& target,
                                                  const float epsilon, const bool with_logits) {
    auto& selected = runtime(prediction.device());
    std::scoped_lock lock(selected.mutex());
    auto output = std::make_shared<Buffer>(prediction.device(), 1);
    Kernel kernel(selected, "binary_loss");
    api::UInt logits = with_logits ? 1U : 0U;
    auto p = prediction.handle(), t = target.handle(), out = output->handle();
    auto count = to_ulong(prediction.size());
    kernel.argument(0, logits);
    kernel.argument(1, p);
    kernel.argument(2, t);
    kernel.argument(3, out);
    kernel.argument(4, epsilon);
    kernel.argument(5, count);
    kernel.run({1});
    return output;
}

BinaryLossGradients binary_cross_entropy_gradients(const Buffer& prediction, const Buffer& target,
                                                   const Buffer& upstream, const float epsilon,
                                                   const bool with_logits,
                                                   const bool prediction_required,
                                                   const bool target_required) {
    BinaryLossGradients result;
    auto& selected = runtime(prediction.device());
    std::shared_ptr<Buffer> prediction_result, target_result;
    {
        std::scoped_lock lock(selected.mutex());
        auto run = [&](bool target_gradient) {
            auto output = std::make_shared<Buffer>(prediction.device(), prediction.size());
            Kernel kernel(selected, "binary_loss_grad");
            api::UInt logits = with_logits ? 1U : 0U, target_flag = target_gradient ? 1U : 0U;
            auto p = prediction.handle(), t = target.handle(), out = output->handle();
            auto count = to_ulong(prediction.size());
            kernel.argument(0, logits);
            kernel.argument(1, target_flag);
            kernel.argument(2, p);
            kernel.argument(3, t);
            kernel.argument(4, out);
            auto up = upstream.handle();
            kernel.argument(5, up);
            kernel.argument(6, epsilon);
            kernel.argument(7, count);
            kernel.run({prediction.size()});
            return output;
        };
        if (prediction_required)
            prediction_result = run(false);
        if (target_required)
            target_result = run(true);
    }
    result.prediction = std::move(prediction_result);
    result.target = std::move(target_result);
    return result;
}

std::shared_ptr<Buffer> cross_entropy_loss(const Buffer& logits,
                                           const std::vector<std::size_t>& labels,
                                           const std::size_t batch, const std::size_t classes) {
    auto& selected = runtime(logits.device());
    std::scoped_lock lock(selected.mutex());
    RawMemory label_memory(selected, labels.size() * sizeof(std::size_t), labels.data());
    auto output = std::make_shared<Buffer>(logits.device(), 1);
    Kernel kernel(selected, "cross_entropy_loss_op");
    auto in = logits.handle(), label = label_memory.handle(), out = output->handle();
    auto b = to_ulong(batch), c = to_ulong(classes);
    kernel.argument(0, in);
    kernel.argument(1, label);
    kernel.argument(2, out);
    kernel.argument(3, b);
    kernel.argument(4, c);
    kernel.run({1});
    return output;
}

std::shared_ptr<Buffer> cross_entropy_gradient(const Buffer& logits,
                                               const std::vector<std::size_t>& labels,
                                               const Buffer& upstream, const std::size_t batch,
                                               const std::size_t classes) {
    auto& selected = runtime(logits.device());
    auto output = std::make_shared<Buffer>(logits.device(), logits.size());
    {
        std::scoped_lock lock(selected.mutex());
        RawMemory label_memory(selected, labels.size() * sizeof(std::size_t), labels.data());
        Kernel kernel(selected, "cross_entropy_grad_op");
        auto in = logits.handle(), label = label_memory.handle(), out = output->handle();
        auto b = to_ulong(batch), c = to_ulong(classes);
        kernel.argument(0, in);
        kernel.argument(1, label);
        kernel.argument(2, out);
        auto up = upstream.handle();
        kernel.argument(3, up);
        kernel.argument(4, b);
        kernel.argument(5, c);
        kernel.run({batch});
    }
    return output;
}

void sgd_update(Buffer& parameter, const Buffer& gradient, Buffer& velocity,
                const float learning_rate, const float momentum, const float weight_decay) {
    if (gradient.device() != parameter.device() || velocity.device() != parameter.device() ||
        gradient.size() != parameter.size() || velocity.size() != parameter.size())
        throw std::invalid_argument("SGD state size mismatch");
    auto& selected = runtime(parameter.device());
    std::scoped_lock lock(selected.mutex());
    Kernel kernel(selected, "sgd_update_op");
    auto p = parameter.handle(), g = gradient.handle(), v = velocity.handle();
    api::UInt use_momentum = momentum != 0.0F ? 1U : 0U;
    auto count = to_ulong(parameter.size());
    kernel.argument(0, p);
    kernel.argument(1, g);
    kernel.argument(2, v);
    kernel.argument(3, learning_rate);
    kernel.argument(4, momentum);
    kernel.argument(5, weight_decay);
    kernel.argument(6, use_momentum);
    kernel.argument(7, count);
    kernel.run({parameter.size()});
}

void adam_update(Buffer& parameter, const Buffer& gradient, Buffer& first_moment,
                 Buffer& second_moment, const float learning_rate, const float beta1,
                 const float beta2, const float epsilon, const float weight_decay,
                 const float first_correction, const float second_correction,
                 const bool decoupled_weight_decay) {
    if (gradient.device() != parameter.device() || first_moment.device() != parameter.device() ||
        second_moment.device() != parameter.device() || gradient.size() != parameter.size() ||
        first_moment.size() != parameter.size() || second_moment.size() != parameter.size())
        throw std::invalid_argument("Adam state size mismatch");
    auto& selected = runtime(parameter.device());
    std::scoped_lock lock(selected.mutex());
    Kernel kernel(selected, "adam_update_op");
    auto p = parameter.handle(), g = gradient.handle(), first = first_moment.handle(),
         second = second_moment.handle();
    api::UInt decoupled = decoupled_weight_decay ? 1U : 0U;
    auto count = to_ulong(parameter.size());
    kernel.argument(0, p);
    kernel.argument(1, g);
    kernel.argument(2, first);
    kernel.argument(3, second);
    kernel.argument(4, learning_rate);
    kernel.argument(5, beta1);
    kernel.argument(6, beta2);
    kernel.argument(7, epsilon);
    kernel.argument(8, weight_decay);
    kernel.argument(9, first_correction);
    kernel.argument(10, second_correction);
    kernel.argument(11, decoupled);
    kernel.argument(12, count);
    kernel.run({parameter.size()});
}

} // namespace clnn::opencl
