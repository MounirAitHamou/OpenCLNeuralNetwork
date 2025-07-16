__kernel void sgd_update_parameters(
    __global float* params_buf,
    __global const float* grads_buf,
    const unsigned int num_elements,
    const float learning_rate,
    const float weight_decay_rate) {

    unsigned int gid = get_global_id(0);

    if (gid >= num_elements) return;
    
    params_buf[gid] -= learning_rate * (grads_buf[gid] + weight_decay_rate * params_buf[gid]);
}