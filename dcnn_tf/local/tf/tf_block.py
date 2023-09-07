import tensorflow as tf
from tensorflow.python.framework import ops


def __get_variable(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)

def batch_norm_wrapper(inputs, is_training, decay=0.99, epsilon=1e-3, name_prefix=''):
    gamma = __get_variable(name_prefix + 'gamma', inputs.get_shape()[-1], tf.constant_initializer(1.0))
    beta = __get_variable(name_prefix + 'beta', inputs.get_shape()[-1], tf.constant_initializer(0.0))
    pop_mean = __get_variable(name_prefix + 'mean', inputs.get_shape()[-1], tf.constant_initializer(0.0),
                              trainable=False)
    pop_var = __get_variable(name_prefix + 'variance', inputs.get_shape()[-1], tf.constant_initializer(1.0),
                             trainable=False)
    axis = list(range(len(inputs.get_shape()) - 1))

    def in_training():
        batch_mean, batch_var = tf.nn.moments(inputs, axis)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

    def in_evaluation():
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

    return tf.cond(is_training, lambda: in_training(), lambda: in_evaluation())


def conv2d_bn_layer_v2(inpt, filter_shape, strides, phase):
    filter_ = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=strides, padding="SAME")
    out = batch_norm_wrapper(conv, decay=0.95, is_training=phase)

    return out

def sbsa(inpt,phase,group=4):
    inpt_depth = inpt.get_shape().as_list()[-1]
    inpt_width = inpt.get_shape().as_list()[-2]
    
    with tf.variable_scope("aux_q"):
        filter_reduce_q = tf.Variable(tf.truncated_normal([1,1,inpt_depth,inpt_depth], stddev=0.1), name="reduce")
        q = tf.nn.conv2d(inpt,filter=filter_reduce_q, strides=[1, 1, 1, 1], padding="SAME")
    with tf.variable_scope("aux_k"):
        filter_reduce_k = tf.Variable(tf.truncated_normal([1,1,inpt_depth,inpt_depth], stddev=0.1), name="reduce")
        k = tf.nn.conv2d(inpt,filter=filter_reduce_k, strides=[1, 1, 1, 1], padding="SAME")
    with tf.variable_scope("aux_v"):
        filter_reduce_v = tf.Variable(tf.truncated_normal([1,1,inpt_depth,inpt_depth], stddev=0.1), name="reduce")
        v = tf.nn.conv2d(inpt,filter=filter_reduce_v, strides=[1, 1, 1, 1], padding="SAME")
        
    q = tf.reshape(q,[tf.shape(inpt)[0],tf.shape(inpt)[1],inpt_width*inpt_depth])
    k = tf.reshape(k,[tf.shape(inpt)[0],tf.shape(inpt)[1],inpt_width*inpt_depth])
    v = tf.reshape(v,[tf.shape(inpt)[0],tf.shape(inpt)[1],inpt_width*inpt_depth])
        
    q_list = tf.split(q,group,-1)
    k_list = tf.split(k,group,-1)
    v_list = tf.split(v,group,-1)
    out_list = []
    for i in range(group):
        temp_q,temp_k,temp_v = q_list[i],k_list[i],v_list[i]
        k_t_temp = tf.transpose(temp_k, [0,2,1])
        my_alpha_temp = tf.nn.softmax(tf.einsum("ijk,ikm->ijm",temp_q,k_t_temp)/(tf.sqrt(inpt_width/group*inpt_depth)),2)
        v_alpha_temp = tf.einsum("ijk,imj->imk",temp_v,my_alpha_temp)
        out_list.append(v_alpha_temp)
    v_alpha = tf.concat(out_list,-1)
    v_alpha = tf.reshape(v_alpha, [tf.shape(inpt)[0],tf.shape(inpt)[1],inpt_width,inpt_depth])
    
    return v_alpha


def dykconv_bn_layer_v2(inpt, phase, head=2, in_ksize=7, ratio=4):
    inpt_depth = inpt.get_shape().as_list()[-1]
    inpt_width = inpt.get_shape().as_list()[-2]
    
    pro_down = tf.nn.relu(conv2d_bn_layer_v2(inpt, [3,3,inpt_depth,int(inpt_depth/ratio)], [1,1,1,1],phase))
    
    with tf.variable_scope("global"):
        v_alpha_list = []
        for i in range(head):
            with tf.variable_scope("self_head_%d"%i):
                v_alpha_list.append(sbsa(pro_down,phase))
            
        ctx_global = tf.concat(v_alpha_list,-1)
        ctx_global = tf.nn.relu(conv2d_bn_layer_v2(ctx_global,[3,3,int(inpt_depth/ratio*head),inpt_depth],[1,1,1,1],phase))
        
    with tf.variable_scope("local"):
        filter_depwise = tf.Variable(tf.zeros([in_ksize, in_ksize, int(inpt_depth/ratio), 1])+1.0/(in_ksize**2), name="reduce")
        inpt_avg=tf.nn.depthwise_conv2d(pro_down,filter_depwise,[1,1,1,1],padding="SAME")
        ctx_local = tf.nn.relu(conv2d_bn_layer_v2(inpt_avg,[1,1,int(inpt_depth/ratio),inpt_depth],[1,1,1,1],phase))
        
    with tf.variable_scope("acve"):
        ctx_global_local = ctx_global+ctx_local+inpt
    
    with tf.variable_scope("mod_fac"):
        mod_fac = 1.0 + tf.nn.tanh(conv2d_bn_layer_v2(ctx_global_local, [1,1,inpt_depth,inpt_depth],[1,1,1,1],phase))
    
    '''
    A trick for the dynamic convolution implementation:
    output = X*K(X), K(X) = mod_fac*K1+(2-mod_fac)*K2 => output = (mod_fac*X)*K1 + ((2-mod_fac)*X)*K2
    '''
    
    with tf.variable_scope("_sks1"):
        out_k1 = inpt*mod_fac
        with tf.variable_scope("_re_filter"):
            out_k1 = conv2d_bn_layer_v2(out_k1,[3,3,inpt_depth,inpt_depth],[1,1,1,1],phase)
    
    with tf.variable_scope("_sks2"):
        out_k2 = inpt*(2.0-mod_fac)
        with tf.variable_scope("_re_filter"):
            out_k2 = conv2d_bn_layer_v2(out_k2,[3,3,inpt_depth,inpt_depth],[1,1,1,1],phase)
    
    return out_k1+out_k2


def residual_block_dykcnn_v2(inpt, output_depth, phase, down_sample=False, projection=True):
    input_depth = inpt.get_shape().as_list()[3]
    # if down_sample:
    #     filter_ = [1,2,2,1]
    #     inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    if down_sample:
        stride = 2
    else:
        stride = 1
        
    with tf.variable_scope("_conv1"):
        if input_depth != output_depth:
            conv1 = conv2d_bn_layer_v2(inpt, [3, 3, input_depth, output_depth], [1,1,stride,1], phase)
        else:
            conv1 = conv2d_bn_layer_v2(inpt, [3, 3, input_depth, output_depth], [1,1,stride,1], phase)
        conv1 = tf.nn.relu(conv1)
    with tf.variable_scope("_conv2"):
        conv2 = dykconv_bn_layer_v2(conv1, phase)
        
    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv2d_bn_layer_v2(inpt, [1, 1, input_depth, output_depth], [1,1,stride,1], phase)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    out = conv2 + input_layer
    return tf.nn.relu(out)

def residual_block_v2(inpt, output_depth, phase, down_sample=False, projection=True):
    input_depth = inpt.get_shape().as_list()[3]
    # if down_sample:
    #     filter_ = [1,2,2,1]
    #     inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    if down_sample:
        stride = 2
    else:
        stride = 1
    with tf.variable_scope("_conv1"):
        conv1 = conv2d_bn_layer_v2(inpt, [3, 3, input_depth, output_depth], [1,1,stride,1], phase)
        conv1 = tf.nn.relu(conv1)
    with tf.variable_scope("_conv2"):
        conv2 = conv2d_bn_layer_v2(conv1, [3, 3, output_depth, output_depth], [1,1,1,1], phase)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv2d_bn_layer_v2(inpt, [1, 1, input_depth, output_depth], [1,1,stride,1], phase)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    out = conv2 + input_layer
    return tf.nn.relu(out)




def AM_logits_compute_use_onehot(embeddings, label_onehot, embedding_size, nrof_classes, is_training, s = 30, m = 0.15):
    

    with tf.name_scope('AM_logits'):
        kernel = tf.get_variable(name='kernel',dtype=tf.float32,shape=[embedding_size,nrof_classes],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings_norm')
        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady

        def in_training():
            phi = cos_theta - m 
            #label_onehot = tf.one_hot(label_batch, nrof_classes)
            adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)
            return adjust_theta

        def in_evaluation():
            return s*cos_theta

        return tf.cond(is_training, lambda: in_training(), lambda: in_evaluation())
