import math

import tensorflow as tf


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    # Using float32 as default dtype (exemption #2)
    # No explicit device setting needed (exemption #3)
    t = tf.cast(points_to_project, tf.float32)
    w = tf.cast(tf.identity(w_hyperplane), tf.float32)
    b = tf.cast(tf.identity(b_hyperplane), tf.float32)

    sign = 2 * tf.cast(tf.reduce_sum(w * t, axis=1) - b >= 0, tf.float32) - 1
    w = w * tf.expand_dims(sign, 1)
    b = b * sign

    a = tf.cast(w < 0, tf.float32)
    d = (a - t) * tf.cast(w != 0, tf.float32)

    p = a - t * (2 * a - 1)
    indp = tf.argsort(p, axis=1)

    b = b - tf.reduce_sum(w * t, axis=1)
    b0 = tf.reduce_sum(w * d, axis=1)

    indp2 = tf.reverse(indp, axis=[1])
    ws = tf.gather(w, indp2, axis=1, batch_dims=1)
    bs2 = -ws * tf.gather(d, indp2, axis=1, batch_dims=1)

    s = tf.cumsum(tf.abs(ws), axis=1)
    sb = tf.cumsum(bs2, axis=1) + tf.expand_dims(b0, 1)

    b2 = sb[:, -1] - s[:, -1] * tf.squeeze(tf.gather(p, indp[:, 0:1], axis=1, batch_dims=1), axis=1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    
    c2_indices = tf.where(c2)
    num_c2 = tf.shape(c2_indices)[0]
    
    if num_c2 > 0:
        lb = tf.zeros(num_c2, dtype=tf.float32)
        ub = tf.fill([num_c2], tf.cast(tf.shape(w)[1] - 1, tf.float32))
        nitermax = math.ceil(math.log2(tf.shape(w)[1].numpy() if hasattr(tf.shape(w)[1], 'numpy') else int(tf.shape(w)[1])))

        indp_ = tf.gather(indp, c2_indices[:, 0])
        sb_ = tf.gather(sb, c2_indices[:, 0])
        s_ = tf.gather(s, c2_indices[:, 0])
        p_ = tf.gather(p, c2_indices[:, 0])
        b_ = tf.gather(b, c2_indices[:, 0])
        
        for counter in range(nitermax):
            counter4 = tf.floor((lb + ub) / 2)
            
            counter2 = tf.cast(counter4, tf.int32)
            counter2 = tf.expand_dims(counter2, 1)
            indcurr = tf.gather(indp_, tf.shape(indp_)[1] - 1 - counter2, axis=1, batch_dims=1)
            b2 = tf.squeeze(tf.gather(sb_, counter2, axis=1, batch_dims=1) - 
                           tf.gather(s_, counter2, axis=1, batch_dims=1) * 
                           tf.gather(p_, indcurr, axis=1, batch_dims=1), axis=1)
            c = b_ - b2 > 0
            
            lb = tf.where(c, counter4, lb)
            ub = tf.where(c, ub, counter4)
        
        lb = tf.cast(lb, tf.int32)

    if tf.reduce_any(c_l):
        c_l_indices = tf.where(c_l)
        lmbd_opt = tf.expand_dims(tf.maximum((tf.gather(b, c_l_indices[:, 0]) - 
                                             tf.gather(sb[:, -1], c_l_indices[:, 0])) / 
                                            (-tf.gather(s[:, -1], c_l_indices[:, 0])), 0), -1)
        # Update d for c_l indices
        updates = (2 * tf.gather(a, c_l_indices[:, 0]) - 1) * lmbd_opt
        d = tf.tensor_scatter_nd_update(d, c_l_indices, updates)

    if num_c2 > 0:
        lmbd_opt = tf.expand_dims(tf.maximum(
            (b_ - tf.gather(sb_, lb, axis=1, batch_dims=1)) / 
            (-tf.gather(s_, lb, axis=1, batch_dims=1)), 0), -1)
        
        # Get values at c2 indices
        d_c2 = tf.gather(d, c2_indices[:, 0])
        a_c2 = tf.gather(a, c2_indices[:, 0])
        
        # Update d for c2 indices
        updates = tf.minimum(lmbd_opt, d_c2) * a_c2 + tf.maximum(-lmbd_opt, d_c2) * (1 - a_c2)
        d = tf.tensor_scatter_nd_update(d, c2_indices, updates)

    return d * tf.cast(w != 0, tf.float32)


def projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    # Using float32 as default dtype (exemption #2)
    # No explicit device setting needed (exemption #3)
    t = tf.cast(points_to_project, tf.float32)
    w = tf.cast(tf.identity(w_hyperplane), tf.float32)
    b = tf.cast(b_hyperplane, tf.float32)

    c = tf.reduce_sum(w * t, axis=1) - b
    ind2 = 2 * tf.cast(c >= 0, tf.float32) - 1
    w = w * tf.expand_dims(ind2, 1)
    c = c * ind2

    r = tf.maximum(t / w, (t - 1) / w)
    r = tf.clip_by_value(r, -1e12, 1e12)
    r = tf.where(tf.abs(w) < 1e-8, 1e12, r)
    r = tf.where(r == -1e12, -r, r)
    
    rs, indr = tf.nn.top_k(-r, k=tf.shape(r)[1], sorted=True)
    rs = -rs
    indr = tf.cast(indr, tf.int32)
    
    rs2 = tf.pad(rs[:, 1:], [[0, 0], [0, 1]])
    rs = tf.where(rs == 1e12, 0, rs)
    rs2 = tf.where(rs2 == 1e12, 0, rs2)

    w3s = tf.gather(w ** 2, indr, axis=1, batch_dims=1)
    w5 = tf.reduce_sum(w3s, axis=1, keepdims=True)
    ws = w5 - tf.cumsum(w3s, axis=1)
    
    d = -(r * w)
    d = d * tf.cast(tf.abs(w) > 1e-8, tf.float32)
    
    s = tf.concat([-w5 * rs[:, 0:1], 
                   tf.cumsum((-rs2 + rs) * ws, axis=1) - w5 * rs[:, 0:1]], axis=1)

    c4 = s[:, 0] + c < 0
    c3 = tf.reduce_sum(d * w, axis=1) + c > 0
    c2 = ~(c4 | c3)

    c2_indices = tf.where(c2)
    num_c2 = tf.shape(c2_indices)[0]
    
    if num_c2 > 0:
        lb = tf.zeros(num_c2, dtype=tf.float32)
        ub = tf.fill([num_c2], tf.cast(tf.shape(w)[1] - 1, tf.float32))
        nitermax = math.ceil(math.log2(tf.shape(w)[1].numpy() if hasattr(tf.shape(w)[1], 'numpy') else int(tf.shape(w)[1])))

        s_ = tf.gather(s, c2_indices[:, 0])
        c_ = tf.gather(c, c2_indices[:, 0])
        
        for counter in range(nitermax):
            counter4 = tf.floor((lb + ub) / 2)
            counter2 = tf.expand_dims(tf.cast(counter4, tf.int32), 1)
            c3 = tf.squeeze(tf.gather(s_, counter2, axis=1, batch_dims=1), axis=1) + c_ > 0
            lb = tf.where(c3, counter4, lb)
            ub = tf.where(c3, ub, counter4)
        
        lb = tf.cast(lb, tf.int32)

    if tf.reduce_any(c4):
        c4_indices = tf.where(c4)
        alpha = tf.gather(c, c4_indices[:, 0]) / tf.squeeze(tf.gather(w5, c4_indices[:, 0]), axis=-1)
        updates = -tf.expand_dims(alpha, -1) * tf.gather(w, c4_indices[:, 0])
        d = tf.tensor_scatter_nd_update(d, c4_indices, updates)

    if num_c2 > 0:
        ws_c2_lb = tf.gather(tf.gather(ws, c2_indices[:, 0]), lb, axis=1, batch_dims=1)
        rs_c2_lb = tf.gather(tf.gather(rs, c2_indices[:, 0]), lb, axis=1, batch_dims=1)
        s_c2_lb = tf.gather(s_, lb, axis=1, batch_dims=1)
        
        alpha = (s_c2_lb + c_) / ws_c2_lb + rs_c2_lb
        alpha = tf.where(ws_c2_lb == 0, 0, alpha)
        
        r_c2 = tf.gather(r, c2_indices[:, 0])
        c5 = tf.cast(tf.expand_dims(alpha, -1) > r_c2, tf.float32)
        
        d_c2 = tf.gather(d, c2_indices[:, 0])
        w_c2 = tf.gather(w, c2_indices[:, 0])
        
        updates = d_c2 * c5 - tf.expand_dims(alpha, -1) * w_c2 * (1 - c5)
        d = tf.tensor_scatter_nd_update(d, c2_indices, updates)

    return d * tf.cast(tf.abs(w) > 1e-8, tf.float32)


def projection_l1(points_to_project, w_hyperplane, b_hyperplane):
    # Using float32 as default dtype (exemption #2)
    # No explicit device setting needed (exemption #3)
    t = tf.cast(points_to_project, tf.float32)
    w = tf.cast(tf.identity(w_hyperplane), tf.float32)
    b = tf.cast(b_hyperplane, tf.float32)

    c = tf.reduce_sum(w * t, axis=1) - b
    ind2 = 2 * tf.cast(c >= 0, tf.float32) - 1
    w = w * tf.expand_dims(ind2, 1)
    c = c * ind2

    r = tf.clip_by_value(tf.abs(1 / w), 0, 1e12)
    indr = tf.argsort(r, axis=1)
    indr_rev = tf.argsort(indr, axis=1)

    c6 = tf.cast(w < 0, tf.float32)
    d = (-t + c6) * tf.cast(w != 0, tf.float32)
    
    ds = tf.gather(tf.minimum(-w * t, w * (1 - t)), indr, axis=1, batch_dims=1)
    ds2 = tf.concat([tf.expand_dims(c, -1), ds], axis=1)
    s = tf.cumsum(ds2, axis=1)

    c2 = s[:, -1] < 0

    c2_indices = tf.where(c2)
    num_c2 = tf.shape(c2_indices)[0]
    
    if num_c2 > 0:
        lb = tf.zeros(num_c2, dtype=tf.float32)
        ub = tf.fill([num_c2], tf.cast(tf.shape(s)[1], tf.float32))
        nitermax = math.ceil(math.log2(tf.shape(w)[1].numpy() if hasattr(tf.shape(w)[1], 'numpy') else int(tf.shape(w)[1])))

        s_ = tf.gather(s, c2_indices[:, 0])
        
        for counter in range(nitermax):
            counter4 = tf.floor((lb + ub) / 2)
            counter2 = tf.expand_dims(tf.cast(counter4, tf.int32), 1)
            c3 = tf.squeeze(tf.gather(s_, counter2, axis=1, batch_dims=1), axis=1) > 0
            lb = tf.where(c3, counter4, lb)
            ub = tf.where(c3, ub, counter4)
        
        lb2 = tf.cast(lb, tf.int32)

        # Get indr values for c2 indices
        indr_c2 = tf.gather(indr, c2_indices[:, 0])
        indr_selected = tf.squeeze(tf.gather(indr_c2, tf.expand_dims(lb2, 1), axis=1, batch_dims=1), axis=1)
        
        # Calculate alpha
        s_c2_lb2 = tf.gather(tf.gather(s, c2_indices[:, 0]), lb2, axis=1, batch_dims=1)
        w_c2_indr = tf.gather_nd(w, tf.stack([c2_indices[:, 0], indr_selected], axis=1))
        alpha = -s_c2_lb2 / w_c2_indr
        
        # Create mask for updates
        u2 = tf.range(tf.shape(w)[1], dtype=tf.float32)
        u2 = tf.expand_dims(u2, 0)
        c5 = u2 < tf.expand_dims(lb, -1)
        
        # Get indr_rev values for c2
        indr_rev_c2 = tf.gather(indr_rev, c2_indices[:, 0])
        
        # Create u3 mask
        u3_list = []
        for i in range(num_c2):
            mask_row = tf.gather(c5[i], indr_rev_c2[i])
            u3_list.append(mask_row)
        u3 = tf.stack(u3_list, axis=0)
        
        # Update d for c2 indices with mask
        d_c2 = tf.gather(d, c2_indices[:, 0])
        d_c2_masked = d_c2 * tf.cast(u3, tf.float32)
        
        # Set alpha values at indr positions
        # First update with masked values
        d = tf.tensor_scatter_nd_update(d, c2_indices, d_c2_masked)
        
        # Then update specific indr positions with alpha
        alpha_indices = tf.stack([c2_indices[:, 0], indr_selected], axis=1)
        d = tf.tensor_scatter_nd_update(d, alpha_indices, alpha)

    return d * tf.cast(tf.abs(w) > 1e-8, tf.float32)