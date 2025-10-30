import math

import paddle
import paddle.nn.functional as F


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane.clone()

    sign = 2 * ((w * t).sum(1) - b >= 0).astype('float32') - 1
    w = w * sign.unsqueeze(1)
    b = b * sign

    a = (w < 0).astype('float32')
    d = (a - t) * (w != 0).astype('float32')

    p = a - t * (2 * a - 1)
    indp = paddle.argsort(p, axis=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)

    indp2 = paddle.flip(indp, axis=[1])
    # Use take_along_axis instead of gather for 2D indexing
    ws = paddle.take_along_axis(w, indp2, axis=1)
    bs2 = -ws * paddle.take_along_axis(d, indp2, axis=1)

    s = paddle.cumsum(ws.abs(), axis=1)
    sb = paddle.cumsum(bs2, axis=1) + b0.unsqueeze(1)

    b2 = sb[:, -1] - s[:, -1] * paddle.take_along_axis(p, indp[:, 0:1], axis=1).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    lb = paddle.zeros([c2.sum()], dtype='float32')
    ub = paddle.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = paddle.floor((lb + ub) / 2)

        counter2 = counter4.astype('int64').unsqueeze(1)
        indcurr = paddle.take_along_axis(indp_, indp_.shape[1] - 1 - counter2, axis=1)
        b2 = (paddle.take_along_axis(sb_, counter2, axis=1) - paddle.take_along_axis(s_, counter2, axis=1) * paddle.take_along_axis(p_, indcurr, axis=1)).squeeze(1)
        c = b_ - b2 > 0

        lb = paddle.where(c, counter4, lb)
        ub = paddle.where(c, ub, counter4)

    lb = lb.astype('int64')

    if c_l.any():
        lmbd_opt = paddle.clip((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), min=0).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = paddle.clip((b[c2] - paddle.take_along_axis(sb[c2], lb.unsqueeze(1), axis=1).squeeze(1)) / 
                          (-paddle.take_along_axis(s[c2], lb.unsqueeze(1), axis=1).squeeze(1)), min=0).unsqueeze(-1)
    d[c2] = paddle.minimum(lmbd_opt, d[c2]) * a[c2] + paddle.maximum(-lmbd_opt, d[c2]) * (1 - a[c2])

    return d * (w != 0).astype('float32')


def projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0).astype('float32') - 1
    w = w * ind2.unsqueeze(1)
    c = c * ind2

    r = paddle.maximum(t / w, (t - 1) / w)
    r = paddle.clip(r, min=-1e12, max=1e12)
    r = paddle.where(w.abs() < 1e-8, paddle.full_like(r, 1e12), r)
    r = paddle.where(r == -1e12, r * -1, r)
    rs = paddle.sort(r, axis=1)
    indr = paddle.argsort(r, axis=1)
    rs2 = F.pad(rs[:, 1:], [0, 1])
    rs = paddle.where(rs == 1e12, paddle.zeros_like(rs), rs)
    rs2 = paddle.where(rs2 == 1e12, paddle.zeros_like(rs2), rs2)

    w3s = paddle.take_along_axis((w ** 2), indr, axis=1)
    w5 = w3s.sum(axis=1, keepdim=True)
    ws = w5 - paddle.cumsum(w3s, axis=1)
    d = -(r * w)
    d = d * (w.abs() > 1e-8).astype('float32')
    s = paddle.concat([
        -w5 * rs[:, 0:1], 
        paddle.cumsum((-rs2 + rs) * ws, axis=1) - w5 * rs[:, 0:1]
    ], axis=1)

    c4 = s[:, 0] + c < 0
    c3 = (d * w).sum(axis=1) + c > 0
    c2 = ~(c4 | c3)

    lb = paddle.zeros([c2.sum()], dtype='float32')
    ub = paddle.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = paddle.floor((lb + ub) / 2)
        counter2 = counter4.astype('int64').unsqueeze(1)
        c3 = paddle.take_along_axis(s_, counter2, axis=1).squeeze(1) + c_ > 0
        lb = paddle.where(c3, counter4, lb)
        ub = paddle.where(c3, ub, counter4)

    lb = lb.astype('int64')

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -alpha.unsqueeze(-1) * w[c4]

    if c2.any():
        alpha = (paddle.take_along_axis(s[c2], lb.unsqueeze(1), axis=1).squeeze(1) + c[c2]) / \
                paddle.take_along_axis(ws[c2], lb.unsqueeze(1), axis=1).squeeze(1) + \
                paddle.take_along_axis(rs[c2], lb.unsqueeze(1), axis=1).squeeze(1)
        alpha = paddle.where(paddle.take_along_axis(ws[c2], lb.unsqueeze(1), axis=1).squeeze(1) == 0, 
                            paddle.zeros_like(alpha), alpha)
        c5 = (alpha.unsqueeze(-1) > r[c2]).astype('float32')
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).astype('float32')


def projection_l1(points_to_project, w_hyperplane, b_hyperplane):
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0).astype('float32') - 1
    w = w * ind2.unsqueeze(1)
    c = c * ind2

    r = paddle.clip((1 / w).abs(), max=1e12)
    indr = paddle.argsort(r, axis=1)
    indr_rev = paddle.argsort(indr)

    c6 = (w < 0).astype('float32')
    d = (-t + c6) * (w != 0).astype('float32')
    ds = paddle.take_along_axis(paddle.minimum(-w * t, w * (1 - t)), indr, axis=1)
    ds2 = paddle.concat([c.unsqueeze(-1), ds], axis=1)
    s = paddle.cumsum(ds2, axis=1)

    c2 = s[:, -1] < 0

    lb = paddle.zeros([c2.sum()], dtype='float32')
    ub = paddle.full_like(lb, s.shape[1])
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_ = s[c2]
    for counter in range(nitermax):
        counter4 = paddle.floor((lb + ub) / 2)
        counter2 = counter4.astype('int64').unsqueeze(1)
        c3 = paddle.take_along_axis(s_, counter2, axis=1).squeeze(1) > 0
        lb = paddle.where(c3, counter4, lb)
        ub = paddle.where(c3, ub, counter4)

    lb2 = lb.astype('int64')

    if c2.any():
        indr_selected = paddle.take_along_axis(indr[c2], lb2.unsqueeze(1), axis=1).squeeze(1)
        u = paddle.arange(0, w.shape[0], dtype='int64').unsqueeze(1)
        u2 = paddle.arange(0, w.shape[1], dtype='float32').unsqueeze(0)
        
        # Calculate alpha values
        alpha = -paddle.take_along_axis(s[c2], lb2.unsqueeze(1), axis=1).squeeze(1)
        # Need to gather w values at the selected indices
        rows = paddle.arange(c2.sum(), dtype='int64')
        w_selected = paddle.zeros([c2.sum()], dtype='float32')
        for i in range(c2.sum()):
            w_selected[i] = w[paddle.nonzero(c2)[i], indr_selected[i]]
        alpha = alpha / w_selected
        
        c5 = u2 < lb.unsqueeze(-1)
        # Simplified indexing for u3
        indr_rev_selected = paddle.take_along_axis(indr_rev[c2], indr_selected.unsqueeze(1), axis=1).squeeze(1)
        u3 = paddle.zeros([c2.sum(), w.shape[1]], dtype='bool')
        for i in range(c2.sum()):
            if i < c5.shape[0]:
                u3[i] = c5[i, indr_rev_selected[i]]
        d[c2] = d[c2] * u3.astype('float32')
        
        # Update d at specific indices
        c2_indices = paddle.nonzero(c2).squeeze(-1)
        for i, idx in enumerate(c2_indices.numpy()):
            d[idx, indr_selected[i]] = alpha[i]

    return d * (w.abs() > 1e-8).astype('float32')