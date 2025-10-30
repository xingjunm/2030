import math

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    t, w, b = points_to_project, w_hyperplane.copy(), b_hyperplane.copy()

    sign = 2 * ((w * t).sum(1) - b >= 0) - 1
    w = w * ops.expand_dims(sign, 1)
    b = b * sign

    a = ops.cast(w < 0, ms.float32)
    d = (a - t) * ops.cast(w != 0, ms.float32)

    p = a - t * (2 * a - 1)
    indp = ops.argsort(p, axis=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)

    indp2 = ops.ReverseV2(axis=(1,))(indp)
    ws = ops.GatherD()(w, 1, indp2)
    bs2 = - ws * ops.GatherD()(d, 1, indp2)

    s = ops.cumsum(ws.abs(), axis=1)
    sb = ops.cumsum(bs2, axis=1) + ops.expand_dims(b0, 1)

    b2 = sb[:, -1] - s[:, -1] * ops.GatherD()(p, 1, indp[:, 0:1]).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    c2_count = int(c2.sum().asnumpy())
    lb = ops.zeros(c2_count, ms.float32)
    ub = ops.full((c2_count,), float(w.shape[1] - 1))
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = ops.floor((lb + ub) / 2)

        counter2 = ops.cast(counter4, ms.int32).reshape(-1, 1)
        indcurr = ops.GatherD()(indp_, 1, indp_.shape[1] - 1 - counter2)
        b2 = (ops.GatherD()(sb_, 1, counter2) - 
              ops.GatherD()(s_, 1, counter2) * 
              ops.GatherD()(p_, 1, indcurr)).squeeze(1)
        c = b_ - b2 > 0

        lb = ops.select(c, counter4, lb)
        ub = ops.select(c, ub, counter4)

    lb = ops.cast(lb, ms.int32)

    if c_l.any():
        lmbd_opt = ops.maximum((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), 0).reshape(-1, 1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = ops.maximum((b[c2] - ops.GatherD()(sb[c2], 1, lb.reshape(-1, 1)).squeeze(1)) / 
                          (-ops.GatherD()(s[c2], 1, lb.reshape(-1, 1)).squeeze(1)), 0).reshape(-1, 1)
    d[c2] = ops.minimum(lmbd_opt, d[c2]) * a[c2] + ops.maximum(-lmbd_opt, d[c2]) * (1 - a[c2])

    return d * ops.cast(w != 0, ms.float32)


def projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    t, w, b = points_to_project, w_hyperplane.copy(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1
    w = w * ops.expand_dims(ind2, 1)
    c = c * ind2

    r = ops.maximum(t / w, (t - 1) / w)
    r = ops.clip_by_value(r, -1e12, 1e12)
    r = ops.select(w.abs() < 1e-8, ops.fill(ms.float32, r.shape, 1e12), r)
    r = ops.select(r == -1e12, r * -1, r)
    rs, indr = ops.sort(r, axis=1)
    rs2 = ops.concat((rs[:, 1:], ops.zeros((rs.shape[0], 1), ms.float32)), 1)
    rs = ops.select(rs == 1e12, 0, rs)
    rs2 = ops.select(rs2 == 1e12, 0, rs2)

    w3s = ops.GatherD()(w ** 2, 1, indr)
    w5 = w3s.sum(axis=1, keepdims=True)
    ws = w5 - ops.cumsum(w3s, axis=1)
    d = -(r * w)
    d = d * ops.cast(w.abs() > 1e-8, ms.float32)
    s = ops.concat((-w5 * rs[:, 0:1], ops.cumsum((-rs2 + rs) * ws, axis=1) - w5 * rs[:, 0:1]), 1)

    c4 = s[:, 0] + c < 0
    c3 = (d * w).sum(axis=1) + c > 0
    c2 = ~(c4 | c3)

    c2_count = int(c2.sum().asnumpy())
    lb = ops.zeros(c2_count, ms.float32)
    ub = ops.full((c2_count,), float(w.shape[1] - 1))
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = ops.floor((lb + ub) / 2)
        counter2 = ops.cast(counter4, ms.int32).reshape(-1, 1)
        c3 = ops.GatherD()(s_, 1, counter2).squeeze(1) + c_ > 0
        lb = ops.select(c3, counter4, lb)
        ub = ops.select(c3, ub, counter4)

    lb = ops.cast(lb, ms.int32)

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -ops.expand_dims(alpha, -1) * w[c4]

    if c2.any():
        alpha = (ops.GatherD()(s[c2], 1, lb.reshape(-1, 1)).squeeze(1) + c[c2]) / \
                ops.GatherD()(ws[c2], 1, lb.reshape(-1, 1)).squeeze(1) + \
                ops.GatherD()(rs[c2], 1, lb.reshape(-1, 1)).squeeze(1)
        alpha = ops.select(ops.GatherD()(ws[c2], 1, lb.reshape(-1, 1)).squeeze(1) == 0, 0, alpha)
        c5 = ops.cast(ops.expand_dims(alpha, -1) > r[c2], ms.float32)
        d[c2] = d[c2] * c5 - ops.expand_dims(alpha, -1) * w[c2] * (1 - c5)

    return d * ops.cast(w.abs() > 1e-8, ms.float32)


def projection_l1(points_to_project, w_hyperplane, b_hyperplane):
    t, w, b = points_to_project, w_hyperplane.copy(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1
    w = w * ops.expand_dims(ind2, 1)
    c = c * ind2

    r = ops.minimum((1 / w).abs(), 1e12)
    indr = ops.argsort(r, axis=1)
    indr_rev = ops.argsort(indr)

    c6 = ops.cast(w < 0, ms.float32)
    d = (-t + c6) * ops.cast(w != 0, ms.float32)
    ds = ops.GatherD()(ops.minimum(-w * t, w * (1 - t)), 1, indr)
    ds2 = ops.concat((ops.expand_dims(c, -1), ds), 1)
    s = ops.cumsum(ds2, axis=1)

    c2 = s[:, -1] < 0

    c2_count = int(c2.sum().asnumpy())
    lb = ops.zeros(c2_count, ms.float32)
    ub = ops.full((c2_count,), float(s.shape[1]))
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_ = s[c2]
    for counter in range(nitermax):
        counter4 = ops.floor((lb + ub) / 2)
        counter2 = ops.cast(counter4, ms.int32).reshape(-1, 1)
        c3 = ops.GatherD()(s_, 1, counter2).squeeze(1) > 0
        lb = ops.select(c3, counter4, lb)
        ub = ops.select(c3, ub, counter4)

    lb2 = ops.cast(lb, ms.int32)

    if c2.any():
        indr_selected = ops.GatherD()(indr[c2], 1, lb2.reshape(-1, 1)).squeeze(1)
        u = mnp.arange(0, w.shape[0]).reshape(-1, 1)
        u2 = mnp.arange(0, w.shape[1], dtype=ms.float32).reshape(1, -1)
        
        # Calculate alpha
        s_selected = ops.GatherD()(s[c2], 1, lb2.reshape(-1, 1)).squeeze(1)
        w_selected = w[c2, indr_selected]
        alpha = -s_selected / w_selected
        
        # Create mask
        c5 = u2 < ops.expand_dims(lb, -1)
        u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
        d[c2] = d[c2] * ops.cast(u3, ms.float32)
        d[c2, indr_selected] = alpha

    return d * ops.cast(w.abs() > 1e-8, ms.float32)