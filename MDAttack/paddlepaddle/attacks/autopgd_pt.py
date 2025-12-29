import numpy as np
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class APGDAttack():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        # device parameter is kept for compatibility but not used in PaddlePaddle
        self.device = device

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = paddle.sort(x, axis=1, descending=True), paddle.argsort(x, axis=1, descending=True)
        ind = (ind_sorted[:, 0] == y).astype('float32')

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, 1] * ind - x_sorted[:, 0] * (1. - ind)) / (x_sorted[:, 0] - x_sorted[:, 2] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * paddle.rand(x.shape, dtype='float32') - 1
            x_adv = x.detach() + self.eps * paddle.ones([x.shape[0], 1, 1, 1], dtype='float32') * t / (t.reshape([t.shape[0], -1]).abs().max(axis=1, keepdim=True).reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = paddle.randn(x.shape, dtype='float32')
            x_adv = x.detach() + self.eps * paddle.ones([x.shape[0], 1, 1, 1], dtype='float32') * t / ((t ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12)
        x_adv = paddle.clip(x_adv, 0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = paddle.zeros([self.n_iter, x.shape[0]], dtype='float32')
        loss_best_steps = paddle.zeros([self.n_iter + 1, x.shape[0]], dtype='float32')
        acc_steps = paddle.zeros_like(loss_best_steps)

        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss')

        x_adv.stop_gradient = False
        grad = paddle.zeros_like(x)
        for _ in range(self.eot_iter):
            logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()
            
            grad += paddle.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().argmax(axis=1) == y
        acc_steps[0] = acc.astype('float32')
        loss_best = loss_indiv.detach().clone()

        step_size = self.eps * paddle.ones([x.shape[0], 1, 1, 1], dtype='float32') * paddle.to_tensor([2.0], dtype='float32').reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            with paddle.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * paddle.sign(grad)
                    x_adv_1 = paddle.clip(paddle.minimum(paddle.maximum(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = paddle.clip(paddle.minimum(paddle.maximum(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = paddle.clip(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12) * paddle.minimum(
                        self.eps * paddle.ones(x.shape, dtype='float32'), ((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = paddle.clip(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12) * paddle.minimum(
                        self.eps * paddle.ones(x.shape, dtype='float32'), ((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.stop_gradient = False
            grad = paddle.zeros_like(x)
            for _ in range(self.eot_iter):
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
                
                grad += paddle.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = logits.detach().argmax(axis=1) == y
            acc = paddle.minimum(acc, pred)
            acc_steps[i + 1] = acc.astype('float32')
            # Handle indexing with nonzero
            pred_false_idx = paddle.nonzero(~pred).squeeze(axis=-1)
            if pred_false_idx.numel() > 0:
                x_best_adv[pred_false_idx] = x_adv[pred_false_idx].clone()
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum().item()))

            ### check step size
            with paddle.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1
              ind = paddle.nonzero(y1 > loss_best).squeeze(axis=-1)
              if ind.numel() > 0:
                  x_best[ind] = x_adv[ind].clone()
                  grad_best[ind] = grad[ind].clone()
                  loss_best[ind] = y1[ind]
              loss_best_steps[i + 1] = loss_best

              counter3 += 1

              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().numpy(), i, k, loss_best.detach().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.numpy() >= loss_best.numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()

                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()

                      fl_oscillation = np.where(fl_oscillation)

                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.model(x).argmax(axis=1) == y
        loss = -1e10 * paddle.ones_like(acc, dtype='float32')
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.astype('float32').mean().item()))
        startt = time.time()

        if not best_loss:
            paddle.seed(self.seed)

            if not cheap:
                raise ValueError('not implemented yet')

            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = paddle.nonzero(acc).squeeze(axis=-1)
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = paddle.nonzero(~acc_curr).squeeze(axis=-1)
                        if ind_curr.numel() > 0:
                            # Update acc and adv for successful attacks
                            actual_indices = ind_to_fool[ind_curr]
                            acc[actual_indices] = False
                            adv[actual_indices] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.astype('float32').mean().item(), time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = paddle.ones([x.shape[0]], dtype='float32') * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = paddle.nonzero(loss_curr > loss_best).squeeze(axis=-1)
                if ind_curr.numel() > 0:
                    adv_best[ind_curr] = best_curr[ind_curr].clone()
                    loss_best[ind_curr] = loss_curr[ind_curr].clone()

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum().item()))

            return adv_best


class APGDAttack_targeted():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, eot_iter=1, rho=.75, verbose=False, device='cuda',
                 n_target_classes=9):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        # device parameter is kept for compatibility but not used in PaddlePaddle
        self.device = device
        self.n_target_classes = n_target_classes

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = paddle.sort(x, axis=1, descending=True), paddle.argsort(x, axis=1, descending=True)

        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, 0] - .5 * x_sorted[:, 2] - .5 * x_sorted[:, 3] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * paddle.rand(x.shape, dtype='float32') - 1
            x_adv = x.detach() + self.eps * paddle.ones([x.shape[0], 1, 1, 1], dtype='float32') * t / (t.reshape([t.shape[0], -1]).abs().max(axis=1, keepdim=True).reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = paddle.randn(x.shape, dtype='float32')
            x_adv = x.detach() + self.eps * paddle.ones([x.shape[0], 1, 1, 1], dtype='float32') * t / ((t ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12)
        x_adv = paddle.clip(x_adv, 0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = paddle.zeros([self.n_iter, x.shape[0]], dtype='float32')
        loss_best_steps = paddle.zeros([self.n_iter + 1, x.shape[0]], dtype='float32')
        acc_steps = paddle.zeros_like(loss_best_steps)

        output = self.model(x)
        y_target = paddle.sort(output, axis=1, descending=True)[1][:, self.target_class - 1]

        x_adv.stop_gradient = False
        grad = paddle.zeros_like(x)
        for _ in range(self.eot_iter):
            logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
            loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
            loss = loss_indiv.sum()
            
            grad += paddle.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().argmax(axis=1) == y
        acc_steps[0] = acc.astype('float32')
        loss_best = loss_indiv.detach().clone()

        step_size = self.eps * paddle.ones([x.shape[0], 1, 1, 1], dtype='float32') * paddle.to_tensor([2.0], dtype='float32').reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            with paddle.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * paddle.sign(grad)
                    x_adv_1 = paddle.clip(paddle.minimum(paddle.maximum(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = paddle.clip(paddle.minimum(paddle.maximum(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = paddle.clip(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12) * paddle.minimum(
                        self.eps * paddle.ones(x.shape, dtype='float32'), ((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = paddle.clip(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12) * paddle.minimum(
                        self.eps * paddle.ones(x.shape, dtype='float32'), ((x_adv_1 - x) ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.stop_gradient = False
            grad = paddle.zeros_like(x)
            for _ in range(self.eot_iter):
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()
                
                grad += paddle.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = logits.detach().argmax(axis=1) == y
            acc = paddle.minimum(acc, pred)
            acc_steps[i + 1] = acc.astype('float32')
            # Handle indexing with nonzero
            pred_false_idx = paddle.nonzero(~pred).squeeze(axis=-1)
            if pred_false_idx.numel() > 0:
                x_best_adv[pred_false_idx] = x_adv[pred_false_idx].clone()
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum().item()))

            ### check step size
            with paddle.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1
              ind = paddle.nonzero(y1 > loss_best).squeeze(axis=-1)
              if ind.numel() > 0:
                  x_best[ind] = x_adv[ind].clone()
                  grad_best[ind] = grad[ind].clone()
                  loss_best[ind] = y1[ind]
              loss_best_steps[i + 1] = loss_best

              counter3 += 1

              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().numpy(), i, k, loss_best.detach().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.numpy() >= loss_best.numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()

                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()

                      fl_oscillation = np.where(fl_oscillation)

                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.model(x).argmax(axis=1) == y
        loss = -1e10 * paddle.ones_like(acc, dtype='float32')
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.astype('float32').mean().item()))
        startt = time.time()

        paddle.seed(self.seed)

        if not cheap:
            raise ValueError('not implemented yet')

        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = paddle.nonzero(acc).squeeze(axis=-1)
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = paddle.nonzero(~acc_curr).squeeze(axis=-1)
                        if ind_curr.numel() > 0:
                            # Update acc and adv for successful attacks
                            actual_indices = ind_to_fool[ind_curr]
                            acc[actual_indices] = False
                            adv[actual_indices] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, acc.astype('float32').mean().item(), self.eps, time.time() - startt))

        return adv