import tensorflow as tf
import numpy as np
import torch  # Still needed since model is PyTorch
from . import PGD, MD, autopgd_pt, fab_pt
from .utils import adv_check_and_update

# Per exemption #3, we don't need explicit device handling in TensorFlow
# TensorFlow will automatically use GPU if available


class Attacker():
    def __init__(self, model, epsilon=8./255., v_min=0., v_max=1.,
                 num_classes=10, data_loader=None, logger=None,
                 version='MD', verbose=True):
        self.model = model  # This is a PyTorch model from defense
        self.epsilon = epsilon
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.logger = logger
        self.verbose = verbose
        
        # Initialize attack methods with TensorFlow implementations
        self.md = MD.MDAttack(model, epsilon, num_steps=50, step_size=2/255,
                              v_min=v_min, v_max=v_max, change_point=20,
                              first_step_size=16./255., seed=0, norm='Linf',
                              num_classes=num_classes, use_odi=False)
        self.md_dlr = MD.MDAttack(model, epsilon, num_steps=50, step_size=2/255,
                                  v_min=v_min, v_max=v_max, change_point=20,
                                  first_step_size=16./255., seed=0, norm='Linf',
                                  num_classes=num_classes, use_odi=False,
                                  use_dlr=True)
        self.mdmt = MD.MDMTAttack(model, epsilon, num_steps=50, step_size=2/255,
                                  v_min=v_min, v_max=v_max, change_point=20,
                                  first_step_size=16./255., seed=0, norm='Linf',
                                  num_classes=num_classes, use_odi=False)
        self.mdmt_plus = MD.MDMTAttack(model, epsilon, num_steps=50,
                                       num_random_starts=10, step_size=2/255,
                                       v_min=v_min, v_max=v_max, change_point=20,
                                       first_step_size=16./255., seed=0,
                                       norm='Linf', num_classes=num_classes)
        self.pgd = PGD.PGDAttack(model, epsilon, num_steps=20, step_size=0.8/255.,
                                 num_restarts=5, v_min=v_min, v_max=v_max,
                                 num_classes=num_classes, random_start=True,
                                 type='PGD', use_odi=False)
        self.mt = PGD.MTPGDAttack(model, epsilon, num_steps=20, step_size=0.8/255,
                                  num_restarts=5, v_min=v_min, v_max=v_max,
                                  num_classes=num_classes, random_start=True,
                                  use_odi=False)
        self.odi = PGD.PGDAttack(model, epsilon, num_steps=20, step_size=0.8/255.,
                                 num_restarts=5, v_min=v_min, v_max=v_max,
                                 num_classes=num_classes, random_start=True,
                                 type='PGD', use_odi=True)
        self.cw = PGD.PGDAttack(model, epsilon, num_steps=100, step_size=0.8/255,
                                num_restarts=1, v_min=v_min, v_max=v_max,
                                num_classes=num_classes, random_start=True,
                                type='CW', use_odi=False)
        # autopgd_pt and fab_pt don't take device parameter in TensorFlow version
        self.apgd = autopgd_pt.APGDAttack(model, eps=epsilon)
        self.apgd_mt = autopgd_pt.APGDAttack_targeted(model, eps=epsilon)
        self.fab = fab_pt.FABAttack_PT(model, n_restarts=5, n_iter=100,
                                       eps=epsilon, seed=0,
                                       verbose=False)
        self.fab.targeted = True
        self.fab.n_restarts = 1

        self.attacks_to_run = []

        if version == 'MD':
            self.attacks_to_run = [self.md]
        elif 'MD_first_stage_step_search' in version:
            if '_05' in version:
                self.md.change_point = 5
            else:
                change_point = int(version[-2:])
                self.md.change_point = change_point
            self.attacks_to_run = [self.md]
        elif 'MD_first_stage_initial_step_size_search' in version:
            eps = int(version[-2:])
            self.md.initial_step_size = eps/255
            self.attacks_to_run = [self.md]
        elif version == 'MD_DLR':
            self.attacks_to_run = [self.md_dlr]
        elif version == 'MDMT':
            self.attacks_to_run = [self.mdmt]
        elif version == 'PGD':
            self.attacks_to_run = [self.pgd]
        elif version == 'MT':
            self.attacks_to_run = [self.mt]
        elif version == 'DLR':
            self.apgd.loss = 'dlr'
            self.attacks_to_run = [self.apgd]
        elif version == 'DLRMT':
            self.apgd_mt.loss = 'dlr'
            self.attacks_to_run = [self.apgd_mt]
        elif version == 'ODI':
            self.attacks_to_run = [self.odi]
        elif version == 'CW':
            self.attacks_to_run = [self.cw]
        elif version == 'MDE':
            self.apgd.loss = 'ce'
            self.attacks_to_run = [self.apgd, self.md, self.mdmt, self.fab]
        elif version == 'MDMT+':
            self.attacks_to_run = [self.mdmt_plus]
        else:
            raise ValueError('Unknown')  # Fixed to use ValueError instead of bare raise

    def evaluate(self):
        clean_count = 0
        adv_count = 0
        total = 0

        for images, labels in self.data_loader:
            # Data comes from PyTorch dataloader as torch tensors
            # Convert to numpy for processing
            images_np = images.numpy()
            labels_np = labels.numpy()
            
            # Convert to TensorFlow tensors
            images_tf = tf.convert_to_tensor(images_np, dtype=tf.float32)
            labels_tf = tf.convert_to_tensor(labels_np, dtype=tf.int64)
            
            nc = tf.zeros_like(labels_tf, dtype=tf.int64)
            total += labels_tf.shape[0]

            # Check Clean Acc
            # Model is PyTorch, so we need to use torch tensors
            with torch.no_grad():
                clean_logits = self.model(images.cuda() if torch.cuda.is_available() else images)
                if isinstance(clean_logits, list):
                    clean_logits = clean_logits[-1]
            
            # Convert PyTorch output to TensorFlow for processing
            clean_logits_np = clean_logits.cpu().numpy()
            clean_logits_tf = tf.convert_to_tensor(clean_logits_np, dtype=tf.float32)
            
            clean_pred = tf.argmax(clean_logits_tf, axis=1)
            clean_correct = tf.reduce_sum(tf.cast(tf.equal(clean_pred, labels_tf), tf.int32)).numpy()
            clean_count += clean_correct

            # Build x_adv
            x_adv = tf.identity(images_tf)
            x_adv_targets = tf.identity(images_tf)
            x_adv, nc = adv_check_and_update(x_adv_targets, clean_logits_tf,
                                             labels_tf, nc, x_adv)

            # All attacks and update x_adv
            for a in self.attacks_to_run:
                # Attack expects TensorFlow tensors and returns TensorFlow tensors
                x_p = a.perturb(images_tf, labels_tf)
                
                # Convert to PyTorch for model evaluation
                x_p_np = x_p.numpy()
                x_p_torch = torch.from_numpy(x_p_np).float()
                
                with torch.no_grad():
                    adv_logits = self.model(x_p_torch.cuda() if torch.cuda.is_available() else x_p_torch)
                    if isinstance(adv_logits, list):
                        adv_logits = adv_logits[-1]
                
                # Convert back to TensorFlow
                adv_logits_np = adv_logits.cpu().numpy()
                adv_logits_tf = tf.convert_to_tensor(adv_logits_np, dtype=tf.float32)
                
                x_adv, nc = adv_check_and_update(x_p, adv_logits_tf, labels_tf,
                                                 nc, x_adv)
            
            # Robust Acc
            # Convert final x_adv to PyTorch for evaluation
            x_adv_np = x_adv.numpy()
            x_adv_torch = torch.from_numpy(x_adv_np).float()
            
            with torch.no_grad():
                adv_logits = self.model(x_adv_torch.cuda() if torch.cuda.is_available() else x_adv_torch)
                if isinstance(adv_logits, list):
                    adv_logits = adv_logits[-1]

            # Convert to TensorFlow for final prediction
            adv_logits_np = adv_logits.cpu().numpy()
            adv_logits_tf = tf.convert_to_tensor(adv_logits_np, dtype=tf.float32)
            
            adv_pred = tf.argmax(adv_logits_tf, axis=1)
            adv_correct = tf.reduce_sum(tf.cast(tf.equal(adv_pred, labels_tf), tf.int32)).numpy()
            adv_count += adv_correct

            # Log
            if self.verbose:
                rs = (clean_count, total, clean_count * 100 / total,
                      adv_count, total, adv_count * 100 / total)
                payload = (('Clean: %d/%d Clean Acc: %.2f Adv: %d/%d '
                           + 'Adv_Acc: %.2f') % rs)
                self.logger.info('\033[33m'+payload+'\033[0m')

        clean_acc = clean_count * 100 / total
        adv_acc = adv_count * 100 / total
        return clean_acc, adv_acc