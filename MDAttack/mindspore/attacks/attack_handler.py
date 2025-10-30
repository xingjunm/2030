import torch
import mindspore as ms
import mindspore.ops as ops
import numpy as np
from . import PGD, MD, autopgd_pt, fab_pt
from .utils import adv_check_and_update

# Set MindSpore context - using device exemption #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class Attacker():
    def __init__(self, model, epsilon=8./255., v_min=0., v_max=1.,
                 num_classes=10, data_loader=None, logger=None,
                 version='MD', verbose=True):
        """
        Attack handler that coordinates multiple attack methods.
        
        Note: The model parameter is expected to be a PyTorch model, as per the 
        project plan where defenses remain in PyTorch. The attack methods have
        been converted to MindSpore, so we'll handle conversions at the boundaries.
        """
        self.model = model  # This is a PyTorch model
        self.epsilon = epsilon
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.logger = logger
        self.verbose = verbose
        
        # Create wrapper for PyTorch model to work with MindSpore attacks
        wrapped_model = self._create_model_wrapper(model)
        
        # Initialize attack methods with the wrapped model
        self.md = MD.MDAttack(wrapped_model, epsilon, num_steps=50, step_size=2/255,
                              v_min=v_min, v_max=v_max, change_point=20,
                              first_step_size=16./255., seed=0, norm='Linf',
                              num_classes=num_classes, use_odi=False)
        self.md_dlr = MD.MDAttack(wrapped_model, epsilon, num_steps=50, step_size=2/255,
                                  v_min=v_min, v_max=v_max, change_point=20,
                                  first_step_size=16./255., seed=0, norm='Linf',
                                  num_classes=num_classes, use_odi=False,
                                  use_dlr=True)
        self.mdmt = MD.MDMTAttack(wrapped_model, epsilon, num_steps=50, step_size=2/255,
                                  v_min=v_min, v_max=v_max, change_point=20,
                                  first_step_size=16./255., seed=0, norm='Linf',
                                  num_classes=num_classes, use_odi=False)
        self.mdmt_plus = MD.MDMTAttack(wrapped_model, epsilon, num_steps=50,
                                       num_random_starts=10, step_size=2/255,
                                       v_min=v_min, v_max=v_max, change_point=20,
                                       first_step_size=16./255., seed=0,
                                       norm='Linf', num_classes=num_classes)
        self.pgd = PGD.PGDAttack(wrapped_model, epsilon, num_steps=20, step_size=0.8/255.,
                                 num_restarts=5, v_min=v_min, v_max=v_max,
                                 num_classes=num_classes, random_start=True,
                                 type='PGD', use_odi=False)
        self.mt = PGD.MTPGDAttack(wrapped_model, epsilon, num_steps=20, step_size=0.8/255,
                                  num_restarts=5, v_min=v_min, v_max=v_max,
                                  num_classes=num_classes, random_start=True,
                                  use_odi=False)
        self.odi = PGD.PGDAttack(wrapped_model, epsilon, num_steps=20, step_size=0.8/255.,
                                 num_restarts=5, v_min=v_min, v_max=v_max,
                                 num_classes=num_classes, random_start=True,
                                 type='PGD', use_odi=True)
        self.cw = PGD.PGDAttack(wrapped_model, epsilon, num_steps=100, step_size=0.8/255,
                                num_restarts=1, v_min=v_min, v_max=v_max,
                                num_classes=num_classes, random_start=True,
                                type='CW', use_odi=False)
        # Note: autopgd and fab don't use device parameter in MindSpore version
        self.apgd = autopgd_pt.APGDAttack(wrapped_model, eps=epsilon)
        self.apgd_mt = autopgd_pt.APGDAttack_targeted(wrapped_model, eps=epsilon)
        self.fab = fab_pt.FABAttack_PT(wrapped_model, n_restarts=5, n_iter=100,
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
            # Using raise with string to match original behavior (exemption #1)
            raise Exception('Unknown')
    
    def _create_model_wrapper(self, pytorch_model):
        """
        Create a wrapper that allows PyTorch models to work with MindSpore attacks.
        This wrapper converts MindSpore tensors to PyTorch tensors for inference,
        and converts the output back to MindSpore tensors.
        """
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                # Determine device from model parameters
                self.device = next(model.parameters()).device
                
            def __call__(self, x_ms):
                # Convert MindSpore tensor to PyTorch tensor
                x_np = x_ms.asnumpy()
                x_torch = torch.from_numpy(x_np)
                
                # Move to same device as model
                x_torch = x_torch.to(self.device)
                
                # Run model inference
                with torch.no_grad():
                    output = self.model(x_torch)
                    
                # Handle list outputs (some models return multiple outputs)
                if isinstance(output, list):
                    output = output[-1]
                
                # Convert back to MindSpore tensor
                output_np = output.cpu().numpy()
                output_ms = ms.Tensor(output_np, ms.float32)
                
                return output_ms
        
        return ModelWrapper(pytorch_model)

    def evaluate(self):
        clean_count = 0
        adv_count = 0
        total = 0

        for images, labels in self.data_loader:
            # Data loader provides PyTorch tensors, convert to numpy first
            if isinstance(images, torch.Tensor):
                images_np = images.numpy()
                labels_np = labels.numpy()
            else:
                images_np = images
                labels_np = labels
                
            # Convert to MindSpore tensors
            images_ms = ms.Tensor(images_np, ms.float32)
            labels_ms = ms.Tensor(labels_np, ms.int32)  # MindSpore CrossEntropyLoss requires int32
            
            # Move PyTorch tensors to same device as model
            device = next(self.model.parameters()).device
            images_torch = torch.from_numpy(images_np).to(device)
            labels_torch = torch.from_numpy(labels_np).to(device)
            
            nc = ops.zeros_like(labels_ms)
            total += labels_ms.shape[0]

            # Check Clean Acc using PyTorch model
            with torch.no_grad():
                clean_logits = self.model(images_torch)
                if isinstance(clean_logits, list):
                    clean_logits = clean_logits[-1]
            clean_pred = clean_logits.data.max(1)[1].detach()
            clean_correct = (clean_pred == labels_torch).sum().item()
            clean_count += clean_correct
            
            # Convert clean logits to MindSpore for adv_check_and_update
            clean_logits_np = clean_logits.cpu().numpy()
            clean_logits_ms = ms.Tensor(clean_logits_np, ms.float32)

            # Build x_adv in MindSpore
            x_adv = images_ms.copy()
            x_adv_targets = images_ms.copy()
            x_adv, nc = adv_check_and_update(x_adv_targets, clean_logits_ms,
                                             labels_ms, nc, x_adv)

            # All attacks and update x_adv
            for a in self.attacks_to_run:
                # Attack methods work with MindSpore tensors
                x_p = a.perturb(images_ms, labels_ms)
                
                # Convert to PyTorch for model inference
                x_p_np = x_p.asnumpy()
                x_p_torch = torch.from_numpy(x_p_np).to(device)
                    
                with torch.no_grad():
                    adv_logits = self.model(x_p_torch)
                    if isinstance(adv_logits, list):
                        adv_logits = adv_logits[-1]
                
                # Convert back to MindSpore
                adv_logits_np = adv_logits.cpu().numpy()
                adv_logits_ms = ms.Tensor(adv_logits_np, ms.float32)
                
                x_adv, nc = adv_check_and_update(x_p, adv_logits_ms, labels_ms,
                                                 nc, x_adv)
            
            # Robust Acc - convert final x_adv to PyTorch
            x_adv_np = x_adv.asnumpy()
            x_adv_torch = torch.from_numpy(x_adv_np).to(device)
                
            with torch.no_grad():
                adv_logits = self.model(x_adv_torch)
                if isinstance(adv_logits, list):
                    adv_logits = adv_logits[-1]

            adv_pred = adv_logits.data.max(1)[1].detach()
            adv_correct = (adv_pred == labels_torch).sum().item()
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