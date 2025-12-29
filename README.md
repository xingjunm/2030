# 2030 Adversarial & Backdoor Defense Suite

**Supports:**
- AdvDet — [PyTorch](AdvDet/torch) · [TensorFlow](AdvDet/tensorflow) · [PaddlePaddle](AdvDet/paddle) · [MindSpore](AdvDet/mindspore)
- Cognitive Distillation — [PyTorch](CognitiveDistillation/pytorch) · [TensorFlow](CognitiveDistillation/tensorflow) · [PaddlePaddle](CognitiveDistillation/paddlepaddle) · [MindSpore](CognitiveDistillation/mindspore)
- MD Attack — [PyTorch](MDAttack/pytorch) · [TensorFlow](MDAttack/tensorflow) · [PaddlePaddle](MDAttack/paddlepaddle) · [MindSpore](MDAttack/mindspore)
- PrivDet — [PyTorch](PrivDet/pytorch) · [TensorFlow](PrivDet/tensorflow) · [PaddlePaddle](PrivDet/paddlepaddle) · [MindSpore](PrivDet/mindspore)

---

This repository aggregates several research implementations on adversarial attack detection and defense, covering AdvDet, Cognitive Distillation, MD Attack, and the newly added PrivDet project. Each submodule offers matched implementations for PyTorch, TensorFlow, PaddlePaddle, and MindSpore so you can reproduce results and run comparisons across different hardware or deployment environments.

## Projects
- **AdvDet** (`AdvDet/`): Adversarial contrastive prompt tuning for detecting query-based adversarial attacks. Framework entry points: [PyTorch](AdvDet/torch) · [TensorFlow](AdvDet/tensorflow) · [PaddlePaddle](AdvDet/paddle) · [MindSpore](AdvDet/mindspore)
- **Cognitive Distillation** (`CognitiveDistillation/`): Distilling cognitive backdoor patterns to enhance backdoor sample detection. Framework entry points: [PyTorch](CognitiveDistillation/pytorch) · [TensorFlow](CognitiveDistillation/tensorflow) · [PaddlePaddle](CognitiveDistillation/paddlepaddle) · [MindSpore](CognitiveDistillation/mindspore)
- **MD Attack** (`MDAttack/`): Investigating imbalanced gradients that cause overestimated robustness with multiple attack/defense pairings. Framework entry points: [PyTorch](MDAttack/pytorch) · [TensorFlow](MDAttack/tensorflow) · [PaddlePaddle](MDAttack/paddlepaddle) · [MindSpore](MDAttack/mindspore)
- **PrivDet** (`PrivDet/`): Private dataset origin detection to differentiate images from distinct distributions (e.g., COCO vs. CIFAR-10). Framework entry points: [PyTorch](PrivDet/pytorch) · [TensorFlow](PrivDet/tensorflow) · [PaddlePaddle](PrivDet/paddlepaddle) · [MindSpore](PrivDet/mindspore)
