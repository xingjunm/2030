import os
import util
import time
import argparse
import datasets
import numpy as np
import paddle
import torch
import torch.nn as nn
# Import PaddlePaddle version of attacks
from attacks.attack_handler import Attacker

# PaddlePaddle automatically handles device placement
# No need for explicit device management (exemption #3)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch_device = torch.device('cuda')
else:
    torch_device = torch.device('cpu')

parser = argparse.ArgumentParser(description='MD Attacks')
parser.add_argument('--defence', type=str, default='RST')
parser.add_argument('--attack', type=str, default='MD')
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--datapath', type=str, default='../../datasets')
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--result_path', default='results/')
args = parser.parse_args()
args.eps = args.eps/255
logger = util.setup_logger('MD Attack')


def test(model, testloader):
    """Test function for PyTorch models (defense models remain in PyTorch)"""
    model.eval()
    total = 0
    corrects = np.zeros(5)
    with torch.no_grad():
        for data, labels in testloader:
            # Convert PaddlePaddle tensor or numpy to PyTorch tensor if needed
            if isinstance(data, paddle.Tensor):
                data = torch.from_numpy(data.numpy())
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(labels, paddle.Tensor):
                labels = torch.from_numpy(labels.numpy())
            elif isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
                
            data = data.to(torch_device)
            outputs = model(data)[-5:]
            predictions = np.array(
                [o.max(1)[1].cpu().numpy() for o in outputs])
            labels = labels.reshape(1, -1).detach().numpy()
            corrects += (predictions == labels).sum(1)
            total += labels.size
    accs = corrects / total
    return accs, total


class PyTorchToPaddleModel:
    """Wrapper class to make PyTorch models compatible with PaddlePaddle attacks"""
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()
        # Get the device of the model
        self.device = next(pytorch_model.parameters()).device
        
    def __call__(self, x):
        # Convert PaddlePaddle tensor to PyTorch tensor
        if isinstance(x, paddle.Tensor):
            x_torch = torch.from_numpy(x.numpy()).to(self.device)
        else:
            x_torch = x.to(self.device) if hasattr(x, 'to') else x
            
        # Run through PyTorch model
        with torch.no_grad():
            output = self.pytorch_model(x_torch)
            
        # Handle both single output and list of outputs
        if isinstance(output, list):
            # Convert each output to PaddlePaddle tensor
            paddle_outputs = []
            for o in output:
                o_np = o.cpu().numpy()
                paddle_outputs.append(paddle.to_tensor(o_np))
            return paddle_outputs
        else:
            # Convert single output to PaddlePaddle tensor
            output_np = output.cpu().numpy()
            return paddle.to_tensor(output_np)
    
    def eval(self):
        self.pytorch_model.eval()
        return self
        
    def parameters(self):
        return self.pytorch_model.parameters()


def main():
    util.build_dirs(args.result_path)
    data = datasets.DatasetGenerator(eval_bs=args.bs, n_workers=args.n_workers,
                                     train_path=args.datapath,
                                     test_path=args.datapath)
    _, test_loader = data.get_loader()
    
    # Dynamically import only the needed defense model from PyTorch implementation
    import sys
    sys.path.insert(0, '/root/MDAttack')
    
    # Import the specific defense module based on args.defence
    if args.defence == "RST":
        from defense import RST
        pytorch_model = RST.DefenseRST()
    elif args.defence == "UAT":
        from defense import UAT
        pytorch_model = UAT.DefenseUAT()
    elif args.defence == "TRADES":
        from defense import TRADES
        pytorch_model = TRADES.DefenseTRADES()
    elif args.defence == "MART":
        from defense import MART
        pytorch_model = MART.DefenseMART()
    elif args.defence == "MMA":
        from defense import MMA
        pytorch_model = MMA.DefenseMMA()
    elif args.defence == "BAT":
        from defense import BAT
        pytorch_model = BAT.DefenseBAT()
    elif args.defence == "ADVInterp":
        from defense import ADVInterp
        pytorch_model = ADVInterp.DefenseADVInterp()
    elif args.defence == "FeaScatter":
        from defense import FeaScatter
        pytorch_model = FeaScatter.DefenseFeaScatter()
    elif args.defence == "Sense":
        from defense import Sense
        pytorch_model = Sense.DefenseSense()
    elif args.defence == "JARN_AT":
        from defense import JARN_AT
        pytorch_model = JARN_AT.DefenseJARN_AT()
    elif args.defence == "Dynamic":
        from defense import Dynamic
        pytorch_model = Dynamic.DefenseDynamic()
    elif args.defence == "AWP":
        from defense import AWP
        pytorch_model = AWP.DefenseAWP()
    elif args.defence == "Overfitting":
        from defense import Overfitting
        pytorch_model = Overfitting.DefenseOverfitting()
    elif args.defence == "ATHE":
        from defense import ATHE
        pytorch_model = ATHE.DefenseATHE()
    elif args.defence == "PreTrain":
        from defense import PreTrain
        pytorch_model = PreTrain.DefensePreTrain()
    elif args.defence == "SAT":
        from defense import SAT
        pytorch_model = SAT.DefenseSAT()
    elif args.defence == "RobustWRN":
        from defense import RobustWRN
        pytorch_model = RobustWRN.DefenseRobustWRN()
    else:
        raise ValueError(f"Unknown defense: {args.defence}")
    
    sys.path.pop(0)
    
    # Load PyTorch defense model
    pytorch_model = pytorch_model.to(torch_device)
    pytorch_model.eval()
    
    if args.data_parallel:
        # Skip parallel processing as instructed
        raise NotImplementedError("Skip")
    
    for param in pytorch_model.parameters():
        param.requires_grad = False
    
    # Wrap PyTorch model for PaddlePaddle attacks
    model = PyTorchToPaddleModel(pytorch_model)
    
    # Prepare test data
    x_test = []
    y_test = []
    for (x, y) in test_loader:
        # Data from PaddlePaddle DataLoader could be paddle.Tensor or numpy
        if isinstance(x, paddle.Tensor):
            x_test.append(x)
        else:
            x_test.append(paddle.to_tensor(x))
        if isinstance(y, paddle.Tensor):
            y_test.append(y)
        else:
            y_test.append(paddle.to_tensor(y))
    
    x_test = paddle.concat(x_test, axis=0)
    y_test = paddle.concat(y_test, axis=0)
    
    start = time.time()
    if args.attack == 'AA':
        # AutoAttack is not yet implemented in PaddlePaddle version
        # Would need to implement a PaddlePaddle version of AutoAttack
        # For now, raise NotImplementedError
        raise NotImplementedError("AutoAttack not yet implemented for PaddlePaddle version")
    else:
        adversary = Attacker(model, epsilon=args.eps, num_classes=10,
                           data_loader=test_loader, logger=logger,
                           version=args.attack)
        rs = adversary.evaluate()
        clean_accuracy, robust_accuracy = rs
    
    end = time.time()
    cost = end - start
    payload = {
        'clean_acc': clean_accuracy,
        'adv_acc': robust_accuracy,
        'cost': cost
    }
    print(robust_accuracy)
    filename = '%s_%s.json' % (args.defence, args.attack)
    filename = os.path.join(args.result_path, filename)
    util.save_json(payload, filename)
    return


if __name__ == '__main__':
    main()