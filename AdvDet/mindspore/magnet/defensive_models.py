import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset as ds
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE)

class Upsample2x(nn.Cell):
    """Custom upsample layer that doubles the spatial dimensions"""
    def __init__(self):
        super().__init__()
        
    def construct(self, x):
        # Use ResizeNearestNeighbor to double the size
        n, c, h, w = x.shape
        resize_op = ops.ResizeNearestNeighbor((h * 2, w * 2))
        return resize_op(x)

class DenoisingAutoEncoder_1():
    def __init__(self, img_shape=(1,28,28)):
        self.img_shape = img_shape
    
        # Create a custom forward method to debug shapes
        self.conv1 = nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1, pad_mode='pad')
        self.sig1 = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, pad_mode='pad')
        self.sig2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, pad_mode='pad')
        self.sig3 = nn.Sigmoid()
        self.upsample = Upsample2x()
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, pad_mode='pad')
        self.sig4 = nn.Sigmoid()
        self.conv5 = nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1, pad_mode='pad')
        
        self.model = nn.SequentialCell(
                self.conv1, self.sig1, self.pool,
                self.conv2, self.sig2,
                self.conv3, self.sig3,
                self.upsample,
                self.conv4, self.sig4,
                self.conv5
                )
    
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=0.001)
        criterion = nn.MSELoss()
        
        # Create training step function
        def forward_fn(data_train, noisy_data):
            output = self.model(noisy_data)
            loss = criterion(output, data_train)
            return loss
        
        # Get gradients function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
        
        def train_step(data_train, noisy_data):
            loss, grads = grad_fn(data_train, noisy_data)
            optimizer(grads)
            return loss
            
        for epoch in range(num_epochs):
            self.model.set_train()
            running_loss = 0.0
            for batch_data in data:
                data_train = Tensor(batch_data[0], dtype=mindspore.float32)
                data_label = batch_data[1]
                
                noise_shape = data_train.shape
                # RISK_INFO: [随机性处理差异] - 'np.random.randn' 与 'torch.randn_like' 的随机数生成器实现和种子管理可能不同，可能导致在相同种子下产生不同的随机噪声序列。
                noise = v_noise * Tensor(np.random.randn(*noise_shape), dtype=mindspore.float32)
                noisy_data = ops.clip_by_value(data_train + noise, clip_value_min=min, clip_value_max=max)
                
                loss = train_step(data_train, noisy_data)
                running_loss += loss.asnumpy()
                
            if epoch % 10 == 0:
                # Get the number of batches
                num_batches = data.get_dataset_size() // data.get_batch_size()
                if data.get_dataset_size() % data.get_batch_size() != 0:
                    num_batches += 1
                print(f"Epoch {epoch}, Loss: {running_loss / num_batches}")

        if if_save:
            mindspore.save_checkpoint(self.model, save_path)

    def load(self, load_path):
        mindspore.load_checkpoint(load_path, self.model)


class DenoisingAutoEncoder_2():
    def __init__(self, img_shape = (1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.SequentialCell(
                nn.Conv2d(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1, pad_mode='pad'),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, pad_mode='pad'),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1, pad_mode='pad'),
                )
        
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=0.001)
        criterion = nn.MSELoss()
        
        # Create training step function
        def forward_fn(data_train, noisy_data):
            output = self.model(noisy_data)
            loss = criterion(output, data_train)
            return loss
        
        # Get gradients function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
        
        def train_step(data_train, noisy_data):
            loss, grads = grad_fn(data_train, noisy_data)
            optimizer(grads)
            return loss
            
        for epoch in range(num_epochs):
            self.model.set_train()
            running_loss = 0.0
            for batch_data in data:
                data_train = Tensor(batch_data[0], dtype=mindspore.float32)
                data_label = batch_data[1]
                
                noise_shape = data_train.shape
                # RISK_INFO: [随机性处理差异] - 'np.random.randn' 与 'torch.randn_like' 的随机数生成器实现和种子管理可能不同，可能导致在相同种子下产生不同的随机噪声序列。
                noise = v_noise * Tensor(np.random.randn(*noise_shape), dtype=mindspore.float32)
                noisy_data = ops.clip_by_value(data_train + noise, clip_value_min=min, clip_value_max=max)
                
                loss = train_step(data_train, noisy_data)
                running_loss += loss.asnumpy()
                
            if epoch % 10 == 0:
                # Get the number of batches
                num_batches = data.get_dataset_size() // data.get_batch_size()
                if data.get_dataset_size() % data.get_batch_size() != 0:
                    num_batches += 1
                print(f"Epoch {epoch}, Loss: {running_loss / num_batches}")

        if if_save:
            mindspore.save_checkpoint(self.model, save_path)
        
    def load(self, load_path):
        mindspore.load_checkpoint(load_path, self.model)