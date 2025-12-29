import paddle
from paddle import nn

device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
paddle.set_device(device)

class DenoisingAutoEncoder_1():
    def __init__(self, img_shape=(1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.Sequential(
                nn.Conv2D(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.AvgPool2D(2),
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=2),
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2D(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
                )
    
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=0.001)
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch_i, (data_train, data_label) in enumerate(data):
                noise = v_noise * paddle.randn(data_train.shape, dtype=data_train.dtype)
                noisy_data = paddle.clip(data_train + noise, min=min, max=max)
                # RISK_INFO: [API 行为不等价] - 原代码使用 .to(device) 显式地将数据移动到指定设备。当前实现依赖于全局的 paddle.set_device() 和 DataLoader 的正确配置，如果数据未被自动放置在正确设备上，可能会导致运行时错误。
                data_train = paddle.to_tensor(data_train)
                noisy_data = paddle.to_tensor(noisy_data)
                optimizer.clear_grad()
                output = self.model(noisy_data)
                loss = criterion(output, data_train)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {running_loss / len(data)}")

        if if_save:
            paddle.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        # RISK_INFO: [API 行为不等价] - torch.load 的 map_location 参数提供了更精细的设备控制。虽然 paddle.set_device() 提供了全局设备设置，但在复杂的设备映射场景下，行为可能存在差异。
        self.model.set_state_dict(paddle.load(load_path))


class DenoisingAutoEncoder_2():
    def __init__(self, img_shape = (1,28,28)):
        self.img_shape = img_shape
    
        self.model = nn.Sequential(
                nn.Conv2D(in_channels=self.img_shape[0], out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid(),
                nn.Conv2D(in_channels=3, out_channels=self.img_shape[0], kernel_size=3, padding=1),
                )
        
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=0.001)
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch_i, (data_train, data_label) in enumerate(data):
                noise = v_noise * paddle.randn(data_train.shape, dtype=data_train.dtype)
                noisy_data = paddle.clip(data_train + noise, min=min, max=max) 
                # RISK_INFO: [API 行为不等价] - 原代码使用 .to(device) 显式地将数据移动到指定设备。当前实现依赖于全局的 paddle.set_device() 和 DataLoader 的正确配置，如果数据未被自动放置在正确设备上，可能会导致运行时错误。
                data_train = paddle.to_tensor(data_train)
                noisy_data = paddle.to_tensor(noisy_data)
                optimizer.clear_grad()
                output = self.model(noisy_data)
                loss = criterion(output, data_train)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {running_loss / len(data)}")

        if if_save:
            paddle.save(self.model.state_dict(), save_path)
        
    def load(self, load_path):
        # RISK_INFO: [API 行为不等价] - torch.load 的 map_location 参数提供了更精细的设备控制。虽然 paddle.set_device() 提供了全局设备设置，但在复杂的设备映射场景下，行为可能存在差异。
        self.model.set_state_dict(paddle.load(load_path))