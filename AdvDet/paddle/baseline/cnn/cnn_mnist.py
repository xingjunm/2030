# AUDIT_SKIP
from common.util import *
from setup_paths import *

class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = paddle.nn.Sequential(
            # RISK_INFO: [张量操作差异] - The underlying implementation of Conv2D in PaddlePaddle might differ from PyTorch's Conv2d in terms of numerical precision or algorithm (e.g., cuDNN version/options).
            paddle.nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            # RISK_INFO: [API 行为不等价] - BatchNorm layers' running statistics (mean/var) and momentum can have subtle implementation differences between frameworks, affecting model behavior.
            paddle.nn.BatchNorm2D(64),
            # RISK_INFO: [API 行为不等价] - PaddlePaddle's ReLU might have different behavior for edge cases like NaN/Inf compared to PyTorch's version.
            paddle.nn.ReLU()
        )
        self.layer1 = paddle.nn.Sequential(
            # RISK_INFO: [张量操作差异] - The underlying implementation of Conv2D in PaddlePaddle might differ from PyTorch's Conv2d in terms of numerical precision or algorithm (e.g., cuDNN version/options).
            paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # RISK_INFO: [API 行为不等价] - BatchNorm layers' running statistics (mean/var) and momentum can have subtle implementation differences between frameworks, affecting model behavior.
            paddle.nn.BatchNorm2D(64),
            # RISK_INFO: [API 行为不等价] - PaddlePaddle's ReLU might have different behavior for edge cases like NaN/Inf compared to PyTorch's version.
            paddle.nn.ReLU()
        )
        self.layer2 = paddle.nn.Sequential(
            # RISK_INFO: [张量操作差异] - The underlying implementation of MaxPool2D in PaddlePaddle might differ from PyTorch's, especially in handling ties or border effects.
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            # RISK_INFO: [随机性处理差异] - Dropout implementations can have subtle differences in random number generation, potentially affecting reproducibility.
            paddle.nn.Dropout(0.5),
            # RISK_INFO: [张量操作差异] - Flatten behavior should be consistent, but it's worth noting potential implementation differences.
            paddle.nn.Flatten(),
            # RISK_INFO: [张量操作差异] - The underlying matrix multiplication (GEMM) implementation or its default precision could differ between frameworks.
            paddle.nn.Linear(9216, 128),
            # RISK_INFO: [API 行为不等价] - PaddlePaddle's ReLU might have different behavior for edge cases like NaN/Inf compared to PyTorch's version.
            paddle.nn.ReLU()
        )
        self.layer3 = paddle.nn.Sequential(
            # RISK_INFO: [API 行为不等价] - BatchNorm layers' running statistics (mean/var) and momentum can have subtle implementation differences between frameworks, affecting model behavior.
            paddle.nn.BatchNorm1D(128),
            # RISK_INFO: [随机性处理差异] - Dropout implementations can have subtle differences in random number generation, potentially affecting reproducibility.
            paddle.nn.Dropout(0.5),
            # RISK_INFO: [张量操作差异] - The underlying matrix multiplication (GEMM) implementation or its default precision could differ between frameworks.
            paddle.nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
    
class MNISTCNN:
    def __init__(self, mode='train', filename="cnn_mnist.pdparams", epochs=50, batch_size=128):
        self.mode = mode # train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        # Set device to GPU if available, matching PyTorch behavior
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device('gpu')
            self.device = self.__create_device_placeholder('gpu')
        else:
            paddle.device.set_device('cpu')
            self.device = self.__create_device_placeholder('cpu')

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_mnist()

        # Swap axes to PyTorch's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)
        
        # Ensure labels are float32 to match model output
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if self.mode=='train':
            self.classifier = self.art_classifier(Net())
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)
            # save model
            # RISK_INFO: [API 行为不等价] - `paddle.save` and `torch.save` use different serialization mechanisms. While saving the whole model is attempted in both, subtle differences upon loading could exist.
            paddle.save(self.classifier.model.state_dict(), str(os.path.join(checkpoints_dir, self.filename)))
        elif self.mode=='load':
            # RISK_INFO: [API 行为不等价] - `paddle.load` and `torch.load` use different serialization mechanisms. The loaded model object might have subtle behavioral differences from the PyTorch equivalent.
            model = Net()
            state_dict = paddle.load(str(os.path.join(checkpoints_dir, self.filename)))
            model.set_state_dict(state_dict)
            self.classifier = self.art_classifier(model)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

        pred = self.classifier.predict(self.x_test, training_mode=False)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))

    def art_classifier(self, net):
        # In PaddlePaddle, models are automatically placed on the correct device
        # based on the global device setting, no explicit .to() call needed
        # summary(net, input_size=self.input_shape)
        
        mean = [0.1307]
        std  = [0.3081]
        
        # Custom loss wrapper to handle data type conversion
        class CrossEntropyLossWrapper(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.loss_fn = paddle.nn.CrossEntropyLoss(soft_label=True)
            
            def forward(self, input, label):
                # Ensure both input and label are float32
                input = paddle.cast(input, 'float32')
                label = paddle.cast(label, 'float32')
                return self.loss_fn(input, label)
        
        criterion = CrossEntropyLossWrapper()
        optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.01)
        # RISK_INFO: [实现差异的潜在风险] - `PaddleClassifier` may have internal implementation differences compared to `PyTorchClassifier` (e.g., in batch processing, gradient handling) that could affect results beyond the model architecture itself.
        classifier = PaddleClassifier(
            model=net,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            preprocessing=(mean, std)
        )

        return classifier

    def __create_device_placeholder(self, device_type):
        """Create a device placeholder object to maintain interface compatibility"""
        class DevicePlaceholder:
            def __init__(self, device_type):
                self.type = device_type
            
            def __str__(self):
                return self.type
            
            def __repr__(self):
                return f"device('{self.type}')"
        
        return DevicePlaceholder(device_type)