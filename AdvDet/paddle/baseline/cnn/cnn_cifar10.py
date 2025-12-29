from common.util import *
from setup_paths import *
# from baseline.models import *

class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()

        basic_dropout_rate = 0.1
        self.layer0 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(64),
            paddle.nn.ReLU()
        )
        self.layer1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(64),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Dropout(basic_dropout_rate)
        )
        self.layer2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(128),
            paddle.nn.ReLU()
        )
        self.layer3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(128),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Dropout(basic_dropout_rate + 0.1)
        )
        self.layer4 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(256),
            paddle.nn.ReLU()
        )
        self.layer5 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(256),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Dropout(basic_dropout_rate + 0.2)
        )
        self.layer6 = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            paddle.nn.BatchNorm2D(512),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Dropout(basic_dropout_rate + 0.3)
        )
        self.layer7 = paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(2048, 512),
        )
        self.classification_head = paddle.nn.Sequential(
            paddle.nn.Linear(512, 10)
        )
    
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.classification_head(out)
        return out

class CIFAR10CNN:
    def __init__(self, mode='train', filename="cnn_cifar10.pdparams", epochs=100, batch_size=512):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_cifar10()
        
        # Swap axes to PyTorch's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)
        # RISK_INFO: [实现差异的潜在风险] - 标签数据类型被提前转换为float32，这在原架构中没有。此举是为了适应Paddle的CrossEntropyLoss(soft_label=True)，但改变了数据预处理流程。
        # Convert labels to float32 for soft_label=True in CrossEntropyLoss
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if mode=='train':
            # build model
            # self.classifier = ResNet18()
            self.classifier = Net()
            self.classifier = self.art_classifier(self.classifier)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)  
            # save model
            paddle.save(self.classifier.model.state_dict(), str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            model = Net()
            model.set_state_dict(paddle.load(str(os.path.join(checkpoints_dir, self.filename))))
            self.classifier = self.art_classifier(model)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def art_classifier(self, net):
        # summary(net, input_size=self.input_shape)
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
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
        optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.001)
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