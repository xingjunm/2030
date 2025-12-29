from common.util import *
from setup_paths import *
from baseline.models import *

class SVHNCNN:
    def __init__(self, mode='train', filename="cnn_svhn.pd", epochs=100, batch_size=256):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        # RISK_INFO: [API 行为不等价] - paddle.device.get_device() 的行为可能与 torch.device('cuda' if torch.cuda.is_available() else 'cpu') 在设备选择逻辑上存在细微差异，例如在多GPU环境下的默认设备选择。
        self.device = paddle.device.get_device()

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_svhn()
        
        # Swap axes to PyTorch's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)
        # Convert labels to float32 for soft_label=True in CrossEntropyLoss
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if mode=='train':
            # build model
            self.classifier = ResNet18()
            self.classifier = self.art_classifier(self.classifier)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)  
            # save model
            paddle.save(self.classifier.model.state_dict(), str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            model = ResNet18()
            model.set_state_dict(paddle.load(str(os.path.join(checkpoints_dir, self.filename))))
            self.classifier = self.art_classifier(model)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def art_classifier(self, net):
        # summary(net, input_size=self.input_shape)
        
        mean = np.asarray((0.4377, 0.4438, 0.4728)).reshape((3, 1, 1))
        std = np.asarray((0.1980, 0.2010, 0.1970)).reshape((3, 1, 1))
        
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
        optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.0001)
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