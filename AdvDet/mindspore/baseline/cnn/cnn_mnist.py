# AUDIT_SKIP
from common.util import *
from setup_paths import *

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = nn.SequentialCell(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Dense(9216, 128),
            nn.ReLU()
        )
        self.layer3 = nn.SequentialCell(
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Dense(128, 10)
        )

    def construct(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
    
class MNISTCNN:
    def __init__(self, mode='train', filename="cnn_mnist.ckpt", epochs=50, batch_size=128):
        self.mode = mode # train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Set MindSpore context
        # Try to use GPU if available
        try:
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        except:
            # Fall back to CPU if GPU is not available
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_mnist()

        # Swap axes to MindSpore's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)

        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if self.mode=='train':
            self.classifier = self.art_classifier(Net())
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)
            # save model
            mindspore.save_checkpoint(self.classifier.model, str(os.path.join(checkpoints_dir, self.filename)))
        elif self.mode=='load':
            net = Net()
            mindspore.load_checkpoint(str(os.path.join(checkpoints_dir, self.filename)), net=net)
            self.classifier = self.art_classifier(net)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

        pred = self.classifier.predict(self.x_test, training_mode=False)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))

    def art_classifier(self, net):
        from art.estimators.classification.mindspore import MindSporeClassifier
        
        mean = [0.1307]
        std  = [0.3081]
        
        # Create loss function
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        
        # Create optimizer
        # RISK_INFO: [API 行为不等价] - 'net.trainable_params()' 仅返回可训练参数，而原代码的 'net.parameters()' 返回所有参数，在存在冻结层等情况下行为可能不同。
        optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)
        
        classifier = MindSporeClassifier(
            model=net,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            preprocessing=(mean, std)
        )

        return classifier