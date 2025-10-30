from common.util import *
from setup_paths import *

class ImageNetCNN:
    def __init__(self, filename="cnn_imagenet.pd"):
        self.filename = filename
        self.device = paddle.device.get_device()

        # Load data
        self.num_classes = 1000
        path_imagenet = '/remote-home/wangxin/Data/imagenet_s/'

        transform = paddle.vision.transforms.Compose([
            paddle.vision.transforms.Resize(256),
            paddle.vision.transforms.CenterCrop(224),
            paddle.vision.transforms.ToTensor()
        ])
        
        # PaddlePaddle doesn't have built-in ImageNet dataset, so we need to use custom loader
        # For now, we'll raise NotImplementedError as per the guidelines
        raise NotImplementedError("Skip - ImageNet dataset loading requires custom implementation in PaddlePaddle")
        
        # The rest of the code would follow similar pattern:
        # train_dataset = custom ImageNet loader
        # val_dataset = custom ImageNet loader
        # train_loader = paddle.io.DataLoader(train_dataset, batch_size=10000, shuffle=True, num_workers=4)
        # val_loader = paddle.io.DataLoader(val_dataset, batch_size=5000, shuffle=False, num_workers=4)
        #
        # self.x_train, self.y_train = next(iter(train_loader))
        # self.x_test, self.y_test = next(iter(val_loader))
        #
        # self.x_train, self.y_train = self.x_train.numpy(), paddle.nn.functional.one_hot(self.y_train, self.num_classes).numpy().astype(np.float32)
        # self.x_test, self.y_test = self.x_test.numpy(), paddle.nn.functional.one_hot(self.y_test, self.num_classes).numpy().astype(np.float32)
        # self.min_pixel_value, self.max_pixel_value = 0, 1
        #
        # self.input_shape = self.x_train.shape[1:]
        # print(self.input_shape)
        #
        # self.classifier = self.art_classifier(paddle.vision.models.resnet50(pretrained=True))
        
    def art_classifier(self, net):
        # CRITICAL_ERROR: [核心功能缺失] - 模型未被移动到指定的计算设备 (device)，原作中有 net.to(self.device) 的显式调用。
        # 不一致的实现:
        # # (此功能在新架构中缺失)
        # 原架构中的对应代码:
        # net.to(self.device)
        raise NotImplementedError("模型未被移动到指定的计算设备 (device)，原作中有 net.to(self.device) 的显式调用。")
        # summary(net, input_size=self.input_shape)

        mean = np.asarray((0.485, 0.456, 0.406)).reshape((3, 1, 1))
        std = np.asarray((0.229, 0.224, 0.225)).reshape((3, 1, 1))
        
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