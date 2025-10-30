import sys
sys.path.append('../..')

from common.util import *
from setup_paths import *
from baseline.models import ResNet18

class SVHNCNN:
    def __init__(self, mode='train', filename="cnn_svhn.h5", epochs=100, batch_size=256):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_svhn()
        
        # TensorFlow uses NHWC format by default, no need to transpose
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)
        
        if mode=='train':
            # build model
            self.classifier = ResNet18()
            self.classifier.build(input_shape=(None,) + self.input_shape)
            self.classifier = self.art_classifier(self.classifier)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)  
            # save model
            self.classifier.model.save(str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            # CRITICAL_ERROR: [命名不一致] - 变量 'keras' 未定义，应为 'tf.keras'。
            # 不一致的实现:
            # self.classifier = self.art_classifier(keras.models.load_model(str(os.path.join(checkpoints_dir, self.filename))))
            # 原架构中的对应代码:
            # self.classifier = self.art_classifier(torch.load(str(os.path.join(checkpoints_dir, self.filename)), weights_only=False))
            raise NotImplementedError("变量 'keras' 未定义，应为 'tf.keras'。")
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def art_classifier(self, net):
        # TensorFlow uses NHWC format, so we don't need reshape(3, 1, 1)
        mean = np.asarray((0.4377, 0.4438, 0.4728))
        std = np.asarray((0.1980, 0.2010, 0.1970))
        
        # RISK_INFO: [API 行为不等价] - tf.keras.losses.CategoricalCrossentropy and nn.CrossEntropyLoss might have subtle implementation differences affecting numerical stability or gradients, although both expect logits.
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # CRITICAL_ERROR: [命名不一致] - 变量 'optimizers' 未定义，应为 'tf.keras.optimizers'。
        # 不一致的实现:
        # optimizer = optimizers.Adam(learning_rate=0.0001)
        # 原架构中的对应代码:
        # optimizer = optim.Adam(net.parameters(), lr=0.0001)
        raise NotImplementedError("变量 'optimizers' 未定义，应为 'tf.keras.optimizers'。")
        
        classifier = TensorFlowV2Classifier(
            model=net,
            clip_values=(0, 1),
            loss_object=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            preprocessing=(mean, std)
        )
        
        return classifier