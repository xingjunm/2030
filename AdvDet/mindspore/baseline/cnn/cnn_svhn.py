from common.util import *
from setup_paths import *
from baseline.models import *

class SVHNCNN:
    def __init__(self, mode='train', filename="cnn_svhn.ckpt", epochs=100, batch_size=256):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Set MindSpore context
        try:
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        except:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_svhn()
        
        # Swap axes to MindSpore's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)

        if mode=='train':
            # build model
            self.classifier = ResNet18()
            self.classifier = self.art_classifier(self.classifier)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)  
            # save model
            mindspore.save_checkpoint(self.classifier.model, str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            model = ResNet18()
            mindspore.load_checkpoint(str(os.path.join(checkpoints_dir, self.filename)), net=model)
            self.classifier = self.art_classifier(model)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def art_classifier(self, net):
        mean = np.asarray((0.4377, 0.4438, 0.4728)).reshape((3, 1, 1))
        std = np.asarray((0.1980, 0.2010, 0.1970)).reshape((3, 1, 1))
        
        criterion = CrossEntropyLossWrapper()
        optimizer = nn.Adam(net.trainable_params(), learning_rate=0.0001)
        
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