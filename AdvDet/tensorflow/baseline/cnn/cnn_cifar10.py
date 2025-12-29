import sys
sys.path.append('../..')

from common.util import *
from setup_paths import *

class CIFAR10CNN:
    def __init__(self, mode='train', filename="cnn_cifar10.h5", epochs=100, batch_size=512):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_cifar10()
        
        # TensorFlow uses NHWC format by default, no need to transpose
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)
        
        if mode=='train':
            # build model
            self.model = self.build_model()
            self.classifier = self.art_classifier(self.model)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)  
            # save model
            self.model.save(str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            self.model = keras.models.load_model(str(os.path.join(checkpoints_dir, self.filename)))
            self.classifier = self.art_classifier(self.model)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def build_model(self):
        basic_dropout_rate = 0.1
        
        model = models.Sequential([
            # Layer 0
            layers.Conv2D(64, kernel_size=3, padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Layer 1
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(basic_dropout_rate),
            
            # Layer 2
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Layer 3
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(basic_dropout_rate + 0.1),
            
            # Layer 4
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Layer 5
            layers.Conv2D(256, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(basic_dropout_rate + 0.2),
            
            # Layer 6
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(basic_dropout_rate + 0.3),
            
            # Layer 7
            layers.Flatten(),
            layers.Dense(512),
            
            # Classification head
            layers.Dense(10)
        ])
        
        return model
    
    def art_classifier(self, model):
        mean = np.array([0.4914, 0.4822, 0.4465]).astype(np.float32)
        std = np.array([0.2023, 0.1994, 0.2010]).astype(np.float32)
        
        # Custom loss wrapper to handle data type conversion
        class CrossEntropyLossWrapper(tf.keras.losses.Loss):
            def __init__(self):
                super().__init__()
                self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            
            def call(self, y_true, y_pred):
                # Ensure both are float32
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                return self.loss_fn(y_true, y_pred)
        
        criterion = CrossEntropyLossWrapper()
        optimizer = optimizers.Adam(learning_rate=0.001)
        
        classifier = TensorFlowV2Classifier(
            model=model,
            clip_values=(0, 1),
            loss_object=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            preprocessing=(mean, std)
        )
        
        return classifier