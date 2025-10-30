import sys
sys.path.append('../..')

from common.util import *
from setup_paths import *

class MNISTCNN:
    def __init__(self, mode='train', filename="cnn_mnist.h5", epochs=50, batch_size=128):
        self.mode = mode # train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_mnist()
        
        # TensorFlow uses NHWC format by default, no need to transpose
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)
        
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)
        
        if self.mode=='train':
            self.model = self.build_model()
            self.classifier = self.art_classifier(self.model)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)
            # save model
            self.model.save(str(os.path.join(checkpoints_dir, self.filename)))
        elif self.mode=='load':
            self.model = keras.models.load_model(str(os.path.join(checkpoints_dir, self.filename)))
            self.classifier = self.art_classifier(self.model)
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test, training_mode=False)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def build_model(self):
        model = models.Sequential([
            # Layer 0
            # RISK_INFO: [API 行为不等价] - Keras' Conv2D defaults to 'glorot_uniform' weight initialization, while PyTorch's Conv2d uses 'kaiming_uniform'. This can lead to different training dynamics.
            layers.Conv2D(64, kernel_size=3, strides=1, input_shape=self.input_shape),
            # RISK_INFO: [API 行为不等价] - Keras' BatchNormalization defaults (momentum=0.99, epsilon=0.001) differ from PyTorch's (momentum=0.1, eps=1e-05). This may affect model convergence and final weights.
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Layer 1
            # RISK_INFO: [API 行为不等价] - Keras' Conv2D defaults to 'glorot_uniform' weight initialization, while PyTorch's Conv2d uses 'kaiming_uniform'. This can lead to different training dynamics.
            layers.Conv2D(64, kernel_size=3, strides=1),
            # RISK_INFO: [API 行为不等价] - Keras' BatchNormalization defaults (momentum=0.99, epsilon=0.001) differ from PyTorch's (momentum=0.1, eps=1e-05). This may affect model convergence and final weights.
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Layer 2
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Dropout(0.5),
            layers.Flatten(),
            # RISK_INFO: [API 行为不等价] - Keras' Dense layer defaults to 'glorot_uniform' weight initialization, while PyTorch's Linear layer uses 'kaiming_uniform'. This can lead to different training dynamics.
            layers.Dense(128),
            layers.ReLU(),
            
            # Layer 3
            # RISK_INFO: [API 行为不等价] - Keras' BatchNormalization defaults (momentum=0.99, epsilon=0.001) differ from PyTorch's (momentum=0.1, eps=1e-05). This may affect model convergence and final weights.
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            # RISK_INFO: [API 行为不等价] - Keras' Dense layer defaults to 'glorot_uniform' weight initialization, while PyTorch's Linear layer uses 'kaiming_uniform'. This can lead to different training dynamics.
            layers.Dense(10)
        ])
        
        return model
    
    def art_classifier(self, model):
        mean = np.array([0.1307]).astype(np.float32)
        std = np.array([0.3081]).astype(np.float32)
        
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
        
        loss_object = CrossEntropyLossWrapper()
        # RISK_INFO: [API 行为不等价] - Keras' Adam optimizer defaults to epsilon=1e-7, while PyTorch's defaults to eps=1e-8. This small difference can affect numerical stability and convergence in some cases.
        optimizer = optimizers.Adam(learning_rate=0.01)
        
        classifier = TensorFlowV2Classifier(
            model=model,
            clip_values=(0, 1),
            loss_object=loss_object,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
            preprocessing=(mean, std)
        )
        
        return classifier