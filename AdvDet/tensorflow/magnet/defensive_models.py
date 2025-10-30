# AUDIT_SKIP
import tensorflow as tf
from tensorflow import keras

device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'

class DenoisingAutoEncoder_1():
    def __init__(self, img_shape=(28,28,1)):
        self.img_shape = img_shape
    
        self.model = keras.Sequential([
                keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', input_shape=img_shape),
                keras.layers.Activation('sigmoid'),
                keras.layers.AveragePooling2D(pool_size=2),
                keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
                keras.layers.Activation('sigmoid'),
                keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
                keras.layers.Activation('sigmoid'),
                keras.layers.UpSampling2D(size=2),
                keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
                keras.layers.Activation('sigmoid'),
                keras.layers.Conv2D(filters=self.img_shape[2], kernel_size=3, padding='same'),
                ])
    
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        # RISK_INFO: [API 行为不等价] - TensorFlow的Adam优化器默认epsilon为1e-7，而PyTorch中为1e-8，这可能导致数值上的细微差异。
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        criterion = keras.losses.MeanSquaredError()
        
        @tf.function
        def train_step(data_train, noisy_data):
            with tf.GradientTape() as tape:
                output = self.model(noisy_data, training=True)
                loss = criterion(data_train, output)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_i, (data_train,data_label) in enumerate(data):
                # Convert from channels first to channels last
                # data_train = tf.transpose(data_train, [0, 2, 3, 1])
                # RISK_INFO: [随机性处理差异] - tf.random.normal 与 torch.randn_like 使用不同的伪随机数生成器，即使种子相同，生成的随机数序列也可能不同。
                noise = v_noise * tf.random.normal(shape=data_train.shape)
                noisy_data = tf.clip_by_value(data_train+noise, min, max)
                loss = train_step(data_train, noisy_data)
                running_loss += loss.numpy()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {running_loss / len(data)}")

        if if_save:
            self.model.save_weights(save_path)

    def load(self, load_path):
        # RISK_INFO: [API 行为不等价] - The original torch.load used weights_only=False, which can execute arbitrary code via pickle. tf.keras.Model.load_weights is safer as it only loads weight values. This represents a behavioral difference, especially in handling potentially malicious model files.
        self.model.load_weights(load_path)


class DenoisingAutoEncoder_2():
    def __init__(self, img_shape = (28,28,1)):
        self.img_shape = img_shape
    
        self.model = keras.Sequential([
                keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', input_shape=img_shape),
                keras.layers.Activation('sigmoid'),
                keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
                keras.layers.Activation('sigmoid'),
                keras.layers.Conv2D(filters=self.img_shape[2], kernel_size=3, padding='same'),
                ])
        
    def train(self, data, save_path, v_noise=0, min=0, max=1, num_epochs=100, if_save=True):
        # RISK_INFO: [API 行为不等价] - TensorFlow的Adam优化器默认epsilon为1e-7，而PyTorch中为1e-8，这可能导致数值上的细微差异。
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        criterion = keras.losses.MeanSquaredError()
        
        @tf.function
        def train_step(data_train, noisy_data):
            with tf.GradientTape() as tape:
                output = self.model(noisy_data, training=True)
                loss = criterion(data_train, output)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_i, (data_train,data_label) in enumerate(data):
                # Convert from channels first to channels last
                # data_train = tf.transpose(data_train, [0, 2, 3, 1])
                # RISK_INFO: [随机性处理差异] - tf.random.normal 与 torch.randn_like 使用不同的伪随机数生成器，即使种子相同，生成的随机数序列也可能不同。
                noise = v_noise * tf.random.normal(shape=data_train.shape) 
                noisy_data = tf.clip_by_value(data_train+noise, min, max)
                loss = train_step(data_train, noisy_data)
                running_loss += loss.numpy()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {running_loss / len(data)}")

        if if_save:
            self.model.save_weights(save_path)
        
    def load(self, load_path):
        # RISK_INFO: [API 行为不等价] - The original torch.load used weights_only=False, which can execute arbitrary code via pickle. tf.keras.Model.load_weights is safer as it only loads weight values. This represents a behavioral difference, especially in handling potentially malicious model files.
        self.model.load_weights(load_path)