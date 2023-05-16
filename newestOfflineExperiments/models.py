import tensorflow as tf
from keras import layers
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import gc
from keras.applications import MobileNetV2

class ContinualLearningModel:

    def __init__(self,image_size=224,name="None",replay_buffer=3000):
        print("> Continual Learning Model Initiated")
        # base = bases.MobileNetV2Base(image_size=224)
        self.image_size = image_size
        self.name = name
        self.replay_representations_x = []
        self.replay_representations_y = []
        self.replay_buffer = replay_buffer # The number of patterns stored
        
    def buildBaseHidden(self,hidden_layers=0):
        baseModel = tf.keras.applications.MobileNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                      alpha=1.0,
                                                      include_top=False,
                                                      weights='imagenet')
        
        # Batch normalization layers replaced with bactch renormalization layers - Better for CL
        for l in baseModel.layers:
            if ('_BN' in l.name):
                l.renorm = True
        
        baseModel.trainable = False

        base_model_truncated = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.layers[-hidden_layers-1].output)
        self.base = base_model_truncated

        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        f = inputs
        f_out = self.base(f)
        self.feature_extractor = tf.keras.Model(f, f_out)
        self.feature_extractor.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy',
                                       metrics=['accuracy'])

    # Head with hidden layers
    def buildHeadHidden(self, sl_units=32, hidden_layers=0):

        baseModel = tf.keras.applications.MobileNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                      alpha=1.0,
                                                      include_top=False,
                                                      weights='imagenet')

        self.sl_units = sl_units

        # Create a new head model
        self.head = tf.keras.Sequential()

        # Add the last N layers of MobileNetV2 to the head model
        for i in range(-hidden_layers, 0):
            layer = baseModel.layers[i]
            layer.trainable = True
            self.head.add(layer)

        self.head.add(layers.Flatten(input_shape=(4, 4, 1280)))
        # Removed the dense layer since we add Hidden Layers from MobileNetV2 now
        # self.head.add(layers.Dense(
        # units=sl_units,
        # activation='relu',
        # kernel_regularizer=l2(0.01),
        # bias_regularizer=l2(0.01)
        # ))

        # Softmax layer (Last layer)
        self.head.add(layers.Dense(
            units=50,  # Number of classes
            activation='softmax',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
        )

        self.head.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])    

    def buildCompleteModel(self):
        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        x = inputs
        # we dont use inputs here
        x = self.base(x)
        outputs = self.head(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        # self.model.summary()

    # Refreshing replay memory with new samples and removing old ones if necessary
    # Fix bug with patterns being a little less than they should be
    # This bug I think that it is caused because the replacement batch size is calculated as a percentage of the replay buffer
    # Which is greater than the actual number of patterns stored in the replay buffer
    def storeRepresentations(self, train_x, train_y):
        # We ened to add replay representations and replacement batch size not train_x
        replacement_batch_size = int(self.replay_buffer * 0.015) # 1.5% sample replacement
        replay_memory_size = len(self.replay_representations_x) + replacement_batch_size

        # It's good to know that the first batch has around 3000+ samples 
        # The rest are around 300ish
        # Towards the end we have again around 
        # For testing purposes
        # print("Replay Memory Size: ",replay_memory_size)
        # print("replay representation x: ",len(self.replay_representations_x))
        # print("train x: ",len(train_x))

        # If the replay buffer will overfill we need to remove some old samples
        if replay_memory_size >= self.replay_buffer:

            x_sample, y_sample = zip(*random.choices(list(zip(train_x, train_y)), k=replacement_batch_size))
            x_sample = self.feature_extractor.predict(np.array(x_sample))

            # Removing old samples
            patterns_to_delete = random.sample(range(len(self.replay_representations_x)), replacement_batch_size)
            for pat in sorted(patterns_to_delete, reverse=True):
                del self.replay_representations_x[pat]
                del self.replay_representations_y[pat]

        # If it will not overfill we just add the new samples
        else:
            x_sample, y_sample = zip(*random.choices(list(zip(train_x, train_y)), k=replacement_batch_size))
            x_sample = self.feature_extractor.predict(np.array(x_sample))

        # Adding new ones
        for i in range(len(x_sample)):
            self.replay_representations_x.append(x_sample[i])
            self.replay_representations_y.append(y_sample[i])

        gc.collect()
        print("Replay X: ",len(self.replay_representations_x)," Replay Y: ",len(self.replay_representations_y))

    def replay(self):

        replay_x = np.array(self.replay_representations_x)
        replay_y = np.array(self.replay_representations_y)

        print("> REPLAYING")
        # Fitting for just 1 epoch
        self.head.fit(replay_x,replay_y,epochs=1,verbose=0) 
