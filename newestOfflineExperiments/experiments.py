from models import ContinualLearningModel
from data_loader import CORE50
from utils import *
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from matplotlib import pyplot as plt
import json

DATASET_ROOT = 'C:/Users/nikol/Desktop/University/Year-4/ADE/ThesisCodeExperiments/CORe50-Dataset/core50_128x128/'

class Experiments:

    def __init__(self):
        print("> Experiments Initialized")

    def plotExperiment(self,experiment_name,title):
        min_val = 100
        max_val = 50
        with open('experiments/' + experiment_name + '.json', ) as json_file:
            usecases = json.load(json_file)
            for usecase in usecases:
                for key, value in usecase.items():
                    plt.plot(value['acc'], label=key)
                    cur_min = min(value['acc'])
                    cur_max = max(value['acc'])
                    if cur_min < min_val:
                        min_val = cur_min
                    elif cur_max > max_val:
                        max_val = cur_max

        plt.title(title)
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Encountered Batches")
        plt.yticks(np.arange(round(min_val), round(max_val)+10, 5))
        plt.grid()
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(experiment_name)
    
    def storeExperimentOutputNew(self, experiment_name, usecase_name, accuracies, losses):
        data = []

        # Load previously recorded usescases
        with open('experiments/' + experiment_name + '.json', ) as json_file:
            data = json.load(json_file)

        # Store new usecase
        exp = dict()
        exp[usecase_name] = dict()
        exp[usecase_name]["acc"] = accuracies
        exp[usecase_name]["loss"] = losses
        data.append(exp)

        # Write the updated data back to the file
        with open('experiments/' + experiment_name + '.json', 'w') as outfile:
            json.dump(data, outfile)

    def print_trainable_status(self, model):
        for layer in model.layers:
            print(f"{layer.name}: {layer.trainable}")

    # New implementation with hidden layers
    def runHiddenLayersExperiment(self, experiment_name, usecase, replay_size, num_hidden_layers):
        print("> Running Hidden Layers experiment")

        dataset = CORE50(root=DATASET_ROOT, scenario="nicv2_391", preload=False)
        test_x, test_y = dataset.get_test_set()
        test_x = preprocess(test_x)

        # Building main model
        cl_model = ContinualLearningModel(image_size=128, name=usecase,replay_buffer=replay_size)
        cl_model.buildBaseHidden(hidden_layers=num_hidden_layers)
        cl_model.buildHeadHidden(sl_units=128, hidden_layers=num_hidden_layers)
        cl_model.buildCompleteModel()

        # Used for debugging

        # After building the complete model
        # print("Base model trainable status:")
        # self.print_trainable_status(cl_model.base)

        # print("\nHead model trainable status:")
        # self.print_trainable_status(cl_model.head)

        # print("\nComplete model trainable status:")
        # self.print_trainable_status(cl_model.model)

        # print("\nComplete model summary:")
        # cl_model.model.summary()

        # Stop the program
        # exit()

        accuracies = []
        losses = []

        # Training, loop over the training incremental batches
        for i, train_batch in enumerate(dataset):
            train_x, train_y = train_batch
            train_x = preprocess(train_x)

            print("----------- batch {0} -------------".format(i))
            print("train_x shape: {}, train_y shape: {}"
                  .format(train_x.shape, train_y.shape))

            if i == 1:
                cl_model.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                cl_model.head.compile(optimizer=tf.keras.optimizers.SGD(lr=0.00005),
                                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Padding of the first batch. Unsure about this
            # Reimplement these
            if i == 0:
                (train_x, train_y), it_x_ep = pad_data([train_x, train_y], 128)
            shuffle_in_unison([train_x, train_y], in_place=True)

            print("---------------------------------")

            features = cl_model.feature_extractor.predict(train_x)
            cl_model.head.fit(features, train_y, epochs=4, verbose=0)
            if i >= 1:
                cl_model.replay()

            cl_model.storeRepresentations(train_x, train_y)

            # Evaluate the model on the test set
            loss, acc = cl_model.model.evaluate(test_x, test_y)
            accuracies.append(round(acc*100,1))
            losses.append(loss)
            print("> ", cl_model.name, " Accuracy: ", acc, " Loss: ", loss)
            print("---------------------------------")

        # Store results in json file
        self.storeExperimentOutputNew(experiment_name=experiment_name,
                                   usecase_name=usecase,
                                   accuracies=accuracies,
                                   losses=losses)