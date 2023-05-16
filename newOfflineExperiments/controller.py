from experiments import Experiments
import argparse
import numpy as np
import tensorflow as tf

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_cl_default", action="store_true")
    parser.add_argument("--exp_sample_replacement_method", action="store_true")
    parser.add_argument("--exp_hidden_layers", action="store_true")
    args = parser.parse_args()
    experiments = Experiments()

    # using 7500 on the replay buffer as the default for the continual learning experiments - best 

    if args.exp_cl_default:
        print("> Experiment: Continual Learning Default")
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="CONTINUAL_VS_TRANSFER_LEARNING",
                                                    usecase="Continual Learning", replay_size=7500,
                                                    random_selection=True)
        
        experiments.plotExperiment(experiment_name="CONTINUAL_LEARNING_DEFAULT",
                                   title="Continual Learning Default (CORe50 NICv2 - 391)")
        
    elif args.exp_sample_replacement_method:
        print("> Experiment: FIFO VS Random Selection - Buffer Sample Replacement")
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="FIFO_VS_RANDOM_SELECTION_BS_10000", usecase="FIFO",
                                                    replay_size=10000, random_selection=False)
        experiments.runRandomVSFIFOReplayExperiment(experiment_name="FIFO_VS_RANDOM_SELECTION_BS_10000",
                                                    usecase="RANDOM SELECTION", replay_size=10000,
                                                    random_selection=True)
        
        experiments.plotExperiment(experiment_name="FIFO_VS_RANDOM_SELECTION_BS_10000",
                                   title="FIFO VS Random Selection (CORe50 NICv2 - 391)")
    elif args.exp_hidden_layers:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        # We can alter the replay_size to experiment with different replay buffer sizes
        print("> Experiment: Number of Hidden Layers")
        # experiments.runRandomVSFIFOReplayExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_5000_old",
        #                                             replay_size=5000, random_selection=True)
        experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
                                                    usecase="RBS_5000_new",
                                                    replay_size=5000, random_selection=True, num_hidden_layers=0)
        # experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_20000_HL1",
        #                                             replay_size=20000, random_selection=True, num_hidden_layers=1)
        # experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_20000_HL2",
        #                                             replay_size=20000, random_selection=True, num_hidden_layers=2)
        # experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_20000_HL3",
        #                                             replay_size=20000, random_selection=True, num_hidden_layers=3)
        # experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_20000_HL4",
        #                                             replay_size=20000, random_selection=True, num_hidden_layers=4)
        # experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_20000_HL5",
        #                                             replay_size=20000, random_selection=True, num_hidden_layers=5)
        # experiments.runHiddenLayersExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
        #                                             usecase="RBS_20000_HL6",
        #                                             replay_size=20000, random_selection=True, num_hidden_layers=6)

        experiments.plotExperiment(experiment_name="HIDDEN_LAYERS_EXPERIMENTS",
                                   title="Hidden Layers Experiments (CORe50 NICv2 - 391)")
    else:
        print("> No valid experiment option provided")