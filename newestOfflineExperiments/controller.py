from experiments import Experiments
import argparse
import numpy as np
import tensorflow as tf

SEED = 1

np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_RBS_3000", action="store_true")
    parser.add_argument("--exp_RBS_5000", action="store_true")
    parser.add_argument("--exp_RBS_7500", action="store_true")
    args = parser.parse_args()
    experiments = Experiments()

    # Experimenting with different replay buffer sizes.
    # For each experiment we move the replay layer 1 hidden layer backwards (We move hidden layers from MobileNetV2 to the head of the model and make them trainable)

    if args.exp_RBS_3000:
        print("> Experiment: RBS_3000")

        # experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
        #                                             usecase="RBS_3000_HL0",
        #                                             replay_size=3000, num_hidden_layers=0)
        
        # experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
        #                                             usecase="RBS_3000_HL1",
        #                                             replay_size=3000, num_hidden_layers=1)
        
        # experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
        #                                             usecase="RBS_3000_HL2",
        #                                             replay_size=3000, num_hidden_layers=2)
        
        # experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
        #                                             usecase="RBS_3000_HL3",
        #                                             replay_size=3000, num_hidden_layers=3)
        
        # experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
        #                                             usecase="RBS_3000_HL4",
        #                                             replay_size=3000, num_hidden_layers=4)
        
        # experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",    
        #                                             usecase="RBS_3000_HL5",
        #                                             replay_size=3000, num_hidden_layers=5)
        
        experiments.plotExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                   title="RBS_3000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")
        
    elif args.exp_RBS_5000:
        print("> Experiment: RBS_5000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL0",
                                                    replay_size=5000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL1",
                                                    replay_size=5000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL2",
                                                    replay_size=5000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL3",
                                                    replay_size=5000, num_hidden_layers=3)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL4",
                                                    replay_size=5000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL5",
                                                    replay_size=5000, num_hidden_layers=5)
        
        # experiments.plotExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
        #                            title="RBS_5000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    elif args.exp_RBS_7500:
        print("> Experiment: RBS_7500")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL0",
                                                    replay_size=7500, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL1",
                                                    replay_size=7500, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL2",
                                                    replay_size=7500, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL3",
                                                    replay_size=7500, num_hidden_layers=3)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL4",
                                                    replay_size=7500, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL5",
                                                    replay_size=7500, num_hidden_layers=5)
        
        # experiments.plotExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
        #                            title="RBS_7500_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    else:
        print("> No valid experiment option provided")