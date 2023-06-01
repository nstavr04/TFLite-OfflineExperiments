# Continual Learning - Offline Experiments

This is a repository for experimenting with different models with the goal of performing well in continual learning scenarios while being lightweight and suitable for edge deployment e.g In an android application such as the tensorflow-lite model personalization demo app which enables on-device training.

## Structure of models

For the models we keep a structure of the base being a frozen model and the head being a series of layers that can be trained real time on-device.

## Latest methodology

- We evaluate our models on the CORe50 NICv2 - 391 dataset.
- So far, the model with the highest accuracy was a frozen MobileNetV2 for the base and a head which utilizes latent replay by moving some hidden layers from the base of the model to the head.
- We utilize random replay with a replay buffer which is between the base and the head of the model. 
- On the CORe50 NICv2 - 391, the model with 9 HL moved to the head managed to achieve 69.7% accuracy. 
