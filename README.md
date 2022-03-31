# Project Description

This project aims to provide a comprehensive study on the original [Paper](https://arxiv.org/abs/2202.01771). 

We proposed to experiment on other language models to evaluate methods discussed above. In addition to the models, we also propose a few ways of evaluation to further quantify how the decision making process can be improved. 

The project is still under development. 

# Data Generation

You may refer to the file ```Data_Generation.ipynb``` for our way to generate expert demonstration for our project. The code was developed in Google Colab with GPU support. It leverages the implementaion of zero-shot learning technique proposed in (Language Models as Zero-Shot Planners:
Extracting Actionable Knowledge for Embodied Agents)[https://arxiv.org/pdf/2201.07207.pdf]. (TODO) Our implementation further supports VirtualHome validation on generated data in order to ensure the fidelity of the data during training. Details can be found in the notebook.

# Language Model on Deicsion Making

You may refer to our implementation of [Pre-Trained Language Models for Interactive Decision-Making](https://arxiv.org/abs/2202.01771).(TODO) We further support more experiment options with different model and our customized evaluation metircs:

### Success Rate: We first consider the simple success rate of different Language Models using different number of training demos measured by

$$S = \frac{\text{number of successes}}{\text{number of demos}}$$

When using this evaluation metric, we only consider if the model successfully completes the goal and ignore the specific trajectory. If we set the goal as “put the blue key next to the purple ball”, an ideally efficient model would directly start to look for the blue key,  pick it up, walk to the purple ball and place the key next to it. However, a model that is not well trained may waste time wandering around the room, but still successfully find the key and drop it in the right position. The success rate would treat these two events equal and show no preference.

### Trajectory Loss: In reality, not only would we care about whether the model successfully completes the task, but the time it spends, the trajectory it chooses, and whether it does irrelevant things are also the key elements for evaluation. We hope to introduce other metrics to include these concerns. In contrast to success rate, the loss of a model can simultaneously take if the model completes the task and if a model completes it in an efficient way into account. We define the loss of a model for a single demo i as

$$loss_i = 1 - \mathbbm{1}_{success}{[i]} \cdot \frac{ODS_i}{ADS_i}$$

where $ODS_i$ and $ADS_i$ are the number of optimal decision steps and the number of actual decision steps. The optimal number of steps can be predetermined. Thus, whenever the model fails the task, the loss will be 1. If the task is completed, then the loss will be smaller as the model uses less decision steps. The minimum value of $loss_i$ would be 0. We aggregate the single loss over all demos to get the loss of the model as

$$L = \frac{1}{N}\displaystyle\sum_{i=1}^{N} loss_i$$


# VirtualHome Demo

We also provide a demostration for the VitualHome environment we are using. Pleas check ```unity_demo.ipynb``` for more details. 
