# Project Description

This project aims to provide a comprehensive study on the original [Paper](https://arxiv.org/abs/2202.01771). 

We proposed to experiment on other language models to evaluate methods discussed above. In addition to the models, we also propose a few ways of evaluation to further quantify how the decision making process can be improved. 

The project is still under development. 

# Data Generation

Steps for simplified environment generation and goals creating are written in ```generate_simple_VH```. You may refer to the file ```LAPKT_Action_Generation.ipynb``` for our way to generate expert demonstration for our project. The code leverages SIW+BFSF algorithm to search for optimal action list for two fully observing agents.

# Language Model on Deicsion Making

The implementation is based on the idea of [Pre-Trained Language Models for Interactive Decision-Making](https://arxiv.org/abs/2202.01771), yet we further expand the work to support multi-agent (duo-agent) prediction. It also grants the ability to predict a specific item instead of the item category. For example, if there were two apples with id <10>, <20>, differnt from our reference paper, our work gives the model the avility to predict which apple to choose. Training code can be found under 'LanguageModelCommanderTrain.ipynb'.

# VirtualHome Demo

We also provide a demostration for the VitualHome environment we are using. Pleas check ```unity_demo.ipynb``` for more details.

# Data

https://drive.google.com/drive/folders/1iPBoi_09QV5UEseg_eZjLq6DBne3NL6_?usp=sharing

# Trained Model

https://drive.google.com/file/d/1-L0TM1NTaxKxNePALauCnE78BvrrEiiT/view?usp=sharing
