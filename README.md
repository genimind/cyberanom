# cyberanom
Cyber Anomaly Detection using RNN Language model

The work in this project is based on the paper ["Recurrent Neural Network Language Models for Open Vocabulary Event-Level Cyber Anomaly Detection"](https://arxiv.org/abs/1712.00557)

The paper is referencing this LANL security data ["Comprehensive, Multi-Source Cyber-Security Events"](https://csr.lanl.gov/data/cyber1/) only the proc.txt and redteam.txt are used in the paper.

The project consists of different experimental code:
* experiment1: The implementation is loosly based on the safekit project [code.](https://github.com/pnnl/safekit) 
* experiment2: Our own implementation using Keras model and Tensorflow Eager execution.


### ToDo
* Extend the language model to include attention as described in the paper ["Recurrent Neural Network Attention Mechanisms for Interpretable System Log Anomaly Detection"](https://arxiv.org/abs/1803.04967)
* Experiment with an encoder/decoder language models
* Running system with multiple tasks to process user logs using RISELab Ray
