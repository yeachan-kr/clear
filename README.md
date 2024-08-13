# CleaR

This repository is about CleaR long paper: [Towards Robust and Generalized Parameter-Efficient Fine-Tuning for Noisy Label Learning](https://aclanthology.org/2024.acl-long.322/) published in ACL 2024. 

### Requirements
 - Python 3
 - Transformers 4.27.2
 - Numpy 
 - pytorch

### Clean Routing
```python
bash train.sh
```

 - ```--dataset```: Train dataset used for training.
 - ```--lr```: Set the learning rate.
 - ```--epochs```: Set the number of epochs. 
 - ```--batch_size```: Set the batch size for conducting at once. 
 - ```--warm_up``` : Set the warm-up epoch
 - ```--alg```: PEFT routing strategy. Choose PEFT routing from : routing_adapter, routing_lora, routing_prefix, routing_bitfit, none.
 - ```--adapter```: PEFT routing strategy. Choose PEFT routing from : routing_adapter, routing_lora, routing_prefix, routing_bitfit, none.

## Contact Info 
For help or issues using CleaR, please submit a GitHub issue. 

For personal communication related to CleaR, please contact Yeachan Kim```<yeachan@korea.ac.kr>``` or Junho Kim ```<monocrat@korea.ac.kr>```
