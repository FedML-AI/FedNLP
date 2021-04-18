# Trainer
Trainer is a bridge to connect model and data. In order to integrate FedML and NLP, we desgin two trainer here.

The first one is federated trainer. This trainer inherite a class called `ModelTrainer` in FedML module. Users have to implment some required fuctions so that they can exploit the federated algorithm correctly.

The second one is NLP trainer. Note that we pass this trainer as an argument to federated trainer. In NLP trainer, users can design their training process and evaluation process as same as what they have done in centralized training. Federated trainer will drive the NLP trainer to finish all stuff. Users do not need to care about the distributed issues.