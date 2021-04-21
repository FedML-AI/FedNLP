from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def eval_model(self):
        pass