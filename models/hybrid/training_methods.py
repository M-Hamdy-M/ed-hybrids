# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

from enum import Enum

class TrainingMethods(Enum):
    CFS = 0
    BP = 1
    @classmethod
    def get_methods(cls):
        return [method.name for method in cls]
    def __str__(self):
        return self.name
    
    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value