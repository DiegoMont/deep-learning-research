import matplotlib.pyplot as plt


class MetricHistory:
    def __init__(self, metric: str, color: str):
        self.train: list[float] = []
        self.validation: list[float] = []
        self.__metric = metric
        self.__color = color
        self.__train_epoch = 0
        self.__val_epoch = 0

    def compute(self, train_steps: int, validation_steps: int):
        self.__train_epoch /= train_steps
        self.__val_epoch /= validation_steps
        self.train.append(self.__train_epoch)
        self.validation.append(self.__val_epoch)
        self.__train_epoch = 0
        self.__val_epoch = 0

    def register_epoch_value(self, train_value, validation_value):
        self.__train_epoch += train_value
        self.__val_epoch += validation_value

    def plot(self):
        epochs = range(1, len(self.train) + 1)
        plt.figure(figsize=(10, 8))
        plt.plot(epochs, self.train, f"{self.__color}o", label=f"Training {self.__metric}")
        plt.plot(epochs, self.validation, f"{self.__color}--", label=f"Validation {self.__metric}")
        plt.title(f"Training {self.__metric}")
        plt.legend()
        plt.grid()
        plt.show()
