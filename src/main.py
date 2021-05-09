import sys
import numpy as np
import matplotlib as matp

a1 = [
    [50, 4, 72, 2, 25],
    [73, 10, 9, 3, 6],
    [30, 40, 45, 92, 61],
    [13, 27, 34, 61, 87],
    [44, 66, 1, 98, 55],
]


class Layer:
    @staticmethod
    def acti(x, n_inputs):
        return x / n_inputs

    def __init__(self, n_inputs, n_neurons):
        self.inputs_num = n_inputs
        # print(self.inputs_num)
        self.weights = (
            np.random.randint(80, 121, (n_inputs, n_neurons), dtype=int) / 100
        )

        # if n_inputs >= 2:
        #     for i in range(len(self.weights)):
        #         self.weights[i][1] = 0.8
        # print("\nself.weights = \n", self.weights, "\n")

        self.biases = np.zeros((n_neurons,), dtype=float)
        # self.biases[1] = self.biases[1] - 20
        # print("\nself.biases = \n", self.biases, "\n ")

    def forward(self, inputs):
        self.output = self.acti(
            np.dot(inputs, self.weights) + self.biases, self.inputs_num
        )


l1 = Layer(5, 3)
l1.forward(a1)
print(l1.output)

l2 = Layer(3, 1)
l2.forward(l1.output)
print(l2.output)
