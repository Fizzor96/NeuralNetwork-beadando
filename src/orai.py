from __future__ import print_function
import math, random, struct
from time import time


class NN:
    # available activation functions
    @staticmethod
    def activation_tanh(x):
        return math.tanh(x)

    @staticmethod
    def dactivation_tanh(x):
        return 1.0 - x ** 2

    @staticmethod
    def activation_sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def dactivation_sigmoid(x):
        return x * (1.0 - x)

    def __init__(self, NoNeurons, bias=True, activation="sigmoid"):
        self.NoNeurons = NoNeurons
        if bias:
            self.bias = 1
            self.NoNeurons[0] += 1
        else:
            self.bias = 0
        # wv = [numpy.random.random((nn[l+1], nn[l]))*2-1 for l in range(len(nn)-1)]
        # wv = [[random.random()*2-1 for _ in range(nn[l])] for _ in range(nn[l+1])]
        self.Weights = [
            [
                [random.random() * 2 - 1 for _ in range(self.NoNeurons[l])]
                for _ in range(self.NoNeurons[l + 1])
            ]
            for l in range(len(self.NoNeurons) - 1)
        ]
        if activation == "sigmoid":
            self.acti, self.dacti = NN.activation_sigmoid, NN.dactivation_sigmoid
        elif activation == "tanh":
            self.acti, self.dacti = NN.activation_tanh, NN.dactivation_tanh
        else:
            raise ValueError("Invalid or missing activation function")

    def forward(self, inp):
        self.Neurons = [inp + [1.0] * self.bias]
        # normalization if needed
        for l in range(len(self.NoNeurons) - 1):
            self.Neurons.append(
                [
                    self.acti(
                        sum(
                            [
                                self.Neurons[l][i] * self.Weights[l][j][i]
                                for i in range(self.NoNeurons[l])
                            ]
                        )
                    )
                    for j in range(self.NoNeurons[l + 1])
                ]
            )
        return self.Neurons[-1]

    def backprop(self, target):
        error = [target[j] - self.Neurons[-1][j] for j in range(self.NoNeurons[-1])]
        delta = [None for _ in range(len(self.NoNeurons) - 1)]
        for l in reversed(range(len(self.NoNeurons) - 1)):
            if l == len(self.NoNeurons) - 2:
                delta[l] = [
                    error[j] * self.dacti(self.Neurons[-1][j])
                    for j in range(self.NoNeurons[-1])
                ]
            else:
                delta[l] = [
                    sum(
                        [
                            delta[l + 1][j] * self.Weights[l + 1][j][i]
                            for j in range(self.NoNeurons[l + 2])
                        ]
                    )
                    * self.dacti(self.Neurons[l + 1][i])
                    for i in range(self.NoNeurons[l + 1])
                ]

            for i in range(self.NoNeurons[l]):
                for j in range(self.NoNeurons[l + 1]):
                    self.Weights[l][j][i] += 0.5 * delta[l][j] * self.Neurons[l][i]

        return sum([error[j] ** 2 for j in range(self.NoNeurons[-1])])

    def train(self, samples, maxSteps=1000, epsilon=0.1, log=10):
        steps = 0
        while steps < maxSteps:
            sumerr = 0.0
            for inp, target in random.sample(samples, len(samples)):
                self.forward(inp)
                sumerr += self.backprop(target)
            steps += 1
            if log > 0 and steps % log == 0:
                print(steps, sumerr)
            if sumerr <= epsilon:
                break

    def test(self, samples):
        sumerr = 0.0
        for inp, target in random.sample(samples, len(samples)):
            self.forward(inp)
            sumerr += sum(
                [
                    (target[j] - self.Neurons[-1][j]) ** 2
                    for j in range(self.NoNeurons[-1])
                ]
            )
        return sumerr


samples = []
for i in range(500):
    samples.append(i)

nt = NN([0, 1, 2])
nt.forward([1, 1, 1, 1, 1])
nt.train()
