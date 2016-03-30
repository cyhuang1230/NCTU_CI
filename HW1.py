import random
import math


def random_floats(count: int) -> [float]:
    random.seed()
    return [random.random() for _ in range(count)]


def calc_logistic_result(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_of_inputs, num_of_hidden_neuron, num_of_output_neuron = 1):

        # init
        self.hidden_layer = NeuralLayer(num_of_hidden_neuron, num_of_inputs)
        self.output_layer = NeuralLayer(num_of_output_neuron, num_of_hidden_neuron)

    def feed_forward(self, inputs: [float]):
        hidden_outputs = self.hidden_layer.feed_forward(inputs)
        self.output_layer.feed_forward(hidden_outputs)


class NeuralLayer:

    def __init__(self, neuron_count: int, input_count: int):
        self.neurons = []
        biases = random_floats(neuron_count)
        for i in range(neuron_count):
            new_neuron = Neuron(biases[i])
            new_neuron.weights = random_floats(input_count)
            self.neurons.append(new_neuron)

    def feed_forward(self, inputs: [float]) -> [float]:
        outputs = []
        for n in self.neurons:
            outputs.append(n.calc_output(inputs))
        return outputs


class Neuron:

    def __init__(self, bias: int):
        self.bias = bias
        self.len = -1
        self.weights = []
        self.inputs = []
        self.output = []

    def calc_output(self, inputs: [float]) -> float:
        assert self.weights.__len__() == inputs.__len__()
        self.inputs = inputs
        self.len = inputs.__len__()
        self.output = calc_logistic_result(self.calc_total_net_input())
        return self.output

    def calc_total_net_input(self) -> float:
        assert self.len != -1
        net_sum = [self.weights[i] * self.inputs[i] for i in range(self.len)]
        return sum(net_sum) + self.bias
