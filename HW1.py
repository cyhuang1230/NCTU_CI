import random
import math

file_names = ['cross200.txt', 'elliptic200.txt']


def readfile(file_name: str) -> ([], []):
    inputs = []
    outputs = []
    with open(file_name, 'r') as file:
        for line in file:
            (x, y, c) = filter((lambda x: x.__len__()), line.strip(' \n').split(' '))
            inputs.append([float(x), float(y)])
            outputs.append(int(c))
    return (inputs, outputs)


def random_floats(count: int) -> [float]:
    random.seed()
    return [random.random() for _ in range(count)]


def calc_logistic_result(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, dimension_of_input, num_of_hidden_neuron, num_of_output_neuron = 1):
        self.hidden_layer = NeuralLayer(num_of_hidden_neuron, dimension_of_input)
        self.output_layer = NeuralLayer(num_of_output_neuron, num_of_hidden_neuron)

    def feed_forward(self, inputs: [float]) -> None:
        hidden_outputs = self.hidden_layer.feed_forward(inputs)
        self.output_layer.feed_forward(hidden_outputs)

    def inspect(self):
        print('------')
        print('Hidden Layer: ', end='')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer: ', end='')
        self.output_layer.inspect()
        print('------')


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

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron ', n)
            print('  Bias:', self.neurons[n].bias)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight', w, ':', self.neurons[n].weights[w])


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


if __name__ == "__main__":
    inputs = []
    outputs = []
    for file_name in file_names:
        (input_, output_) = readfile(file_name)
        inputs += input_
        outputs += output_

    hidden_neuron = 32
    nn = NeuralNetwork(dimension_of_input=2, num_of_hidden_neuron=hidden_neuron, num_of_output_neuron=1)
    nn.inspect()
