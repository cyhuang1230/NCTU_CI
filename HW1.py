import random
import math
import matplotlib.pyplot as plt
import datetime
import sys


def readfile_single(file_name: str) -> ([], []):
    inputs = []
    with open(file_name, 'r') as file:
        for line in file:
            (x, y, c) = filter((lambda x: x.__len__()), line.strip(' \n').split(' '))
            inputs.append([float(x), float(y), 0 if int(c) == 2 else 1])
            # inputs.append([float(x), float(y), int(c)])
    return inputs


def random_floats(count: int) -> [float]:
    random.seed()
    return [random.random() for _ in range(count)]


def calc_logistic_result(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def log(s, end='\n', file=None):

    if file is None:
        global log_file
        if log_file is not None:
            file = log_file
        else:
            file = sys.stdout

    print(s, end=end, file=file)


class NeuralNetwork:

    MAX_EPOCH = -1
    OUTPUT_NEURON_INDEX = 0
    MOMENTUM_ALPHA = 0.8
    LEARNING_RATE = 0.01

    def __init__(self, dimension_of_input, num_of_hidden_neuron):

        # init variables
        self.epoch = 0
        self.data = []
        self.results = []
        self.results_validation = []
        self.early_stopping_epcohs = []
        assert NeuralNetwork.MAX_EPOCH != -1

        # init layers
        self.num_of_hidden_neuron = num_of_hidden_neuron
        self.num_of_output_neuron = 1
        self.hidden_layer = NeuralLayer(num_of_hidden_neuron, dimension_of_input)
        self.output_layer = NeuralLayer(self.num_of_output_neuron, num_of_hidden_neuron)

    def feed_forward(self, inputs: [float]) -> [float]:
        hidden_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_outputs)

    def train_online(self, training_set: [], validation_set=None) -> None:

        self.data = training_set

        try:
            # until stopping criterion is met
            while self.epoch < NeuralNetwork.MAX_EPOCH:

                print('This is epoch #%d/%d' %(self.epoch, NeuralNetwork.MAX_EPOCH), end='\r')

                # shuffle data_arr
                random.shuffle(training_set)

                for data in training_set:

                    # Feed forward
                    assert data.__len__() == 3
                    inputs = [data[0], data[1]]
                    target = data[2]
                    self.feed_forward(inputs)

                    # Output layer deltas
                    # ∂E/∂net = [-(target-out)]*[out(1-out)]
                    # only 1 element b/c only one output neuron
                    cur_neuron_output = self.output_layer.neurons[NeuralNetwork.OUTPUT_NEURON_INDEX].output
                    assert cur_neuron_output != -1.0
                    output_error_wrt_net = -(target-cur_neuron_output) * cur_neuron_output*(1-cur_neuron_output)

                    # Hidden layer deltas
                    hidden_error_wrt_net = [0] * self.num_of_hidden_neuron
                    for h in range(self.num_of_hidden_neuron):

                        # ∂E<total>/∂out<hidden>
                        error_wrt_hidden_out = output_error_wrt_net * \
                                               self.output_layer.neurons[NeuralNetwork.OUTPUT_NEURON_INDEX].weights[h]

                        cur_neuron_output = self.hidden_layer.neurons[h].output
                        hidden_error_wrt_net[h] = error_wrt_hidden_out * cur_neuron_output*(1-cur_neuron_output)

                    # Update output layer
                    cur_output_neuron = self.output_layer.neurons[NeuralNetwork.OUTPUT_NEURON_INDEX]
                    cur_output_neuron_weights = cur_output_neuron.weights.copy()
                    for w in range(self.output_layer.neurons[NeuralNetwork.OUTPUT_NEURON_INDEX].weights.__len__()):
                        # ∂E/∂weight = ∂E/∂net * out<hidden>
                        #                        ^^^^^^^^^^^^ == input
                        # for every weight connected to output neuron
                        error_wrt_weight = output_error_wrt_net * cur_output_neuron.inputs[w]

                        # update weight
                        cur_output_neuron.weights[w] -= NeuralNetwork.LEARNING_RATE * error_wrt_weight + \
                            NeuralNetwork.MOMENTUM_ALPHA * (cur_output_neuron_weights[w] - cur_output_neuron.weights_prev[w])

                    # update weight_prev
                    cur_output_neuron.weights_prev = cur_output_neuron_weights.copy()

                    # Update hidden layer
                    for h in range(self.num_of_hidden_neuron):

                        cur_hidden_neuron = self.hidden_layer.neurons[h]
                        cur_hidden_neuron_weights = cur_hidden_neuron.weights.copy()

                        for w in range(self.hidden_layer.neurons[h].weights.__len__()):
                            error_wrt_weight = hidden_error_wrt_net[h] * cur_hidden_neuron.inputs[w]

                            # update weight
                            cur_hidden_neuron.weights[w] -= NeuralNetwork.LEARNING_RATE * error_wrt_weight + \
                                NeuralNetwork.MOMENTUM_ALPHA * (cur_hidden_neuron_weights[w] - cur_hidden_neuron.weights_prev[w])

                        # update weight_prev
                        cur_hidden_neuron.weights_prev = cur_hidden_neuron_weights.copy()

                self.epoch += 1
                self.results.append(self.calculate_error())
                self.results_validation.append(self.calculate_error(validation_set))

                # print(self.calculate_error())

        finally:
            self.early_stopping_check()

    def calculate_error(self, data=None) -> float:

        if data is None:
            data = self.data

        total_mse = 0.0
        for t in data:
            x, y, target = t
            self.feed_forward([x, y])
            total_mse += self.output_layer.neurons[self.OUTPUT_NEURON_INDEX].calc_mse(target)

        return total_mse/data.__len__()

    def get_output(self, inputs) -> float:
        return self.feed_forward(inputs)[0]

    def early_stopping_check(self):
        len = self.results_validation.__len__()
        for idx in range(500, len-100):
            if self.results_validation[idx] > self.results_validation[idx+1]:
                continue
            should_append = True
            for i in range(idx-100 if idx-100 > 0 else 0, idx+100 if idx+100<len else len):
                if self.results_validation[i] < self.results_validation[idx]:
                    should_append = False
                    break
            if should_append:
                self.early_stopping_epcohs.append(idx)
                return

    def inspect(self):
        log('------')
        log('Hidden Layer: ', end='')
        self.hidden_layer.inspect()
        log('------')
        log('* Output Layer: ', end='')
        self.output_layer.inspect()
        log('------')


class NeuralLayer:

    def __init__(self, neuron_count: int, input_count: int):
        self.neurons = []
        biases = random_floats(neuron_count)
        for i in range(neuron_count):
            new_neuron = Neuron(biases[i])
            new_neuron.weights = random_floats(input_count)
            new_neuron.weights_prev = new_neuron.weights.copy()
            self.neurons.append(new_neuron)

    def feed_forward(self, inputs: [float]) -> [float]:
        outputs = []
        for n in self.neurons:
            outputs.append(n.calc_output(inputs))
        return outputs

    def inspect(self):
        log('%d neurons' %len(self.neurons))
        for n in range(len(self.neurons)):
            log(' Neuron %d' %n)
            log('  Bias: %f' %self.neurons[n].bias)
            for w in range(len(self.neurons[n].weights)):
                log('  Weight %d: %f' %(w, self.neurons[n].weights[w]))


class Neuron:

    def __init__(self, bias: int):
        self.bias = bias
        self.len = -1
        self.weights = []
        self.inputs = []
        self.output = -1.0
        self.weights_prev = [] # for Momentum

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

    def calc_mse(self, target_output) -> float:
        assert self.output != -1
        return 0.5 * ((target_output - self.output) ** 2)


if __name__ == "__main__":

    # hidden_neuron = int(sys.argv[1]) if sys.argv.__len__() >= 2 else 4
    hidden_neuron = 4
    NeuralNetwork.MAX_EPOCH = 10000
    NeuralNetwork.MOMENTUM_ALPHA = int(sys.argv[1]) if sys.argv.__len__() >= 2 else 0.8
    NeuralNetwork.LEARNING_RATE = int(sys.argv[2]) if sys.argv.__len__() >= 3 else 0.01
    # NeuralNetwork.LEARNING_RATE = 0.01

    file_names = ['cross200.txt', 'elliptic200.txt']
    inputs = []
    for file_name in file_names:
        input_ = readfile_single(file_name)
        inputs += input_

    # random.shuffle(inputs)
    validation_size = 40
    block_size = int(validation_size/2)
    training_set = inputs[block_size:-block_size]
    validation_set = inputs[:block_size] + inputs[-block_size:]

    training_set_len = training_set.__len__()
    validation_set_len = validation_set.__len__()

    # XOR
    # training_set = [
    #     [0, 0, 0],
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 0]
    # ]
    # validation_set = [
    #     [0, 0, 0.1],
    #     [0, 1, 1.1],
    #     [1, 0, 0.98],
    #     [1, 1, 0.01]
    # ]

    # momentum_alphas = [0, 0.2, 0.4, 0.6, 0.8]
    # momentum_alphas = [0.2, 0.4, 0.6, 0.8]
    momentum_alphas = [0.0]
    for ma in momentum_alphas:

        NeuralNetwork.MOMENTUM_ALPHA = ma
        lr = NeuralNetwork.LEARNING_RATE

        try:
            start_time = datetime.datetime.now()
            timestamp = start_time.timestamp()

            log_file_name = '%u_log.txt' %timestamp
            # sys.stdout = open('%u_log.txt' %timestamp, 'w+')
            log_file = open(log_file_name, 'w+')
            print('Writing to log file `%s`' %log_file_name)

            nn = NeuralNetwork(dimension_of_input=2, num_of_hidden_neuron=hidden_neuron)
            nn.train_online(training_set, validation_set)
            results, validation = nn.results, nn.results_validation
            end_time = datetime.datetime.now()
            log("(%u) Total %d epochs(%.1falpha, %.2fLR), time elapsed: %s"
                %(timestamp, lr, ma, nn.epoch, end_time-start_time))
            log('results_validation:')
            for idx, rv in enumerate(nn.results_validation):
                log("%d => %f" %(idx, rv))
            # print(results)

            figure_filename = '%u-%dhidden-%.2flr-%.1fMA-%dtraining-%dvalidation' \
                              % (timestamp, hidden_neuron, lr, ma, training_set_len, validation_set_len)
            figure_epoch_filename = figure_filename + '-epoch'
            figure_map_filename = figure_filename + '-map'

            figure_title = '%dHN, %.2fLR, %dTS, %dVS, %.1fMA' \
                           % (hidden_neuron, lr, training_set_len, validation_set_len, ma)
            y_min = min(min(results)*0.98, min(validation)*0.98)
            y_max = max(max(results)*1.01, max(validation)*1.01)
            results_len = results.__len__()
            x = list(range(results_len))  # use len instead of epoch to avoid error when interrupt
            fig_epoch = plt.figure(1)
            plt.plot(x, results, 'r', label='Training set')
            # plt.axis([-1, results.__len__(), min(results) * 0.98, max(results) * 1.02])
            plt.plot(x, validation, 'g', label='Validation set')
            plt.axis([max(-10, -0.05*results_len), results_len, y_min, y_max])
            plt.xlabel('epoch')
            plt.ylabel('J')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fancybox=True, shadow=True)
            fig_epoch.canvas.set_window_title(figure_epoch_filename)
            ax = fig_epoch.add_subplot(111)
            ax.set_title(figure_title)
            print(nn.early_stopping_epcohs)
            for es in nn.early_stopping_epcohs:
                plt.axvline(x=es)
                plt.annotate('%d' %es, xy=(es+10, (y_min+y_max)/2))

            plt.savefig(figure_epoch_filename + '.png')

            # show map
            fig_map = plt.figure(2)
            fig_map.canvas.set_window_title(figure_map_filename)
            ax = fig_map.add_subplot(111)
            plt.grid()
            plt.axis([-1, 1, -1, 1])
            training_correct_count = 0
            for i in training_set:
                cur_output = nn.get_output([i[0], i[1]])
                target_class = i[2]
                cur_class = 0 if abs(cur_output - 1) > abs(cur_output - 0) else 1
                # cur_class = 2 if abs(cur_output-1) > abs(cur_output-2) else 1
                is_correct = (cur_class == i[2])
                if is_correct:
                    training_correct_count += 1
                # log('(%f,%f) => cur_output = %f, cur_class = %d, target_class = %d, is_correct = %d'
                #       % (i[0], i[1], cur_output, cur_class, target_class, is_correct))
                plt.plot(i[0], i[1], ('g' if is_correct else 'r') + ('x' if target_class == 0 else '+'))

            log('--- validation set ---')
            validation_correct_count = 0
            for i in validation_set:
                cur_output = nn.get_output([i[0], i[1]])
                target_class = i[2]
                cur_class = 0 if abs(cur_output - 1) > abs(cur_output - 0) else 1
                # cur_class = 2 if abs(cur_output-1) > abs(cur_output-2) else 1
                is_correct = (cur_class == i[2])
                if is_correct:
                    validation_correct_count += 1
                # log('(%f,%f) => cur_output = %f, cur_class = %d, target_class = %d, is_correct = %d'
                #       % (i[0], i[1], cur_output, cur_class, target_class, is_correct), )
                plt.plot(i[0], i[1], ('g' if is_correct else 'r') + ('*' if target_class == 0 else 'o'))

            ax.set_title('%dHN, %.2fLR, %d/%d(%.2f%%)TS, %d/%d(%.2f%%)VS, %.1fMA'
                         % (hidden_neuron, lr,
                            training_correct_count, training_set_len, 100*training_correct_count/training_set_len,
                            validation_correct_count, validation_set_len, 100*validation_correct_count/validation_set_len,
                            ma))

            nn.inspect()
            plt.savefig(figure_map_filename + '.png')
            # fig_epoch.show()
            # plt.show()
            fig_epoch.clear()
            fig_map.clear()

        finally:
            log_file.close()
