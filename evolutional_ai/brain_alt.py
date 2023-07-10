import math
import numpy as np
import os

from . import utils


def activation(x):
    return 2.0/(1.0+np.exp(-2.0*x))-1.0

def _activation(x):
    """ computes sigmoid of x """
    return (1.0/(1.0 + np.exp(-x)))



def mutate(genomes):
    """ mutates a genome and returns the mutated one """
    # mutate some random indices by a certain amount

    # only mutate with a certain probability
    if np.random.random() < 0.5:
        return genomes

    genome_1 = genomes[0].flatten()
    l = len(genome_1)
    indices = np.random.choice(l, int(l * utils.MUTATION_PROB), replace=False)
    # mutate
    genome_1[indices] += np.random.random() - 0.5

    genome_2 = genomes[1].flatten()
    l = len(genome_2)
    indices = np.random.choice(l, int(l * utils.MUTATION_PROB), replace=False)
    # mutate
    genome_2[indices] += np.random.random() - 0.5

    return (np.reshape(genome_1,genomes[0].shape),np.reshape(genome_2,genomes[1].shape))

def cross_over(m1_vec, m2_vec):
    """ performs cross over between two genomes """
    # interchange a random amount of both vectors
    m1_1 = m1_vec[0].flatten()
    m1_2 = m1_vec[1].flatten()
    m2_1 = m2_vec[0].flatten()
    m2_2 = m2_vec[1].flatten()
    l = len(m1_1)
    indices = np.random.choice(l, np.random.randint(0, int(l * utils.CROSS_OVER_PROB)), replace=False)

    temp = m1_1[indices][:]
    m1_1[indices] = m2_1[indices][:]
    m2_1[indices] = temp[:]

    l = len(m1_2)
    indices = np.random.choice(l, np.random.randint(0, l), replace=False)

    temp = m1_2[indices][:]
    m1_2[indices] = m2_2[indices][:]
    m2_2[indices] = temp[:]

    return (np.reshape(m1_1,m1_vec[0].shape),np.reshape(m1_2,m1_vec[1].shape)),(np.reshape(m2_1,m2_vec[0].shape),np.reshape(m2_2,m2_vec[1].shape))

class Model:
    """ holds the brain of an agent """
    def __init__(self):
        # create weights
        inp_size = len(utils.CAR_INPUT_ANGLES.split(",")) + 1

        self.weights = np.random.random((int(utils.HIDDEN_LAYERS)+1,inp_size))
        self.activation_weights = np.random.random((2,int(utils.HIDDEN_LAYERS)))
        #print("")


    def get_weights(self):
        """ returns a vector with all weights """
        # returns a huge vector of all weights

        # return self.weights, but flattened
        return self.weights,self.activation_weights

    def save_to_file(self, filename):
        # TODO add layer size
        np.save(os.path.join(utils.MODELS_PATH, filename), self.weights)

    def load_from_file(self, filename):
        # TODO add layer size
        fp = filename + "_head.npy"
        self.weights =np.load(os.path.join(utils.MODELS_PATH, fp))
        fp = filename + "_class.npy"
        self.activation_weights =np.load(os.path.join(utils.MODELS_PATH, fp))

    def set_weights(self, w):
        """ set weights from a vector """
        self.weights = w[0]
        self.activation_weights = w[1]
                

    def predict(self, inp):
        # add a bias term to inp
        inp_layer = [activation(inp[i]*self.weights[0][i]) for i in range(len(inp))]

        hidden_layers = [np.sum([inp_layer[i]*self.weights[x+1][i] for i in range(len(inp_layer))]) for x in range(int(utils.HIDDEN_LAYERS))]

        out_1 = np.sum([activation(hidden_layers[x])*self.activation_weights[0][x] for x in range(len(hidden_layers))])
        out_2 = np.sum([activation(hidden_layers[x])*self.activation_weights[1][x] for x in range(len(hidden_layers))])
        
        return np.array([[activation(out_1)],[activation(out_2)]])