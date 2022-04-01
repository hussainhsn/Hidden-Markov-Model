# File: self.py
# Purpose:  Starter code for building and training an HMM in CSC 246.


import argparse, copy
from nlputil import *   # utility methods for working with text
from matplotlib import pyplot as plt

# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size
#
# Note: You may want to add fields for expectations.
class HMM:
    # __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.

    # I changed the parameters' names to what it looked like in the books
    def __init__(self, num_states, vocab_size, vocab):
        self.N = num_states
        self.M = vocab_size
        self.pi = np.array(1.0/num_states).repeat(num_states)
        self.A = np.random.rand(num_states, num_states)
        sum_of_rows = self.A.sum(axis=1)
        self.A = self.A / sum_of_rows[:, np.newaxis]
        self.B = np.random.uniform(size = self.N * vocab_size).reshape((self.N, vocab_size))
        sum_of_rows = self.B.sum(axis=1)
        self.B = self.B / sum_of_rows[:, np.newaxis]

    def __repr__(self):
        string = "N: " + str(self.N) + "\nM: " + str(self.M) + "\nA: " + str(self.A) + "\nB: " + str(self.B) + "\nPi: " + str(self.pi)
        return string

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset):
        matrices = []
        for sample in dataset:
            matrices.append(self.loglikelihood_helper(sample))
        return matrices

    # return the loglikelihood for a single sequence (numpy matrix)
    def loglikelihood_helper(self, sample):
        return self.forward(sample)[0]

    def forward(self, sample):
        T = len(sample)
        c = np.zeros(T)

        

        try:
            while True:  # I made this exception in case of a bad sample
                Alpha = np.zeros((self.N, T))
                Alpha[:, 0] = self.pi * self.B[:, sample[0]]
                c[0] = 1.0 / np.sum(Alpha[:, 0])
                Alpha[:, 0] = c[0] * Alpha[:, 0]
                for t in range(1, T):
                    Alpha[:, t] = np.dot(Alpha[:, t-1], self.A) * self.B[:, sample[t]]
                    c[t] = 1.0 / np.sum(Alpha[:, t])
                    Alpha[:, t] = Alpha[:, t] * c[t]
                log_Prob_Obs = -(np.sum(np.log(c)))
                return (log_Prob_Obs, Alpha, c)
        except IndexError:
            return (-1, -1, -1)

    def backward(self, sample, c):  # c is the scaling constant array
        T = len(sample)
        Beta = np.zeros((self.N, T))
        Beta[:, T-1] = 1.0
        Beta[:, T-1] = Beta[:, T-1] * c[T-1]
        for t in reversed(range(T-1)):
            Beta[:, t] = np.dot(self.A, (self.B[:, sample[t+1]] * Beta[:, t+1]))
            Beta[:, t] = Beta[:, t] * c[t]
        return Beta

    def em_step(self, dataset):
        K = len(dataset)  # number of samples
        start = time()
        start_epoch = time()
        LL = 0
        E_si_all = np.zeros(self.N)  # Vector for expectation of being in state si over all samples
        E_si_all_T1 = np.zeros(self.N)  # Same as before but only until T-1
        E_si_sj_all = np.zeros((self.N, self.N))  # Matrix for expectation of transitioning from si to sj over all samples
        E_si_sj_all_T1 = np.zeros((self.N, self.N))
        E_si_t0_all = np.zeros(self.N)  # Expectation of initially being in si over all samples
        counter = 0
        for sample in dataset:
            counter += 1
            if counter % 100 == 0:
                print (counter, "samples processed")
            sample = list(sample)
            if sample:  # Some samples were empty so this throws them away
                x, y, z = self.forward(sample)  # Compute forward (probability, log-likelihood,scaling) values
                if not (isinstance(z, int)):  # If there were no indexing issues, use the values
                    log_Prob_sample, Alpha, c = x, y, z
                    Beta = self.backward(sample, c)
                    LL += log_Prob_sample                                         # Update log-likelihood
                    T = len(sample)
                    w_k = 1.0 / -(log_Prob_sample + np.log(len(sample)))  # For weight update

                    Gamma0 = Alpha * Beta  # Compute Gamma
                    Gamma = Gamma0 / Gamma0.sum(0)  # Normalize

                    E_si_t0_all += w_k * Gamma[:, 0]  # Update

                    E_si_all += w_k * Gamma.sum(1)
                    E_si_all_T1 += w_k * Gamma[:, :T-1].sum(1)

                    
                    Xi = np.zeros((self.N, self.N, T-1))  # Xi[ i,j,t ] = P(q_t = S_i, q_t+1 = S_j|HMM, Observation )
                    for t in range(T-1):
                        for i in range(self.N):
                            Xi[i, :, t] = Alpha[i, t] * self.A[i, :] * self.B[:, sample[t+1]] * Beta[:, t+1]

                    E_si_sj_all += w_k * Xi.sum(2)
                    E_si_sj_all_T1 += w_k * Xi[:, :, :T-1].sum(2)

        # Update Pi
        E_si_t0_all = E_si_t0_all / np.sum(E_si_t0_all)
        self.pi = E_si_t0_all

        # Update A
        A_bar = np.zeros((self.N, self.N))
        for i in range(self.N):
            A_bar[i, :] = E_si_sj_all_T1[i, :]/E_si_all_T1[i]
        self.A = A_bar

        return LL  # This is returned in case it was needed, but forward() also does the job


def sampler(dataset, size):
    arr = []
    for i in range(size):
        x = np.random.randint(0, len(dataset))
        arr.append(dataset[x])
    return arr


def graphing(LLs):
        plt.figure()
        plt.title("Log Likelihood within Each Iteration")
        plt.xlabel("Epochs")
        plt.ylabel(r"$\log( P ( O | \lambda ) )$")
        plt.plot(LLs,  label="Training data")
        plt.show()
        plt.savefig("plot")


def main():
    
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--dev_path', default=None, help='Path to the development data directory.')
    parser.add_argument('--max_iters', type=int, default=15, help='The maximum number of EM iterations (default 15)')
    parser.add_argument('--hidden_states', type=int, default=5, help='The number of hidden states to use. (default 5)')
    parser.add_argument('--sample_size', type=int, default=1000, help='The number of samples to use. (default 1000)')
    parser.add_argument('--graphing', type=int, default=False, help='The number of samples to use. (default False)')
    args = parser.parse_args()
    

    # 1. load training and testing data into memory
    #paths = ['\\Users\\HUSSA\\Documents\\School\\CSC 246\\Projects\\Project 3\\imdbFor246\\train\\pos', '\\Users\\HUSSA\\Documents\\School\\CSC 246\\Projects\\Project 3\\imdbFor246\\train\\neg']
    print("Begin loading vocab... ", end='')
    sys.stdout.flush()
    paths = [args.train_path]

    # 2. build vocabulary using training data ONLY
    begin = time()
    vocab = build_vocab_chars(paths)
    end = time()
    print('done in', end-begin, 'seconds.  Found', len(vocab), 'unique tokens.')
    print('Begin loading all data and converting to ints... ', end='')
    sys.stdout.flush()
    begin = time()
    data = load_and_convert_data_chars_to_ints(paths, vocab)
    end = time()
    print('done in', end-begin, 'seconds.')

    # 3. instantiate an HMM with given number of states -- initial parameters can
    #    be random or uniform for transitions and inital state distributions,
    #    initial emission parameters could bea uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    #max_iters = 50
    sample_size = args.sample_size

    train = sampler(data, int(sample_size*0.9))
    test = sampler(data, int(sample_size*0.1))

    #hidden_states,
    vocab_size = len(vocab)
    Hmm = HMM(args.hidden_states, vocab_size, vocab)

    # 4. output initial loglikelihood on training data and on testing data

    train_sample = sum(Hmm.loglikelihood(train))
    print("Start Likelihood for training:", train_sample)
    test_sample = sum(Hmm.loglikelihood(test))
    print("Start Likelihood for testing:", test_sample)

    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message

    epsilon = 500
    train_LLs = []
    test_LLs = []
    start0 = time()
    for i in range(args.max_iters):
        print ("Started training for epoch", i+1)
        start = time()
        Hmm.em_step(train)
        print("Training time for epoch", str(i+1) + ":", (time() - start), "seconds.")
        if i > 2:
            if abs(test_LLs[i-1] - test_LLs[i-2]) < epsilon:
                print("Converged at epoch", i+1)
                break
        train_sample = sum(Hmm.loglikelihood(train))
        train_LLs.append(train_sample)
        test_sample = sum(Hmm.loglikelihood(test))
        test_LLs.append(test_sample)

    print("Total time for all epochs ran:", (time() - start0), "seconds.")
    print("Final Likelihood for training:", train_sample)
    print("Final Likelihood for testing:", test_sample)
    if graphing:
        graphing(test_LLs)

if __name__ == '__main__':
    main()
