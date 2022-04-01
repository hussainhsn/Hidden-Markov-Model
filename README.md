# Hidden-Markov-Model

My code is included in both hmm.py and nlputil.py. The reason I also include the latter is because there was
an issue reading the files so I added "encoding="utf8"" to fix it. All the code for the learning and testing are 
in HMM.py. The program runs really slow for large data amounts and high number of states.

I have set up the source code per the instructions that were in the main file originally. I rescattered the instructions
to indicate which part is being applied.

I added an arg that asks the user about the sample size for learning (default = 1000) and wrote a function that
returns a random collection of samples of such a size to train with. I decided that 90% of that data will be for the training
and the rest 10% will be for the testing.


Available args are:
--train_path for folder of data
--max_iters for max iterations if not fully converged
--hidden_states number of hidden states
--sample_size amount of random data(from the path) to train with
--graphing show figure of final LLs (input is 0 or 1)
