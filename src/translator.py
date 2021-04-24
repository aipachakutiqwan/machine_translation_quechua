
from termcolor import colored
import random
import numpy as np

import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training

#get_ipython().system('pip list | grep trax')

# Get generator function for the training set
# This will download the train dataset if no data_dir is specified.
train_stream_fn = trax.data.TFDS('opus/medical',
                                 data_dir='./data/',
                                 keys=('en', 'de'),
                                 eval_holdout_size=0.01, # 1% for eval
                                 train=True)

# Get generator function for the eval set
eval_stream_fn = trax.data.TFDS('opus/medical',
                                data_dir='./data/',
                                keys=('en', 'de'),
                                eval_holdout_size=0.01, # 1% for eval
                                train=False)


train_stream = train_stream_fn()
print(colored('train data (en, de) tuple:', 'red'), next(train_stream))
print()

eval_stream = eval_stream_fn()
print(colored('eval data (en, de) tuple:', 'red'), next(eval_stream))



# global variables that state the filename and directory of the vocabulary file
VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = 'data/'

# Tokenize the dataset.
tokenized_train_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(train_stream)
tokenized_eval_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(eval_stream)



# Append EOS at the end of each sentence.
EOS = 1

# generator helper function to append EOS to each sentence
def append_eos(stream):
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)

# append EOS to the train data
tokenized_train_stream = append_eos(tokenized_train_stream)

# append EOS to the eval data
tokenized_eval_stream = append_eos(tokenized_eval_stream)


# Filter too long sentences to not run out of memory.
# length_keys=[0, 1] means we filter both English and German sentences, so
# both much be not longer that 256 tokens for training / 512 for eval.
filtered_train_stream = trax.data.FilterByLength(
    max_length=256, length_keys=[0, 1])(tokenized_train_stream)
filtered_eval_stream = trax.data.FilterByLength(
    max_length=512, length_keys=[0, 1])(tokenized_eval_stream)

# print a sample input-target pair of tokenized sentences
train_input, train_target = next(filtered_train_stream)
print(colored(f'Single tokenized example input:', 'red' ), train_input)
print(colored(f'Single tokenized example target:', 'red'), train_target)


# Setup helper functions for tokenizing and detokenizing sentences
def tokenize(input_str, vocab_file=None, vocab_dir=None):
    """Encodes a string to an array of integers

    Args:
        input_str (str): human-readable string to encode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file
  
    Returns:
        numpy.ndarray: tokenized version of the input string
    """
    
    # Set the encoding of the "end of sentence" as 1
    EOS = 1
    
    # Use the trax.data.tokenize method. It takes streams and returns streams,
    # we get around it by making a 1-element stream with `iter`.
    inputs =  next(trax.data.tokenize(iter([input_str]),
                                      vocab_file=vocab_file, vocab_dir=vocab_dir))
    
    # Mark the end of the sentence with EOS
    inputs = list(inputs) + [EOS]
    
    # Adding the batch dimension to the front of the shape
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    
    return batch_inputs


def detokenize(integers, vocab_file=None, vocab_dir=None):
    """Decodes an array of integers to a human readable string

    Args:
        integers (numpy.ndarray): array of integers to decode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file
  
    Returns:
        str: the decoded sentence.
    """
    
    # Remove the dimensions of size 1
    integers = list(np.squeeze(integers))
    
    # Set the encoding of the "end of sentence" as 1
    EOS = 1
    
    # Remove the EOS to decode only the original tokens
    if EOS in integers:
        integers = integers[:integers.index(EOS)] 
    
    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)



# As declared earlier:
# VOCAB_FILE = 'ende_32k.subword'
# VOCAB_DIR = 'data/'

# Detokenize an input-target pair of tokenized sentences
print(colored(f'Single detokenized example input:', 'red'), detokenize(train_input, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR))
print(colored(f'Single detokenized example target:', 'red'), detokenize(train_target, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR))
print()

# Tokenize and detokenize a word that is not explicitly saved in the vocabulary file.
# See how it combines the subwords -- 'hell' and 'o'-- to form the word 'hello'.
print(colored(f"tokenize('hello'): ", 'green'), tokenize('hello', vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR))
print(colored(f"detokenize([17332, 140, 1]): ", 'green'), detokenize([17332, 140, 1], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR))


# Bucketing to create streams of batches.

boundaries =  [8,   16,  32, 64, 128, 256, 512]
batch_sizes = [256, 128, 64, 32, 16,    8,   4,  2]

# Create the generators.
train_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes,
    length_keys=[0, 1]  # As before: count inputs and targets to length.
)(filtered_train_stream)

eval_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes,
    length_keys=[0, 1]  # As before: count inputs and targets to length.
)(filtered_eval_stream)

# Add masking for the padding (0s).
train_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(train_batch_stream)
eval_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_stream)


input_batch, target_batch, mask_batch = next(train_batch_stream)

# let's see the data type of a batch
print("input_batch data type: ", type(input_batch))
print("target_batch data type: ", type(target_batch))

# let's see the shape of this particular batch (batch length, sentence length)
print("input_batch shape: ", input_batch.shape)
print("target_batch shape: ", target_batch.shape)


# pick a random index less than the batch size.
index = random.randrange(len(input_batch))

# use the index to grab an entry from the input and target batch
print(colored('THIS IS THE ENGLISH SENTENCE: \n', 'red'), detokenize(input_batch[index], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR), '\n')
print(colored('THIS IS THE TOKENIZED VERSION OF THE ENGLISH SENTENCE: \n ', 'red'), input_batch[index], '\n')
print(colored('THIS IS THE GERMAN TRANSLATION: \n', 'red'), detokenize(target_batch[index], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR), '\n')
print(colored('THIS IS THE TOKENIZED VERSION OF THE GERMAN TRANSLATION: \n', 'red'), target_batch[index], '\n')



# Neural Machine Translation with Attention

def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """ Input encoder runs on the input sentence and creates
    activations that will be the keys and values for attention.
    
    Args:
        input_vocab_size: int: vocab size of the input
        d_model: int:  depth of embedding (n_units in the LSTM cell)
        n_encoder_layers: int: number of LSTM layers in the encoder
    Returns:
        tl.Serial: The input encoder
    """
    
    # create a serial network
    input_encoder = tl.Serial( 
        
        # create an embedding layer to convert tokens to vectors
        tl.Embedding(input_vocab_size, d_model),
        
        # feed the embeddings to the LSTM layers. It is a stack of n_encoder_layers LSTM layers
        [tl.LSTM(d_model) for _ in range(n_encoder_layers)]
    )

    return input_encoder


import w1_unittest
w1_unittest.test_input_encoder_fn(input_encoder_fn)


#Pre-attention decoder

def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    """ Pre-attention decoder runs on the targets and creates
    activations that are used as queries in attention.
    
    Args:
        mode: str: 'train' or 'eval'
        target_vocab_size: int: vocab size of the target
        d_model: int:  depth of embedding (n_units in the LSTM cell)
    Returns:
        tl.Serial: The pre-attention decoder
    """
    
    # create a serial network
    pre_attention_decoder = tl.Serial(
        
        # shift right to insert start-of-sentence token and implement
        # teacher forcing during training
        tl.ShiftRight(),

        # run an embedding layer to convert tokens to vectors
        tl.Embedding(target_vocab_size, d_model),

        # feed to an LSTM layer
        tl.LSTM(d_model)
    )
    
    return pre_attention_decoder


w1_unittest.test_pre_attention_decoder_fn(pre_attention_decoder_fn)

# Preparing the attention input

def prepare_attention_input(encoder_activations, decoder_activations, inputs):
    """Prepare queries, keys, values and mask for attention.
    
    Args:
        encoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the input encoder
        decoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the pre-attention decoder
        inputs fastnp.array(batch_size, padded_input_length): padded input tokens
    
    Returns:
        queries, keys, values and mask for attention.
    """
        
    # set the keys and values to the encoder activations
    keys = encoder_activations
    values = encoder_activations

    # set the queries to the decoder activations
    queries = decoder_activations
    
    # generate the mask to distinguish real tokens from padding
    # inputs is 1 for real tokens and 0 where they are padding
    mask = (inputs > 0)*1

    # add axes to the mask for attention heads and decoder length.
    mask = fastnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))

    # broadcast so mask shape is [batch size, attention heads, decoder-len, encoder-len].
    # note: attention heads is set to 1.
    mask = mask + fastnp.zeros((1, 1, decoder_activations.shape[1], 1))

    return queries, keys, values, mask


w1_unittest.test_prepare_attention_input(prepare_attention_input)


def NMTAttn(input_vocab_size=33300,
            target_vocab_size=33300,
            d_model=1024,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_attention_heads=4,
            attention_dropout=0.0,
            mode='train'):
    """Returns an LSTM sequence-to-sequence model with attention.

    The input to the model is a pair (input tokens, target tokens), e.g.,
    an English sentence (tokenized) and its translation into German (tokenized).

    Args:
    input_vocab_size: int: vocab size of the input
    target_vocab_size: int: vocab size of the target
    d_model: int:  depth of embedding (n_units in the LSTM cell)
    n_encoder_layers: int: number of LSTM layers in the encoder
    n_decoder_layers: int: number of LSTM layers in the decoder after attention
    n_attention_heads: int: number of attention heads
    attention_dropout: float, dropout for the attention layer
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

    Returns:
    A LSTM sequence-to-sequence model with attention.
    """
    
    # Step 0: call the helper function to create layers for the input encoder
    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)

    # Step 0: call the helper function to create layers for the pre-attention decoder
    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)

    # Step 1: create a serial network
    model = tl.Serial( 
        
      # Step 2: copy input tokens and target tokens as they will be needed later.
      tl.Select([0, 1, 0, 1]),
        
      # Step 3: run input encoder on the input and pre-attention decoder the target.
      tl.Parallel(input_encoder, pre_attention_decoder),
        
      # Step 4: prepare queries, keys, values and mask for attention.
      tl.Fn('PrepareAttentionInput', prepare_attention_input, n_out=4),
        
      # Step 5: run the AttentionQKV layer
      # nest it inside a Residual layer to add to the pre-attention decoder activations(i.e. queries)
      tl.Residual(tl.AttentionQKV(d_model, n_heads=n_attention_heads, dropout=attention_dropout, mode=mode)),
      
      # Step 6: drop attention mask (i.e. index = None
      tl.Select([0,2]),
        
      # Step 7: run the rest of the RNN decoder
      [tl.LSTM(d_model) for _ in range(n_decoder_layers)],

      # Step 8: prepare output by making it the right size
      tl.Dense(input_vocab_size),
        
      # Step 9: Log-softmax for output
       tl.LogSoftmax()
    )
        
    return model


w1_unittest.test_NMTAttn(NMTAttn)



# print your model
model = NMTAttn()
print(model)


train_task = training.TrainTask(
    
    # use the train batch stream as labeled data
    labeled_data= train_batch_stream,
    
    # use the cross entropy loss
    loss_layer= tl.CrossEntropyLoss(),
    
    # use the Adam optimizer with learning rate of 0.01
    optimizer= trax.optimizers.Adam(0.01),
    
    # use the `trax.lr.warmup_and_rsqrt_decay` as the learning rate schedule
    # have 1000 warmup steps with a max value of 0.01
    lr_schedule= trax.lr.warmup_and_rsqrt_decay(1000, 0.01),
    
    # have a checkpoint every 10 steps
    n_steps_per_checkpoint= 10,
    
)


w1_unittest.test_train_task(train_task)


eval_task = training.EvalTask(
    
    ## use the eval batch stream as labeled data
    labeled_data=eval_batch_stream,
    
    ## use the cross entropy loss and accuracy as metrics
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
)




# define the output directory
output_dir = '../models/'

# remove old model if it exists. restarts training.
#get_ipython().system('rm -f ~/output_dir/model.pkl.gz  ')

# define the training loop
training_loop = training.Loop(NMTAttn(mode='train'),
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)



# Execute the training loop. This will take around 8 minutes to complete.
print("---------------------------------0---------------------------------------")
training_loop.run(10)

# instantiate the model we built in eval mode
model = NMTAttn(mode='eval')

print("---------------------------------1---------------------------------------")
# initialize weights from a pre-trained model
model.init_from_file("output_dir/model.pkl.gz", weights_only=True)
model = tl.Accelerate(model)

print("---------------------------------2---------------------------------------")


# Decoding
def next_symbol(NMTAttn, input_tokens, cur_output_tokens, temperature):
    """Returns the index of the next token.

    Args:
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
        cur_output_tokens (list): tokenized representation of previously translated words
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)

    Returns:
        int: index of the next token in the translated sentence
        float: log probability of the next symbol
    """


    # set the length of the current output tokens
    token_length = len(cur_output_tokens)

    # calculate next power of 2 for padding length 
    padded_length = int(2**(np.ceil(np.log2(token_length + 1))))

    # pad cur_output_tokens up to the padded_length
    padded = cur_output_tokens + [0]*(padded_length - token_length)
    
    # model expects the output to have an axis for the batch size in front so
    # convert `padded` list to a numpy array with shape (x, <padded_length>) where the
    # x position is the batch axis. (hint: you can use np.expand_dims() with axis=0 to insert a new axis)
    padded_with_batch = np.expand_dims(np.array(padded), axis=0)
    #print("padded_with_batch: ", padded_with_batch)

    # get the model prediction. remember to use the `NMTAttn` argument defined above.
    # hint: the model accepts a tuple as input (e.g. `my_model((input1, input2))`)
    output, _ = NMTAttn((input_tokens, padded_with_batch))

    # get log probabilities from the last token output
    log_probs = output[0,token_length,:]

    # get the next symbol by getting a logsoftmax sample (*hint: cast to an int)
    symbol = int(tl.logsoftmax_sample(log_probs, temperature=temperature))

    return symbol, float(log_probs[symbol])


w1_unittest.test_next_symbol(next_symbol, model)



# sampling_decode  will call the next_symbol() function above several times until the next output is the end-of-sentence token (i.e. `EOS`). 
# It takes in an input string and returns the translated version of that string.


def sampling_decode(input_sentence, NMTAttn = None, temperature=0.0, vocab_file=None, vocab_dir=None):
    """Returns the translated sentence.

    Args:
        input_sentence (str): sentence to translate.
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file

    Returns:
        tuple: (list, str, float)
            list of int: tokenized version of the translated sentence
            float: log probability of the translated sentence
            str: the translated sentence
    """
        
    # encode the input sentence
    input_tokens = tokenize(input_sentence, vocab_file=vocab_file, vocab_dir=vocab_dir)
    
    # initialize the list of output tokens
    cur_output_tokens = []
    
    # initialize an integer that represents the current output index
    cur_output = 0
    
    # Set the encoding of the "end of sentence" as 1
    EOS = 1
    
    # check that the current output is not the end of sentence token
    while cur_output != EOS:
        
        # update the current output token by getting the index of the next word (hint: use next_symbol)
        cur_output, log_prob = next_symbol(NMTAttn, input_tokens, cur_output_tokens, temperature)
        
        # append the current output token to the list of output tokens
        cur_output_tokens.append(cur_output)
    
    # detokenize the output tokens
    sentence = detokenize(cur_output_tokens, vocab_file=vocab_file, vocab_dir=vocab_dir)
        
    return cur_output_tokens, log_prob, sentence


# Test the function above. Try varying the temperature setting with values from 0 to 1.
# Run it several times with each setting and see how often the output changes.
sampling_decode("I love languages.", model, temperature=1.0, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)

w1_unittest.test_sampling_decode(sampling_decode, model)


def greedy_decode_test(sentence, NMTAttn=None, vocab_file=None, vocab_dir=None):
    """Prints the input and output of our NMTAttn model using greedy decode

    Args:
        sentence (str): a custom string.
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file

    Returns:
        str: the translated sentence
    """
    
    _,_, translated_sentence = sampling_decode(sentence, NMTAttn, vocab_file=vocab_file, vocab_dir=vocab_dir)
    
    print("English: ", sentence)
    print("German: ", translated_sentence)
    
    return translated_sentence



# put a custom string here
your_sentence = 'I love languages.'
greedy_decode_test(your_sentence, model, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)


greedy_decode_test('You are almost done with the assignment!', model, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)


# Minimum Bayes-Risk Decoding
# Getting the most probable token at each step may not necessarily produce the best results. 
# Another approach is to do Minimum Bayes Risk Decoding or MBR. The general steps to implement this are:
# 
# 1. take several random samples
# 2. score each sample against all other samples
# 3. select the one with the highest score


def generate_samples(sentence, n_samples, NMTAttn=None, temperature=0.6, vocab_file=None, vocab_dir=None):
    """Generates samples using sampling_decode()

    Args:
        sentence (str): sentence to translate.
        n_samples (int): number of samples to generate
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file
        
    Returns:
        tuple: (list, list)
            list of lists: token list per sample
            list of floats: log probability per sample
    """
    # define lists to contain samples and probabilities
    samples, log_probs = [], []

    # run a for loop to generate n samples
    for _ in range(n_samples):
        
        # get a sample using the sampling_decode() function
        sample, logp, _ = sampling_decode(sentence, NMTAttn, temperature, vocab_file=vocab_file, vocab_dir=vocab_dir)
        
        # append the token list to the samples list
        samples.append(sample)
        
        # append the log probability to the log_probs list
        log_probs.append(logp)
                
    return samples, log_probs


# generate 4 samples with the default temperature (0.6)
generate_samples('I love languages.', 4, model, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)


# Comparing overlaps
# Let us now build our functions to compare a sample against another. 
# There are several metrics available as shown in the lectures and you can try experimenting with any one of these. 
# We will be calculating scores for unigram overlaps. 
# One of the more simple metrics is the [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) 
# which gets the intersection over union of two sets. We've already implemented it below for your perusal.


def jaccard_similarity(candidate, reference):
    """Returns the Jaccard similarity between two token lists

    Args:
        candidate (list of int): tokenized version of the candidate translation
        reference (list of int): tokenized version of the reference translation

    Returns:
        float: overlap between the two token lists
    """
    
    # convert the lists to a set to get the unique tokens
    can_unigram_set, ref_unigram_set = set(candidate), set(reference)  
    
    # get the set of tokens common to both candidate and reference
    joint_elems = can_unigram_set.intersection(ref_unigram_set)
    
    # get the set of all tokens found in either candidate or reference
    all_elems = can_unigram_set.union(ref_unigram_set)
    
    # divide the number of joint elements by the number of all elements
    overlap = len(joint_elems) / len(all_elems)
    
    return overlap


# let's try using the function. remember the result here and compare with the next function below.
jaccard_similarity([1, 2, 3], [1, 2, 3, 4])


# One of the more commonly used metrics in machine translation is the ROUGE score. 
# For unigrams, this is called ROUGE-1, you can output the scores for 
# both precision and recall when comparing two samples. To get the final score, you will want to compute the F1-score as given by:


# for making a frequency table easily
from collections import Counter

def rouge1_similarity(system, reference):
    """Returns the ROUGE-1 score between two token lists

    Args:
        system (list of int): tokenized version of the system translation
        reference (list of int): tokenized version of the reference translation

    Returns:
        float: overlap between the two token lists
    """    

    # make a frequency table of the system tokens (hint: use the Counter class)
    sys_counter = Counter(system)
    #print("sys_counter: ", sys_counter)
    
    # make a frequency table of the reference tokens (hint: use the Counter class)
    ref_counter = Counter(reference)
    #print("ref_counter: ", ref_counter)

    # initialize overlap to 0
    overlap = 0
    
    # run a for loop over the sys_counter object (can be treated as a dictionary)
    for token in sys_counter:
        #print("token: ", token)
        # lookup the value of the token in the sys_counter dictionary (hint: use the get() method)
        token_count_sys = sys_counter[token]
        #print("token_count_sys: ", token_count_sys)

        # lookup the value of the token in the ref_counter dictionary (hint: use the get() method)
        token_count_ref = ref_counter[token]
        #print("token_count_ref: ", token_count_ref)
        
        # update the overlap by getting the smaller number between the two token counts above
        overlap += token_count_sys if token_count_sys < token_count_ref else token_count_ref
    
    # get the precision (i.e. number of overlapping tokens / number of system tokens)
    precision = overlap / len(system)

    # get the recall (i.e. number of overlapping tokens / number of reference tokens)
    recall = overlap / len(reference)
    #print("recall: ", recall)
    
    if precision + recall != 0:
        # compute the f1-score
        rouge1_score = (2.0*precision*recall) / (precision + recall)
        #print("rouge1_score: ", rouge1_score)
    else:
        rouge1_score = 0 
    
    return rouge1_score
    

# notice that this produces a different value from the jaccard similarity earlier
rouge1_similarity([1, 2, 3], [1, 2, 3, 4])

w1_unittest.test_rouge1_similarity(rouge1_similarity)

# Overall score
# We will now build a function to generate the overall score for a particular sample. 
# As mentioned earlier, we need to compare each sample with all other samples. 
# For instance, if we generated 30 sentences, we will need to compare sentence 1 to sentences 2 to 30. 
# Then, we compare sentence 2 to sentences 1 and 3 to 30, and so forth. 
# At each step, we get the average score of all comparisons to get the overall score for a particular sample. 
# To illustrate, these will be the steps to generate the scores of a 4-sample list.
# 
# 1. Get similarity score between sample 1 and sample 2
# 2. Get similarity score between sample 1 and sample 3
# 3. Get similarity score between sample 1 and sample 4
# 4. Get average score of the first 3 steps. This will be the overall score of sample 1.
# 5. Iterate and repeat until samples 1 to 4 have overall scores.


def average_overlap(similarity_fn, samples, *ignore_params):
    """Returns the arithmetic mean of each candidate sentence in the samples

    Args:
        similarity_fn (function): similarity function used to compute the overlap
        samples (list of lists): tokenized version of the translated sentences
        *ignore_params: additional parameters will be ignored

    Returns:
        dict: scores of each sample
            key: index of the sample
            value: score of the sample
    """  
    
    # initialize dictionary
    scores = {}
    
    # run a for loop for each sample
    for index_candidate, candidate in enumerate(samples):    
                
        # initialize overlap to 0.0
        overlap = 0.0
        
        # run a for loop for each sample
        for index_sample, sample in enumerate(samples): 

            # skip if the candidate index is the same as the sample index
            if index_candidate == index_sample:
                continue
                
            # get the overlap between candidate and sample using the similarity function
            sample_overlap = similarity_fn(candidate, sample)
            
            # add the sample overlap to the total overlap
            overlap += sample_overlap
            
        # get the score for the candidate by computing the average
        score = overlap/(len(samples)-1)
        
        # save the score in the dictionary. use index as the key.
        scores[index_candidate] = score
        
    return scores



average_overlap(jaccard_similarity, [[1, 2, 3], [1, 2, 4], [1, 2, 4, 5]], [0.4, 0.2, 0.5])


w1_unittest.test_average_overlap(average_overlap)


# In practice, it is also common to see the weighted mean being used to calculate the overall score instead of just the arithmetic mean. 
# We have implemented it below and you can use it in your experiements to see which one will give better results.


def weighted_avg_overlap(similarity_fn, samples, log_probs):
    """Returns the weighted mean of each candidate sentence in the samples

    Args:
        samples (list of lists): tokenized version of the translated sentences
        log_probs (list of float): log probability of the translated sentences

    Returns:
        dict: scores of each sample
            key: index of the sample
            value: score of the sample
    """
    
    # initialize dictionary
    scores = {}
    
    # run a for loop for each sample
    for index_candidate, candidate in enumerate(samples):    
        
        # initialize overlap and weighted sum
        overlap, weight_sum = 0.0, 0.0
        
        # run a for loop for each sample
        for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):

            # skip if the candidate index is the same as the sample index            
            if index_candidate == index_sample:
                continue
                
            # convert log probability to linear scale
            sample_p = float(np.exp(logp))

            # update the weighted sum
            weight_sum += sample_p

            # get the unigram overlap between candidate and sample
            sample_overlap = similarity_fn(candidate, sample)
            
            # update the overlap
            overlap += sample_p * sample_overlap
            
        # get the score for the candidate
        score = overlap / weight_sum
        
        # save the score in the dictionary. use index as the key.
        scores[index_candidate] = score
    
    return scores



weighted_avg_overlap(jaccard_similarity, [[1, 2, 3], [1, 2, 4], [1, 2, 4, 5]], [0.4, 0.2, 0.5])


# Putting it all together
# We will now put everything together and develop the `mbr_decode()` function. 
# Please use the helper functions you just developed to complete this. 
# You will want to generate samples, get the score for each sample, get the highest score among all samples, 
# then detokenize this sample to get the translated sentence.

def mbr_decode(sentence, n_samples, score_fn, similarity_fn, NMTAttn=None, temperature=0.6, vocab_file=None, vocab_dir=None):
    """Returns the translated sentence using Minimum Bayes Risk decoding

    Args:
        sentence (str): sentence to translate.
        n_samples (int): number of samples to generate
        score_fn (function): function that generates the score for each sample
        similarity_fn (function): function used to compute the overlap between a pair of samples
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)
        vocab_file (str): filename of the vocabulary
        vocab_dir (str): path to the vocabulary file

    Returns:
        str: the translated sentence
    """
    
    # generate samples
    samples, log_probs = generate_samples(sentence, n_samples, NMTAttn=NMTAttn, temperature=temperature, vocab_file=vocab_file, vocab_dir=vocab_dir)
    
    # use the scoring function to get a dictionary of scores
    # pass in the relevant parameters as shown in the function definition of 
    # the mean methods you developed earlier
    scores = average_overlap(similarity_fn, samples)
    
    # find the key with the highest score
    max_index = np.argmax(scores)
    
    # detokenize the token list associated with the max_index
    translated_sentence = detokenize(samples[max_index], vocab_file=vocab_file, vocab_dir=vocab_dir)
    
    ### END CODE HERE ###
    return (translated_sentence, max_index, scores)



TEMPERATURE = 1.0

# put a custom string here
your_sentence = 'She speaks English and German.'



mbr_decode(your_sentence, 4, weighted_avg_overlap, jaccard_similarity, model, TEMPERATURE, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)[0]


mbr_decode('Congratulations!', 4, average_overlap, rouge1_similarity, model, TEMPERATURE, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)[0]



mbr_decode('You have completed the assignment!', 4, average_overlap, rouge1_similarity, model, TEMPERATURE, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)[0]


w1_unittest.test_mbr_decode(mbr_decode, model)






