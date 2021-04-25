
# instantiate the model we built in eval mode
model = NMTAttn(mode='eval')

print("---------------------------------1---------------------------------------")
# initialize weights from a pre-trained model
model.init_from_file("./../models/model.pkl.gz", weights_only=True)
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






