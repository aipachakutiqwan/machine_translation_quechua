import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training

VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = 'data/'

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

# UNIT TEST for UNQ_C5
def test_train_task(train_task):
    target = train_task
    success = 0
    fails = 0
     
    # Test the labeled data parameter
    try:
        strlabel = str(target._labeled_data)
        assert(strlabel.find("generator") and strlabel.find('add_loss_weights'))
        success += 1
    except:
        fails += 1
        print("Wrong labeled data parameter")
    
    # Test the cross entropy loss data parameter
    try:
        strlabel = str(target._loss_layer)
        assert(strlabel == "CrossEntropyLoss_in3")
        success += 1
    except:
        fails += 1
        print("Wrong loss functions. CrossEntropyLoss_in3 was expected")
        
     # Test the optimizer parameter
    try:
        assert(isinstance(target.optimizer, trax.optimizers.adam.Adam))
        success += 1
    except:
        fails += 1
        print("Wrong optimizer")
        
    # Test the schedule parameter
    try:
        assert(isinstance(target._lr_schedule,trax.supervised.lr_schedules._BodyAndTail))
        success += 1
    except:
        fails += 1
        print("Wrong learning rate schedule type")
    
    # Test the _n_steps_per_checkpoint parameter
    try:
        assert(target._n_steps_per_checkpoint==10)
        success += 1
    except:
        fails += 1
        print("Wrong checkpoint step frequency")
        
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")



# UNIT TEST for UNQ_C6
def test_next_symbol(next_symbol, model):
    target = next_symbol
    the_model = model
    success = 0
    fails = 0
        
    tokens_en = np.array([[17332, 140, 172, 207, 1]])
     
    # Test the type and size of output
    try:
        next_de_tokens = target(the_model, tokens_en, [], 0.0) 
        assert(isinstance(next_de_tokens, tuple))
        assert(len(next_de_tokens) == 2)
        assert(type(next_de_tokens[0]) == int and type(next_de_tokens[1]) == float)
        success += 1
    except:
        fails += 1
        print("Output must be a tuple of size 2 containing a integer and a float number")
    
    # Test an output
    try:
        next_de_tokens = target(the_model, tokens_en, [18477], 0.0)
        assert(np.allclose([next_de_tokens[0], next_de_tokens[1]], [140, -0.000217437744]))
        success += 1
    except:
        fails += 1
        print("Expected output: ", [140, -0.000217437744])
    
        
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")


# UNIT TEST for UNQ_C7
def test_sampling_decode(sampling_decode, model):
    target = sampling_decode
    the_model = model
    success = 0
    fails = 0
    
    try:
        output = target("I eat soup.", model, temperature=0, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)
        expected = ([161, 15103, 5, 25132, 35, 3, 1], -0.0003108978271484375, 'Ich iss Suppe.')
        assert(output[2] == expected[2])
        success += 1
    except:
        fails += 1
        print("Test 1 fails")
        
    try:
        output = target("I like your shoes.", model, temperature=0, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)
        expected = ([161, 15103, 5, 25132, 35, 3, 1], -0.0003108978271484375, 'Ich mag Ihre Schuhe.')
        assert(output[2] == expected[2])
        success += 1
    except:
        fails += 1
        print("Test 2 fails")
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        

# UNIT TEST for UNQ_C8
def test_rouge1_similarity(rouge1_similarity):
    target = rouge1_similarity
    success = 0
    fails = 0
    n_samples = 10
    
    test_cases = [
        
        {
            "name":"simple_test_check",
            "input": [[1, 2, 3], [1, 2, 3, 4]],
            "expected":0.8571428571428571,
            "error":"Expected similarity: 0.8571428571428571"
        },
        {
            "name":"simple_test_check",
            "input":[[2, 1], [3, 1]],
            "expected":0.5,
            "error":"Expected similarity: 0.5"
        },
        {
            "name":"simple_test_check",
            "input":[[2], [3]],
            "expected":0,
            "error":"Expected similarity: 0"
        },
        {
            "name":"simple_test_check",
            "input":[[0] * 100 + [2] * 100, [0] * 100 + [1] * 100],
            "expected":0.5,
            "error":"Expected similarity: 0.5"
        }
    ]

    for test_case in test_cases:
        
        try:
            if test_case['name'] == "simple_test_check":
                assert abs(test_case["expected"] -target(*test_case['input'])) < 1e-6
                success += 1
        except:
            print(test_case['error'])
            fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        

# UNIT TEST for UNQ_C9
def test_average_overlap(average_overlap):
    target = average_overlap
    success = 0
    fails = 0
    
    test_cases = [
        
        {
            "name":"dict_test_check",
            "input": [jaccard_similarity, [[1, 2], [3, 4], [1, 2], [3, 5]]],
            "expected":{0: 0.3333333333333333,
                        1: 0.1111111111111111,
                        2: 0.3333333333333333,
                        3: 0.1111111111111111},
            "error":"Expected output does not match"
        },
        {
            "name":"dict_test_check",
            "input":[jaccard_similarity, [[1, 2], [3, 4], [1, 2, 5], [3, 5], [3, 4, 1]]],
            "expected":{0: 0.22916666666666666,
                        1: 0.25,
                        2: 0.2791666666666667,
                        3: 0.20833333333333331,
                        4: 0.3416666666666667},
            "error":"Expected output does not match"
        }
    ]
    for test_case in test_cases:
        try:
            if test_case['name'] == "dict_test_check":
                output = target(*test_case['input'])
                for x in output:
                    assert (abs(output[x] - test_case['expected'][x]) < 1e-5)
                    success += 1
        except:
            print(test_case['error'])
            fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")


# UNIT TEST for UNQ_C10
def test_mbr_decode(mbr_decode, model):
    target = mbr_decode
    success = 0
    fails = 0
    
    TEMPERATURE = 0.0
    
    test_cases = [
        
        {
            "name":"simple_test_check",
            "input": "I am hungry",
            "expected":"Ich bin hungrig.",
            "error":"Expected output does not match"
        },
        {
            "name":"simple_test_check",
            "input":'Congratulations!',
            "expected":'Herzlichen Glückwunsch!',
            "error":"Expected output does not match"
        },
        {
            "name":"simple_test_check",
            "input":'You have completed the assignment!',
            "expected":'Sie haben die Abtretung abgeschlossen!',
            "error":"Expected output does not match"
        }
    ]
    for test_case in test_cases:
        try:
            result = target(test_case['input'], 4, weighted_avg_overlap, jaccard_similarity, 
                                model, TEMPERATURE, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)   

            output = result[0]
            if test_case['name'] == "simple_test_check":
                assert(output == test_case['expected'])
                success += 1
        except:
            print(test_case['error'])
            fails += 1
            
    # Test that function return the most likely translation
    TEMPERATURE = 0.5
    test_case =  test_cases[0]
    try:
        result = target(test_case['input'], 4, weighted_avg_overlap, jaccard_similarity, 
                                model, TEMPERATURE, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)   

        assert  max(result[2], key=result[2].get) == result[1]
        success += 1
    except:
        print('Use max function to select max_index')
        fails += 1
    
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")