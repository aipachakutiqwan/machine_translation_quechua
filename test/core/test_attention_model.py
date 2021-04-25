import trax


def test_input_encoder_fn(input_encoder_fn):
    target = input_encoder_fn
    success = 0
    fails = 0
    
    input_vocab_size = 10
    d_model = 2
    n_encoder_layers = 6
    
    encoder = target(input_vocab_size, d_model, n_encoder_layers)
    lstms = "\n".join([f'  LSTM_{d_model}'] * n_encoder_layers)
    expected = f"Serial[\n  Embedding_{input_vocab_size}_{d_model}\n{lstms}\n]"
    proposed = str(encoder)
    
    # Test all layers are in the expected sequence
    try:
        assert(proposed.replace(" ", "") == expected.replace(" ", ""))
        success += 1
    except:
        fails += 1
        print("Wrong model. \nProposed:\n%s" %proposed, "\nExpected:\n%s" %expected)
    
    # Test the output type
    try:
        assert(isinstance(encoder, trax.layers.combinators.Serial))
        success += 1
        # Test the number of layers
        try:
            # Test 
            assert len(encoder.sublayers) == (n_encoder_layers + 1)
            success += 1
        except:
            fails += 1
            print('The number of sublayers does not match %s <>' %len(encoder.sublayers), " %s" %(n_encoder_layers + 1))
    except:
        fails += 1
        print("The enconder is not an object of ", trax.layers.combinators.Serial)
    
        
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")

