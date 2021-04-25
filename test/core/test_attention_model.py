import trax
from trax.fastmath import numpy as fastnp

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

def test_pre_attention_decoder_fn(pre_attention_decoder_fn):
    target = pre_attention_decoder_fn
    success = 0
    fails = 0    
    mode = 'train'
    target_vocab_size = 10
    d_model = 2
    decoder = target(mode, target_vocab_size, d_model)
    expected = f"Serial[\n  ShiftRight(1)\n  Embedding_{target_vocab_size}_{d_model}\n  LSTM_{d_model}\n]"
    proposed = str(decoder)
    # Test all layers are in the expected sequence
    try:
        assert(proposed.replace(" ", "") == expected.replace(" ", ""))
        success += 1
    except:
        fails += 1
        print("Wrong model. \nProposed:\n%s" %proposed, "\nExpected:\n%s" %expected)
    # Test the output type
    try:
        assert(isinstance(decoder, trax.layers.combinators.Serial))
        success += 1
        # Test the number of layers
        try:
            # Test 
            assert len(decoder.sublayers) == 3
            success += 1
        except:
            fails += 1
            print('The number of sublayers does not match %s <>' %len(decoder.sublayers), " %s" %3)
    except:
        fails += 1
        print("The enconder is not an object of ", trax.layers.combinators.Serial)
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")

def test_prepare_attention_input(prepare_attention_input):
    target = prepare_attention_input
    success = 0
    fails = 0
    #This unit test consider a batch size = 2, number_of_tokens = 3 and embedding_size = 4
    enc_act = fastnp.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
               [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]]])
    dec_act = fastnp.array([[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0]], 
               [[2, 0, 2, 0], [0, 2, 0, 2], [0, 0, 0, 0]]])
    inputs =  fastnp.array([[1, 2, 3], [1, 4, 0]])
    exp_mask = fastnp.array([[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]], 
                             [[[1., 1., 0.], [1., 1., 0.], [1., 1., 0.]]]])
    exp_type = type(enc_act)
    queries, keys, values, mask = target(enc_act, dec_act, inputs)
    try:
        assert(fastnp.allclose(queries, dec_act))
        success += 1
    except:
        fails += 1
        print("Queries does not match the decoder activations")
    try:
        assert(fastnp.allclose(keys, enc_act))
        success += 1
    except:
        fails += 1
        print("Keys does not match the encoder activations")
    try:
        assert(fastnp.allclose(values, enc_act))
        success += 1
    except:
        fails += 1
        print("Values does not match the encoder activations")
    try:
        assert(fastnp.allclose(mask, exp_mask))
        success += 1
    except:
        fails += 1
        print("Mask does not match expected tensor. \nExpected:" +
        "\n%s" %exp_mask, "\nOutput:\n%s" %mask)
    # Test the output type
    try:
        assert(isinstance(queries, exp_type))
        assert(isinstance(keys, exp_type))
        assert(isinstance(values, exp_type))
        assert(isinstance(mask, exp_type))
        success += 1
    except:
        fails += 1
        print("One of the output object are not of type ", jax.interpreters.xla.DeviceArray)
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success, " Tests passed")
        print('\033[91m', fails, " Tests failed")

def test_NMTAttn(NMTAttn):
    test_cases = [
                {
                    "name":"simple_test_check",
                    "expected":"Serial_in2_out2[\n  Select[0,1,0,1]_in2_out4\n  Parallel_in2_out2[\n    Serial[\n      Embedding_33300_1024\n      LSTM_1024\n      LSTM_1024\n    ]\n    Serial[\n      ShiftRight(1)\n      Embedding_33300_1024\n      LSTM_1024\n    ]\n  ]\n  PrepareAttentionInput_in3_out4\n  Serial_in4_out2[\n    Branch_in4_out3[\n      None\n      Serial_in4_out2[\n        Parallel_in3_out3[\n          Dense_1024\n          Dense_1024\n          Dense_1024\n        ]\n        PureAttention_in4_out2\n        Dense_1024\n      ]\n    ]\n    Add_in2\n  ]\n  Select[0,2]_in3_out2\n  LSTM_1024\n  LSTM_1024\n  Dense_33300\n  LogSoftmax\n]",
                    "error":"The NMTAttn is not defined properly."
                },
                {
                    "name":"layer_len_check",
                    "expected":9,
                    "error":"We found {} layers in your model. It should be 9.\nCheck the LSTM stack before the dense layer"
                },
                {
                    "name":"selection_layer_check",
                    "expected":["Select[0,1,0,1]_in2_out4", "Select[0,2]_in3_out2"],
                    "error":"Look at your selection layers."
                }
            ]
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            if test_case['name'] == "simple_test_check":
                assert test_case["expected"] == str(NMTAttn())
                success += 1
            if test_case['name'] == "layer_len_check":
                if test_case["expected"] == len(NMTAttn().sublayers):
                    success += 1
                else:
                    print(test_case["error"].format(len(NMTAttn().sublayers))) 
                    fails += 1
            if test_case['name'] == "selection_layer_check":
                model = NMTAttn()
                output = [str(model.sublayers[0]),str(model.sublayers[4])]
                check_count = 0
                for i in range(2):
                    if test_case["expected"][i] != output[i]:
                        print(test_case["error"])
                        fails += 1
                        break
                    else:
                        check_count += 1
                if check_count == 2:
                    success += 1
        except:
            print(test_case['error'])
            fails += 1
            
    if fails == 0:
        print("\033[92m All tests passed")
    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")