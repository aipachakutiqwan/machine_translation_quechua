"""
Neural Machine Translation with Attention.
"""
from trax import layers as tl
from trax.fastmath import numpy as fastnp


class AttentionModel():
    """
    Class Neural Machine Translation with Attention.
    """

    def __init__(self):
        return None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def prepare_attention_input(encoder_activations, decoder_activations, inputs):
        """Prepare queries, keys, values and mask for attention.

        Args:
            encoder_activations fastnp.array(batch_size, padded_input_length, d_model):
            output from the input encoder
            decoder_activations fastnp.array(batch_size, padded_input_length, d_model):
            output from the pre-attention decoder
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


    def nmt_attn(self, input_vocab_size=33300,
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
        # call the helper function to create layers for the input encoder
        input_encoder = self.input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)
        # call the helper function to create layers for the pre-attention decoder
        pre_attention_decoder = self.pre_attention_decoder_fn(mode, target_vocab_size, d_model)
        # create a serial network
        model = tl.Serial(tl.Select([0, 1, 0, 1]),
                          # copied input tokens and target tokens as they will be needed later.
                          # run input encoder on the input and pre-attention decoder the target.
                          tl.Parallel(input_encoder, pre_attention_decoder),
                          # prepare queries, keys, values and mask for attention.
                          tl.Fn('PrepareAttentionInput', self.prepare_attention_input, n_out=4),
                          # run the AttentionQKV layer
                          # nest it inside a Residual layer to add to the
                          # pre-attention decoder activations(i.e. queries)
                          tl.Residual(tl.AttentionQKV(d_model, n_heads=n_attention_heads,
                                                      dropout=attention_dropout, mode=mode)),
                          # drop attention mask (i.e. index = None
                          tl.Select([0, 2]),
                          # run the rest of the RNN decoder
                          [tl.LSTM(d_model) for _ in range(n_decoder_layers)],
                          # prepare output by making it the right size
                          tl.Dense(input_vocab_size),
                          # Log-softmax for output
                          tl.LogSoftmax())
        return model

if __name__ == "__main__":

    ATTMODEL = AttentionModel()
    ATTMODEL.nmt_attn(input_vocab_size=33300,
                      target_vocab_size=33300,
                      d_model=1024,
                      n_encoder_layers=2,
                      n_decoder_layers=2,
                      n_attention_heads=4,
                      attention_dropout=0.0,
                      mode='train')
