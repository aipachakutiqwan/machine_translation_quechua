"""
Prepare dataset for the training process
"""
import random
import trax
from termcolor import colored
from utils.utils import append_eos, tokenize, detokenize
from core.log_manager import LogManagerBase

class Preprocessor():
    """
    Class for preprocess dataset
    """

    def __init__(self):
        return None

    @staticmethod
    def tokenize_dataset(train_stream, eval_stream, vocab_file, vocab_dir):
        """
        function to tokenize dataset and append EOS to each sentence

        Args:
            vocab_file: vocabulary file
            vocab_dir: vocabulary directory
        Returns:
            tokenized datasets
        """
        # Tokenize the dataset
        tokenized_train_stream = trax.data.Tokenize(vocab_file=vocab_file,
                                                    vocab_dir=vocab_dir)(train_stream)
        tokenized_eval_stream = trax.data.Tokenize(vocab_file=vocab_file,
                                                   vocab_dir=vocab_dir)(eval_stream)
        # Append EOS to the train/eval data
        tokenized_train_stream = append_eos(tokenized_train_stream)
        tokenized_eval_stream = append_eos(tokenized_eval_stream)
        return  tokenized_train_stream, tokenized_eval_stream

    @staticmethod
    def filter_long_sentences(tokenized_train_stream, tokenized_eval_stream):
        """
        function to filter long sentences.
        Filter too long sentences to not run out of memory.
        length_keys=[0, 1] means we filter both English and German sentences, so
        both much be not longer that 256 tokens for training / 512 for eval.

        Args:
            tokenized_train_stream: tokenized train stream
            tokenized_eval_stream: tokenized val stream
        Returns:
            filtered datasets
        """

        filtered_train_stream = trax.data.FilterByLength(
            max_length=256, length_keys=[0, 1])(tokenized_train_stream)
        filtered_eval_stream = trax.data.FilterByLength(
            max_length=512, length_keys=[0, 1])(tokenized_eval_stream)
        return filtered_train_stream, filtered_eval_stream

    @staticmethod
    def bucketing(filtered_train_stream, filtered_eval_stream):
        """
        Bucketing to create streams of batches.

        Args:
            filtered_train_stream: filtered tokenized train stream
            iltered_eval_stream: filtered tokenized val stream
        Returns:
            bucketed in batches
        """
        boundaries = [8, 16, 32, 64, 128, 256, 512]
        batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]
        # Create the generators.
        train_batch_stream = trax.data.BucketByLength(
            boundaries, batch_sizes,
            length_keys=[0, 1]  # As before: count inputs and targets to length.
        )(filtered_train_stream)
        eval_batch_stream = trax.data.BucketByLength(
            boundaries, batch_sizes,
            length_keys=[0, 1]  # As before: count inputs and targets to length.
        )(filtered_eval_stream)
        return train_batch_stream, eval_batch_stream

    @staticmethod
    def padding(train_batch_stream, eval_batch_stream):
        """
        Add masking for the padding (0s).

        Args:
            train_batch_stream: bucketed train stream
            eval_batch_stream: bucketed val stream
        Returns:
            padding datasets
        """
        train_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(train_batch_stream)
        eval_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_stream)
        return train_batch_stream, eval_batch_stream

    def run(self, train_stream, eval_stream, vocab_file, vocab_dir):
        """
        Preprocess train and eval dataset

        Args:
            train_stream: train stream
            eval_stream: val stream
            vocab_file: vocabulary file
            vocab_dir: vocabbulary directory
        Returns:
            datasets processed
        """

        tokenized_train_stream, tokenized_eval_stream = self.tokenize_dataset(
            train_stream, eval_stream, vocab_file, vocab_dir)
        filtered_train_stream, filtered_eval_stream = self.filter_long_sentences(
            tokenized_train_stream, tokenized_eval_stream)

        # Print a sample input-target pair of tokenized sentences
        train_input, train_target = next(filtered_train_stream)
        LogManagerBase.info(colored(f'Single tokenized train example input:', 'red') +
                            str(train_input))
        LogManagerBase.info(colored(f'Single tokenized train example target:', 'red') +
                            str(train_target))

        # Print detokenize an input-target pair of tokenized sentences
        LogManagerBase.info(colored(f'Single detokenized example train input: ', 'green') +
                            detokenize(train_input, vocab_file=vocab_file, vocab_dir=vocab_dir))
        LogManagerBase.info(colored(f'Single detokenized example train target: ', 'green') +
                            detokenize(train_target, vocab_file=vocab_file, vocab_dir=vocab_dir))

        # Tokenize and detokenize a word that is not explicitly saved in the vocabulary file.
        # See how it combines the subwords -- 'hell' and 'o'-- to form the word 'hello'.
        LogManagerBase.info(colored(f"tokenize('hello'): ", 'red') +
                            str(tokenize('hello', vocab_file=vocab_file, vocab_dir=vocab_dir)))
        LogManagerBase.info(colored(f"detokenize([17332, 140, 1]): ", 'red') +
                            str(detokenize([17332, 140, 1],
                                           vocab_file=vocab_file, vocab_dir=vocab_dir)))

        train_batch_stream, eval_batch_stream = self.bucketing(filtered_train_stream,
                                                               filtered_eval_stream)
        train_batch_stream, eval_batch_stream = self.padding(train_batch_stream,
                                                             eval_batch_stream)

        input_batch, target_batch, mask_batch = next(train_batch_stream)
        # Print the data type of a batch
        LogManagerBase.info(f"input_batch data type: " + str(type(input_batch)))
        LogManagerBase.info(f"target_batch data type: " + str(type(target_batch)))
        # Print the shape of this particular batch (batch length, sentence length)
        LogManagerBase.info(f"input_batch shape: " + str(input_batch.shape))
        LogManagerBase.info(f"target_batch shape: " + str(target_batch.shape))
        # pick a random index less than the batch size.
        index = random.randrange(len(input_batch))
        # use the index to grab an entry from the input and target batch
        LogManagerBase.info(colored('THIS IS THE ENGLISH SENTENCE: \n', 'red') +
                            str(detokenize(input_batch[index],
                                           vocab_file=vocab_file, vocab_dir=vocab_dir)))
        LogManagerBase.info(colored('THIS IS THE TOKENIZED VERSION OF THE ENGLISH SENTENCE: \n',
                                    'red') + str(input_batch[index]))
        LogManagerBase.info(colored('THIS IS THE GERMAN TRANSLATION: \n',
                                    'red') + str(detokenize(target_batch[index],
                                                            vocab_file=vocab_file,
                                                            vocab_dir=vocab_dir)))
        LogManagerBase.info(colored('THIS IS THE TOKENIZED VERSION OF THE GERMAN TRANSLATION: \n',
                                    'red') + str(target_batch[index]))
        return train_batch_stream, eval_batch_stream, mask_batch

if __name__ == "__main__":

    PREPR = Preprocessor()
