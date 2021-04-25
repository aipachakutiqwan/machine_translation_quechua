"""
Training proccess with attention model
"""
from termcolor import colored
import trax
from trax import layers as tl
from trax.supervised import training
from core.log_manager import LogManagerBase
from core.attention_model import AttentionModel
from preprocessor import Preprocessor


class Trainer():
    """
    Class for train machine translation model with attention
    """

    def __init__(self):
        return None

    @staticmethod
    def generator_data(folder_data):
        """
        Get generator function for the training/eval dataset.
        This will download the train dataset if no data_dir is specified.

        Args:
            folder_data: folder dataset
        Returns:
            stream generated datasets
        """
        train_stream_fn = trax.data.TFDS('opus/medical',
                                         data_dir=folder_data,
                                         keys=('en', 'de'),
                                         eval_holdout_size=0.01, # 1% for eval
                                         train=True)
        # Get generator function for the eval set
        eval_stream_fn = trax.data.TFDS('opus/medical',
                                        data_dir=folder_data,
                                        keys=('en', 'de'),
                                        eval_holdout_size=0.01, # 1% for eval
                                        train=False)
        return train_stream_fn, eval_stream_fn

    def run(self, folder_data, vocab_file, vocab_dir, output_dir):
        """
        Runner of the training process
        Args:
            folder_data: folder dataset
            vocab_file: vocabulary file
            vocab_dir: vocabulary directory
        Returns:
            trained machine translation model
        """
        train_stream_fn, eval_stream_fn = self.generator_data(folder_data)
        train_stream = train_stream_fn()
        LogManagerBase.info(colored(f'first train data (en, de) tuple: ', 'green') +
                            str(next(train_stream)))
        eval_stream = eval_stream_fn()
        LogManagerBase.info(colored(f'first eval data (en, de) tuple: ', 'green') +
                            str(next(eval_stream)))
        train_batch_stream, eval_batch_stream, mask_batch = \
        Preprocessor().run(train_stream, eval_stream, vocab_file, vocab_dir)
        # Training
        LogManagerBase.info(AttentionModel().nmt_attn())

        train_task = training.TrainTask(
            # use the train batch stream as labeled data
            labeled_data=train_batch_stream,
            # use the cross entropy loss
            loss_layer=tl.CrossEntropyLoss(),
            # use the Adam optimizer with learning rate of 0.01
            optimizer=trax.optimizers.Adam(0.01),
            # use the `trax.lr.warmup_and_rsqrt_decay` as the learning rate schedule
            # have 1000 warmup steps with a max value of 0.01
            lr_schedule=trax.lr.warmup_and_rsqrt_decay(1000, 0.01),
            # have a checkpoint every 10 steps
            n_steps_per_checkpoint=10,)

        #print("test_train_task ....")
        #w1_unittest.test_train_task(train_task)

        eval_task = training.EvalTask(
            ## use the eval batch stream as labeled data
            labeled_data=eval_batch_stream,
            ## use the cross entropy loss and accuracy as metrics
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],)

        # define the training loop
        training_loop = training.Loop(AttentionModel().nmt_attn(mode='train'),
                                      train_task,
                                      eval_tasks=[eval_task],
                                      output_dir=output_dir)
        # Execute the training loop, this will take time.
        training_loop.run(1)
        return train_batch_stream, eval_batch_stream, mask_batch

if __name__ == "__main__":

    FOLDER_DATA = './../data/'
    VOCAB_FILE = 'ende_32k.subword'
    VOCAB_DIR = './../data/'
    OUTPUT_DIR = './../models/'
    TRAINER = Trainer()
    _, _, _ = TRAINER.run(FOLDER_DATA, VOCAB_FILE, VOCAB_DIR, OUTPUT_DIR)
