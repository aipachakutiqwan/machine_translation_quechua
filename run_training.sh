#!/bin/bash
#Script to run training machine translation attention model
IMAGE_NAME='fpaucar/machine_translation:1.0.0'
#Enviromental variables
FOLDER_DATA='/translator/code/data'
VOCAB_FILE='ende_32k.subword'
VOCAB_DIR='/translator/data'
OUTPUT_DIR='/translator/models'
TRAINING_STEPS=1

docker run -v /Users/c325018/Documents/TraductorQuechua:/translator \
-v /Users/c325018/Documents/TraductorQuechua/logs:/app/logs \
--env FOLDER_DATA=$FOLDER_DATA \
--env VOCAB_FILE=$VOCAB_FILE \
--env VOCAB_DIR=$VOCAB_DIR \
--env OUTPUT_DIR=$OUTPUT_DIR \
--env TRAINING_STEPS=$TRAINING_STEPS \
${IMAGE_NAME} /bin/bash -c 'python /app/src/trainer.py'


