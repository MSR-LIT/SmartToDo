#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
mkdir '../data/Avocado'
mkdir '../data/Annotations'
mkdir '../data/SmartToDo_seq2seq_data'
mkdir './SmartToDo_seq2seq/glove_dir'
mkdir './SmartToDo_seq2seq/processed'
mkdir './SmartToDo_seq2seq/logs'
mkdir './SmartToDo_seq2seq/checkpoints'
