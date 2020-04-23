

(0) First ensure that you are in an Environment where the requirements for OpenNMT-py are installed. 

(a) One way to do this is cd to OpenNMT-py folder 

If setting up on a machine from scratch, first git clone: 
git clone https://github.com/OpenNMT/OpenNMT-py
cd OpenNMT-py

Otherwise, just cd to OpenNMT-py folder.
pip install -r requirements.txt


(b) I used anaconda on Azure and Laptop. So in anaconda first create a virtual environment.

conda create -n opennmt python=3.6

Then every time you want to run OpenNMT related codes, activate this virtual environment :

conda activate opennmt

(opennmt) ~/SmartToDo$ 

- Then either 
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd opennmt
pip install -r requirements.txt

or if already existing OpenNMT-py folder, just 
pip install -r requirements.txt


Now, you are all set to run OpenNMT related codes. 


- Make sure you are in BiFocal_NMT folder with the conda environment activated. 

(opennmt) :~/BiFocal_NMT$ 

(1) Create vocabulary and necessary model training inputs from the train and validation dataset

python preprocess.py -train_src BiFocal_seq2seq_final_data/avocado.spacy_tokenized/src-train.txt -train_qry BiFocal_seq2seq_final_data/avocado.spacy_tokenized/qry-train.txt -train_tgt BiFocal_seq2seq_final_data/avocado.spacy_tokenized/tgt-train.txt -valid_src BiFocal_seq2seq_final_data/avocado.spacy_tokenized/src-valid.txt -valid_qry BiFocal_seq2seq_final_data/avocado.spacy_tokenized/qry-valid.txt -valid_tgt BiFocal_seq2seq_final_data/avocado.spacy_tokenized/tgt-valid.txt -save_data processed_final/avocado.spacy_tokenized.separate.copy --src_seq_length 256 --qry_seq_length 256 --src_words_min_frequency 2 --qry_words_min_frequency 2 --tgt_words_min_frequency 2 --dynamic_dict


This will create the following .pt files in ./processed_final folder :
avocado.spacy_tokenized.separate.copy.train.0.pt
avocado.spacy_tokenized.separate.copy.valid.0.pt
avocado.spacy_tokenized.separate.copy.vocab.pt


(2) Initialize word embeddings of the vocabulary from glove. * This assumes that you have the glove dir path as ../OpenNMT-py/glove_dir/ *

python ./tools/embeddings_to_torch.py -emb_file_enc "../OpenNMT-py/glove_dir/glove.6B/glove.6B.100d.txt" -emb_file_match "../OpenNMT-py/glove_dir/glove.6B/glove.6B.100d.txt" -emb_file_dec "../OpenNMT-py/glove_dir/glove.6B/glove.6B.100d.txt" -dict_file "processed_final/avocado.spacy_tokenized.separate.copy.vocab.pt" -output_file "processed_final/avocado.spacy_tokenized.separate.copy.embeddings.d100"


This will create the following .pt files in ./processed_final folder :
avocado.spacy_tokenized.separate.copy.embeddings.d100.dec.pt
avocado.spacy_tokenized.separate.copy.embeddings.d100.enc.pt
avocado.spacy_tokenized.separate.copy.embeddings.d100.match.pt


(3) Now you are all set to start training the model. * Suppose the config file is created and stored as ./BiFocal_config_final/spacy/config-460.yml *

CUDA_VISIBLE_DEVICES=0 python train.py --config "BiFocal_config_final/spacy/config-460.yml"

This will train the model and save the best checkpoint in the directory provided in config-460.yml. Suppose it is checkpoints_BiFocal_final/avocado.spacy_tokenized/460/model_step_best.pt

(4) Now translate the validation and test data respectively and store the predicted sentences in ./logs_BiFocal_final/avocado.spacy_tokenized/

CUDA_VISIBLE_DEVICES=0 python translate.py -model checkpoints_BiFocal_final/avocado.spacy_tokenized/460/model_step_best.pt -src BiFocal_seq2seq_final_data/avocado.spacy_tokenized/src-valid.txt -qry BiFocal_seq2seq_final_data/avocado.spacy_tokenized/qry-valid.txt -output logs_BiFocal_final/avocado.spacy_tokenized/pred-460-valid.txt -verbose --tgt BiFocal_seq2seq_final_data/avocado.spacy_tokenized/tgt-valid.txt --max_length 50 --gpu 0

Followed by 

CUDA_VISIBLE_DEVICES=0 python translate.py -model checkpoints_BiFocal_final/avocado.spacy_tokenized/460/model_step_best.pt -src BiFocal_seq2seq_final_data/avocado.spacy_tokenized/src-test.txt -qry BiFocal_seq2seq_final_data/avocado.spacy_tokenized/qry-test.txt -output logs_BiFocal_final/avocado.spacy_tokenized/pred-460-test.txt -verbose --tgt BiFocal_seq2seq_final_data/avocado.spacy_tokenized/tgt-test.txt --max_length 50 --gpu 0


* Notice that --remove_unk feature is disabled for BiFocal_NMT. It is not required as such and so is removed from available options. *


Thats it ! This will write the logs on the stdout. But, it you want to store the logs as well, alongside the predicted sentences, then use the following : 

CUDA_VISIBLE_DEVICES=0 python translate.py -model checkpoints_BiFocal_final/avocado.spacy_tokenized/460/model_step_best.pt -src BiFocal_seq2seq_final_data/avocado.spacy_tokenized/src-valid.txt -qry BiFocal_seq2seq_final_data/avocado.spacy_tokenized/qry-valid.txt -output logs_BiFocal_final/avocado.spacy_tokenized/pred-460-valid.txt -verbose --tgt BiFocal_seq2seq_final_data/avocado.spacy_tokenized/tgt-valid.txt --max_length 50 --gpu 0  > logs_BiFocal_final/avocado.spacy_tokenized/raw_logs-460-valid.txt

Followed by 


CUDA_VISIBLE_DEVICES=0 python translate.py -model checkpoints_BiFocal_final/avocado.spacy_tokenized/460/model_step_best.pt -src BiFocal_seq2seq_final_data/avocado.spacy_tokenized/src-test.txt -qry BiFocal_seq2seq_final_data/avocado.spacy_tokenized/qry-test.txt -output logs_BiFocal_final/avocado.spacy_tokenized/pred-460-test.txt -verbose --tgt BiFocal_seq2seq_final_data/avocado.spacy_tokenized/tgt-test.txt --max_length 50 --gpu 0  > logs_BiFocal_final/avocado.spacy_tokenized/raw_logs-460-test.txt

