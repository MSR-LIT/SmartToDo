# SmartToDo

Official code for the paper [Smart To-Do : Automatic Generation of To-Do Items from Emails](https://arxiv.org/abs/2005.06282) by Sudipto Mukherjee, Subhabrata Mukherjee, Marcello Hasegawa, Ahmed Hassan Awadallah and Ryen White. If you use the code, please cite our paper. 

## Requirements

The dependency packages are provided in [requirements.txt](./requirements.txt). We strongly encourage using the Anaconda Installation of Python and creating a virtual environment. 

```bash
$ conda create --name SmartToDo python=3.6
$ ./install.sh
```
The install.sh bash script will install all required packages and also the model for Spacy. 

## Dataset 

As per the License Agreements of Avocado, "Users are required to sign two license agreements in order to access this corpus, the Avocado Collection Organizational License Agreement and the Avocado Collection End User Agreement.". So, the dataset for SmartToDo is provided in an Encrypted format. 

#### Decrypt Dataset
Download the Avocado dataset from [Avocado Research Email Collection](https://catalog.ldc.upenn.edu/LDC2015T03). Ensure that this is placed in the folder ./data with path to text files as './data/Avocado/text/\*\*\*.zip'. Then run the following script:

```bash
$ ./decrypt_data.sh
```
This will create vocabulary from Avocado corpus and decrypt the dataset. The decoded dataset will be saved in the folder './data/UHRS_judgements'. This code takes around 30 minutes to run. 


## Input/Output for Seq2Seq 

The ranked sentences for the data instances are provided in the file [sent_ranked_fasttext.txt](./data/Gold_SmartToDo_seq2seq_data/sent_ranked_fasttext.txt). Now run the following code snippet to generate the input/output for Seq2Seq model:

```bash
$ ./gen_seq2seq_data.sh
```

## To-Do item generation

Download the pretrained Glove embeddings from [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip). Extract it and place it in the folder './SmartToDo_seq2seq/glove_dir/'. Ensure that your embeddings have the path as './SmartToDo_seq2seq/glove_dir/glove.6B/glove.6B.100d.txt'. Then run the following:

```bash
$ cd SmartToDo_seq2seq
$ ./gen_ToDo.sh
```
This will create vocaulary for training, train the model and generate To-Do items on the validation and test set. The performance metrics will also be computed and printed on screen.


## Additional Code

* [extractiveSummary](./extractiveSummary) - This folder contains code that was used to extract "Helpful" sentences using fastText, Term frequency (Tf) or BERT emebeddings.

* [wordEmbed](./wordEmbed) - This folder contains code for training fastText embeddings on raw Avocado sentences. 

## Pre-trained Model

In addition to training your own model from scratch, a pre-trained model checkpoint can also be downloaded from the Google Drive link (https://drive.google.com/drive/folders/1r8cLNSeU9l_oTBcaEeze52p0candbTYh?usp=sharing). 

## Acknowledgment

Our seq2seq code is heavily based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). We thank that authors for open-sourcing the NMT code.

## Feedback

Please feel free to email any feedback and/or suggestions about the code to sudipm@uw.edu or submukhe@microsoft.com.


