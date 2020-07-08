# SmartToDo

Official code for the paper [Smart To-Do : Automatic Generation of To-Do Items from Emails](https://www.aclweb.org/anthology/2020.acl-main.767.pdf) by Sudipto Mukherjee, Subhabrata Mukherjee, Marcello Hasegawa, Ahmed Hassan Awadallah and Ryen White. If you use the code, please cite our work :


```
@inproceedings{mukherjee-etal-2020-smart,
    title = "Smart To-Do: Automatic Generation of To-Do Items from Emails",
    author = "Mukherjee, Sudipto  and Mukherjee, Subhabrata  and Hasegawa, Marcello  and Hassan Awadallah, Ahmed  and  White, Ryen",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

## Requirements

The dependency packages are provided in [requirements.txt](./requirements.txt). We strongly encourage using the Anaconda Installation of Python and creating a virtual environment. 

```bash
~/SmartToDo$ conda create --name SmartToDo python=3.6
~/SmartToDo$ conda activate SmartToDo
(SmartToDo)~/SmartToDo$ cd src
(SmartToDo)~/SmartToDo/src$ chmod +x *.sh
(SmartToDo)~/SmartToDo/src$ ./install.sh
```
The install.sh bash script will install all required packages and also the model for Spacy. 

## Dataset 

As per the License Agreements of Avocado, "Users are required to sign two license agreements in order to access this corpus, the Avocado Collection Organizational License Agreement and the Avocado Collection End User Agreement.". So, the To-Do summaries written by judges in SmartToDo are provided in a Coded format. 

#### Decoding Dataset
Download the Avocado dataset from [Avocado Research Email Collection](https://catalog.ldc.upenn.edu/LDC2015T03). Ensure that this is placed in the folder ./data with path to text files as './data/Avocado/text/\*\*\*.zip'. Then run the following script:

```bash
(SmartToDo)~/SmartToDo/src$ ./decode_data.sh
```
This will create vocabulary from Avocado corpus and decode the dataset. The decoded dataset will be saved in the folder './data/Annotations'. This code takes around 1 hour 10 minutes to run.

After decoding, the file './data/Annotations/Spans_ToDo_dataset.tsv' with contain the emails ids, the commitment sentence and the To-Do Item. The user is required to extract the rest of the fields ("reply_to_sent_from", "reply_to_sent_to", "reply_to_subject", "reply_to_body", "current_sent_from", "current_sent_to", "current_subject", "current_body_before_high", "current_body_after_high", etc.) from Avocado data.

The code below assumes that a tsv field is created by the user containing the necessary fields and stored as './data/Annotations/SmartToDo_dataset.tsv'.

## Input/Output for Seq2Seq 

The ranked sentences for the data instances are provided in the file [sent_ranked_fasttext.txt](./data/Gold_SmartToDo_seq2seq_data/sent_ranked_fasttext.txt). Now run the following code snippet to generate the input/output for Seq2Seq model:

```bash
(SmartToDo)~/SmartToDo/src$ ./gen_seq2seq_data.sh
```

## To-Do item generation

Download the pretrained Glove embeddings from [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip). Extract it and place it in the folder './src/SmartToDo_seq2seq/glove_dir/'. Ensure that your embeddings have the path as './src/SmartToDo_seq2seq/glove_dir/glove.6B/glove.6B.100d.txt'. Then run the following:

```bash
(SmartToDo)~/SmartToDo/src$ cd SmartToDo_seq2seq
(SmartToDo)~/SmartToDo/src$ chmod +x *.sh
(SmartToDo)~/SmartToDo/src/SmartToDo_seq2seq$ ./gen_ToDo.sh
```
This will create vocaulary for training, train the model and generate To-Do items on the validation and test set. The performance metrics will also be computed and printed on screen.


## Additional Code

* [extractiveSummary](./src/extractiveSummary) - This folder contains code that was used to extract "Helpful" sentences using fastText, Term frequency (Tf) or BERT emebeddings.

* [wordEmbed](./src/wordEmbed) - This folder contains code for training fastText embeddings on raw Avocado sentences. 


## Licence
The data is released as per [MSR Data license](./MSR%20License_Data.docx). The code is released under [MIT license](./LICENSE).

## Acknowledgment

Our seq2seq code is heavily based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). We thank the authors for open-sourcing the NMT code.

## Feedback

Please feel free to email any feedback and/or suggestions about the code to sudipm@uw.edu or submukhe@microsoft.com.


