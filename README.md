# TaxoRL

code for "End-to-End Reinforcement Learning for Automatic Taxonomy Induction" ACL 2018 [[arXiv]](https://arxiv.org/abs/1805.04044)

## Requirements

python **2.7**  
dynet 2.0  
tqdm  

## Data

**Preprocessed pickled data including everything else for the WordNet data can be downloaded [here](https://drive.google.com/file/d/1EXeMb69fcoQgiNORAXcg2vZPR7yBbjrY/view?usp=sharing).**

Preprocessed pickled data including everything else for the WordNet data and SemEval-2016 can be downloaded [here](https://drive.google.com/file/d/1p70QAe9yYD1kEAeDyjPvnfJKILaS2ZRl/view?usp=sharing). If you run on SemEval-2016, use [dev_twodatasets.tsv](https://drive.google.com/file/d/1n3XuwiXe3HQAl3ogDV0VI3FNe5MwOYt4/view?usp=sharing) instead of dev_wnbo_hyper.tsv.  Caution: it may take 40+ GB memory.

**DIY** 

- To build everything from scratch, first download corpora such as Wikipedia, [UMBC](https://ebiquity.umbc.edu/resource/html/id/351/UMBC-webbase-corpus), and [1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/).
- To preprocess the corpus, generate a vocabulary file and use the scripts under ./corpus/ to find dependency paths between terms in the vocabulary. The scripts are modified based on [LexNET](https://github.com/vered1986/LexNET). Instructions can be found [here](https://github.com/vered1986/LexNET/wiki/Detailed-Guide). It may take several hours to finish this process. 
- Run train_RL.py and it will compute all the features and save them into pickle files.

## Run

Run train_RL.py for training and testing. All the parameters are in *argparse* and have default values so that you can run without specifying any parameters (but feel free to tune them).

In each epoch, the performance on training/validation/test sets is reported. You may exit the program at any time.

## Cite

Citation info is on the way...
