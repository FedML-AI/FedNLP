#!/bin/bash
declare -a data_names=("20news" "agnews" "cnn_dailymail" "cornell_movie_dialogue" "semeval_2010_task8" "sentiment140" "squad_1.1" "sst_2" "w_nut" "wikiner" "wmt_cs-en" "wmt_de-en" "wmt_ru-en" "wmt_zh-en")

declare -a data_dir_paths=("../../data/text_classification/20Newsgroups" "../../data/text_classification/AGNews" 
    "../../data/seq2seq/CNN_Dailymail" "../../data/seq2seq/CornellMovieDialogue/cornell_movie_dialogs_corpus" 
    "../../data/text_classification/SemEval2010Task8/SemEval2010_task8_all_data" "../../data/text_classification/Sentiment140" 
    "../../data/span_extraction/SQuAD_1.1" "../../data/text_classification/SST-2/trees" "../../data/sequence_tagging/W-NUT2017" 
    "../../data/sequence_tagging/wikiner" 
    "../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.zh,../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.en" 
    "../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs,../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en" 
    "../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.ru,../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.en" 
    "../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.de,../../data/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.en")


for i in "${!data_names[@]}"
do
	python ./test_rawdataloader.py --dataset ${data_names[$i]} --data_dir ${data_dir_paths[$i]} --h5_file_path ./${data_names[$i]}_data.h5
done

