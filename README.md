# SenSum: Enhance Unsupervised Video Summarization with Language-based Semantic Reward 

## Overview
This repository contains the code and resources (data) for my ELENE6885 Reinforcement Learning project. The project investigates the effectiveness of a language-based semantic reward in video summarization.

### Team Members
| Name           | Email Address          |
|----------------|----------------|
| Lilin Xu  | lx2331@columbia.edu  | 


## Introduction
With the exponential growth of video data, video summarization has become crucial for generating concise yet informative summaries. Reinforcement learning (RL)-based solutions have regained attention by formulating video summarization as a sequential decision-making problem. Existing RL-based methods either rely on frame-level clustering, ignoring overall video content, or depend on labeled data. We propose SenSum, a content-aware RL-based solution that incorporates LLMs to effectively utilize both visual and textual information with unsupervised learning. By integrating visual rewards and a designed semantic reward derived from textual information, SenSum effectively enhance RL-based frame selection. Experiments on two public datasets show that SenSum outperforms state-of-the-art RL-based solutions by up to 5.3\%.


---

## Results
The results of **SenSum** and figures (generated summary, importantce score, reward curves) for experiments can be viewed and obtained by running the code in `results.ipynb`.

## How to Run the Project

### 1. Prepare the Environment
Set up the required environment by running:
```bash
conda env create -f environment.yml
conda activate rl
```

### 2. Prepare datasets
Please download the preprocessed datasets files `.h5` [here](https://drive.google.com/drive/folders/1VtyGJePG2vfsTLPtcOjb3oGSMdCQW9Gn?usp=sharing) and place them under **`datasets` folder**.

**Optional:**
You can download datasets with original videos [here](https://drive.google.com/drive/folders/1sbZZalh43n6fiSxWt_SIGgv72bt4rdoG) (`SumMe.zip` and `tvsum50_ver_1_1.tgz`) (from [Summarizer](https://github.com/sylvainma/Summarizer) project) and place them under **`dataset_img` folder**. After extracting the frames of the videos with `FFmpeg`, you can use **`exp_frame_select.ipynb`** under the folder to downsample the videos.

This is optional, since the visual features are stored in the `.h5` files and the captions and textual features are provided in **`dataset_img2text` folder**.


### 3. Code Structure
- **`dataset_img` folder**: Stores original videos and downsampled images.
- **`datasets` folder**: Stores preprocessed datasets `.h5` files.
- **`dataset_img2text` folder**: Stores frame-level captions and video-level summaries of these captions, as well as textual features of these captions and summaries.
    - **`*.tar.gz`**: Stores frame-level captions and video-level summaries in `.json` files.
    - **`text_embedding` folder**: Stores textual features extracted from the captions and summaries.
    - **`*_llava16.py`**: Generates frame-level captions for videos.
    - **`caption_summizer.ipynb`**: Generates video-level summaries based on captions.
    - **`caption_embedding.ipynb`**: Extracts textual features.
- **`results.ipynb`**: The results of **SenSum** and figures.
- **`main.py` folder**: Runs experiments. Results are saved in the `log` folder.
- **`rewards.py` folder**: Contains visual rewards and the semantic reward.


### 4. Train
To train the model, you can run:

**TVSum**
```bash
python python main.py -d datasets/eccv16_dataset_tvsum_google_pool5.h5 -s datasets/tvsum_splits.json -m tvsum --gpu 0 --save-dir log/test/tvsum-split0-e60 --split-id 0 --verbose --save-results --max-epoch 60 --rnn-cell gru --text_embedding './dataset_img2text/text_embedding/tvsum/llava-v1.6-mistral-7b-hf/v1/mpnet/'
```

**SumMe**
```bash
python main.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/summe_splits.json -m summe --gpu 0 --save-dir log/test/summe-split0-e60 --split-id 0 --verbose --save-results --max-epoch 60 --rnn-cell gru --text_embedding './dataset_img2text/text_embedding/summe/llava-v1.6-mistral-7b-hf/v1/mpnet/'
```

---

