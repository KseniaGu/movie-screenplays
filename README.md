# Movie screenplays
The project is dedicated to the analysis of screenplays using machine learning techniques. This work was carried out as part of my master's thesis.

### Repository content:
* Movie Scripts Corpus (uploaded to [kaggle.com](https://www.kaggle.com/dataset/e14349f732b3f35aa1bcb5fe68961b4a79a757bc5c84fe678acd0ffa69018c72), more information [here](https://www.kaggle.com/dataset/e14349f732b3f35aa1bcb5fe68961b4a79a757bc5c84fe678acd0ffa69018c72)) gathering and processing (`movie-screenplays/data/parsing/`).
* Parsed screenplay and movie data analysis (`movie-screenplays/ml_analysis/`). [BERT](https://arxiv.org/abs/1810.04805) was used as a core component for the models, or by itself. Multiple tasks were used for training in order to assess the usability of the gathered corpus:
  * Movie characters' screenplay text analysis (gender prediction, significance assessment)
  * Screenplay awards analysis (predicting whether a screenplay will win an award)
  * Movie genre prediction based on screenplay text
  
### BERT application:
A BERT-based annotator was developed and trained to extract the main elements of screenplays (such as headings and dialogues) from scraped data. These elements were then used to divide the texts into scenes. The combination of these scenes was passed through a siamese model, which was implemented based on the paper ['Phenotyping of Clinical Notes with Improved Document Classification Models Using Contextualized Neural Language Models'](https://arxiv.org/pdf/1910.13664.pdf).

*Update: a [ModernBERT](https://arxiv.org/abs/2412.13663) model's training pipeline has been added for script annotation experiments: `movie-screenplays/ml_analysis/notebooks/modernbert_annotator_kaggle.ipynb`.*  

### Dependencies:
* requests~=2.26.0 
* beautifulsoup4~=4.10.0 
* numpy~=1.21.2 
* pandas~=1.3.4 
* transformers~=4.14.1 
* scikit-learn~=0.23.2 
* tqdm~=4.62.3 
* torch~=1.10.1 
* mlflow~=1.22.0

### Usage:
Data processing pipelines are in the `data/parsing` directory, including:
1. `movie_characters` folder with movie characters analysis, processing and matching
2. `web_database` folfer with pipelines for collecting movie scripts and metadata 

The execution pipelines for ML analysis are in the notebooks from `ml_analysis/notebooks`:
  1. `bert_annotator.ipynb` notebook contains the annotation code used to extract screenplay elements from raw text. The labels for the extracted elements are added to the lines of the raw text and then used to extract scenes from the screenplay in the `ml_analysis/datasets` module
  2. `movie_screenplays_colab.ipynb` contains the training pipeline for the models from `ml_analysis/models` adjusted for Google Colaboratory usage 
  3. `modernbert_annotator_kaggle.ipynb` notebook contains updated with ModernBERT training pipeline for the movie screenplay elements annotation    