# Movie screenplays
The project is dedicated to screenplay analysis using machine learning methods. This work was done as a part of my master's thesis.

### Repository content:
* Movie Scripts Corpus (uploaded to [kaggle.com](https://www.kaggle.com/dataset/e14349f732b3f35aa1bcb5fe68961b4a79a757bc5c84fe678acd0ffa69018c72), more information [here](https://www.kaggle.com/dataset/e14349f732b3f35aa1bcb5fe68961b4a79a757bc5c84fe678acd0ffa69018c72)) gathering and processing (`movie-screenplays/data/parsing/`).
* Parsed screenplay and movie data analysis (`movie-screenplays/ml_analysis/`). [BERT](https://arxiv.org/abs/1810.04805) was used as core component for models (or by its own). Multiple tasks were set and used for training to assess gathered corpus usability:
  * movie characters screenplay texts analysis (character gender predicting, character significance assessing),
  * screenplay awards analysis (predicting if a screenplay get an award),
  * movie genre predicting by screenplay text.
  
  BERT-based annotator was created and trained to extract screenplay main elements (like headings and dialogs) from scraped data. Then they were used to divide texts into scenes. Combination of scenes were passed through siamese model (one was implemented based on ['Phenotyping of Clinical Notes with Improved Document Classification Models Using Contextualized Neural Language Models'](https://arxiv.org/pdf/1910.13664.pdf) paper).
