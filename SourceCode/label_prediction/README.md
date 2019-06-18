# AutomaticFactVerification
web search project

## Fine-tuned BERT model for Sentence Selection
Folder TrainSentence is the code for training the fine-tuned BERT model for sentence selection. 
BERT finetune folder include a python script run_classifier which is provided by Google on 
<https://github.com/google-research/bert>  
This script is modified to used for our downstream sentence pair classification task.  
The Generate result notebook is used to combine the result that BERT generated (probabilities) with the original claims. 

##  Label Prediction
### Preprocess wiki texts
The notebook GlobalPreprocess.ipynb is used to clean the brackets (like -LRB- -RRB-) and non-English texts in the original wiki texts provided and output the processed texts to a new file folder. 

### Get evidence text according to sentence head
The notebook file GetEvidence.ipynb is used to extract the original text from the processed wiki files and put the texts in the json stream for next step training.  
Tfidf\_MLP.ipynb is the implement of Method 4.1 in the report.  
BERT_MLP.ipynb is the implement of Method 4.2 in the report.  
run\_classifier.py is the modified entrance script for BERT fine-tune, which corresbonding to Method 4.3 in the report.  

-There are several middle files and script are not put in the folder together. For the whole code of this part, please refer to our github repo.   
<https://github.com/AlanChaw/LabelPrediction>

