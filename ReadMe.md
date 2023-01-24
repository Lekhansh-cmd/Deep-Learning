1. Final.ipynb : Trained model is loaded and tested to check whether a text is sexist or not
2. Bert-base-uncased :  Bert model is trained for task A, B and C on processed and unprocessed data.
3. Machine-Learning : Logistic Regression, SVM and Random Forest are trained on Task A, B, and C.
4. Roberta-base : Roberta model is trained for task A, B and C on processed and unprocessed data.
5. Visualize-processed.ipynb : distilbert-base-uncased is trained on unprocessed data for 3 tasks.
6. Visualize.ipynb : distilbert-base-uncased is trained on processed data for 3 tasks.

**OUTLINE**
In this project, we try our hands on the task - ‘Explainable Detection of Online Sexism (EDOS)’, for which we are required to predict the sexist sentiments of a text and flag these appropriately.

The task contains three hierarchical subtasks:
TASK A:  Binary Sexism Detection: Sexist or Not Sexist
TASK B:  4-class classification of sexist posts: Threats, Derogation, Animosity, Prejudiced Discussion
TASK C: 11-class classification where systems have to predict one of 11 fine-grained vectors. These are the explainable features of Task A.

**PRE-PROCESSING DONE:**
Since we can not get any meaningful inferences from raw data, we pre-process the data. This step contains cleaning the data, balancing the data, and getting rid of unnecessary columns, etc.
Since we are really low and biased on the dataset,  some of the techniques used:
1. Undersampling for heavily skewed data
2. Data Augumentation using Back Translation

**MACHINE LEARNING MODELLING**
Before we move to the deep learning model, we try traditional machine learning techniques as a benchmark on our dataset.
1. Logistic Regression
2. Support Vector Machines
3. Random Forest
We compared the results from these models using a confusion matrix and three indicators: Precision, Recall, and Accuracy.
Random Forest (Task A and B)
Logistic Regression (Task C)

**DEEP LEARNING:** Transformer based Pretrained Model
Advantages:
1. Trained on millions of parameters
2. More Accuracy.
3. Process data in lesser time.
4. Works with any type of sequential data.

**BERT-BASE-UNCASED**	
- 12-layer, 768-hidden, 12-heads, 110M parameters. 
- Trained on lower-cased English text.
- It has this special ability to read in both directions simultaneously. Thus Bi-directional.
- It combines Mask Language Model (MLM) and Next Sentence Prediction (NSP).
- Easy route to using pre-trained models (transfer learning).

**DISTILBERT-BASE-UNCASED**	
- 6-layer, 768-hidden, 12-heads, 66M parameters
- The DistilBERT model distilled from the BERT model bert-base-uncased checkpoint
- Small, fast, cheap and light Transformer model.
- Runs 60% faster while preserving over 95% of BERT’s performances.

**ROBERTA-BASED**	
- 12-layer, 768-hidden, 12-heads, 125M parameters
- RoBERTa using the BERT-base architecture
- Only uses Masked Language Model

**CONCLUSION**
So we can conclude BERT was the best performing model for all of the tasks, where Roberta showed a slightly better result than distilBERT for Task C.
We got the best accuracy with BERT-base-uncased which was approx. 89%
The poor results can be attributed to fewer data, poor fine-tuning, and hyper-parameter selection.
As for future work, we are trying to get rid of the fallacies in our approaches to get competitive results.
