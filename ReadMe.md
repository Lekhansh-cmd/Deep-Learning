1. Final.ipynb : Trained model is loaded and tested to check whether a text is sexist or not
2. Bert-base-uncased :  The Bert model is trained for tasks A, B, and C on processed and unprocessed data.
3. Machine-Learning : Logistic Regression, SVM and Random Forest are trained on Task A, B, and C.
4. Roberta-base : Roberta model is trained for tasks A, B and C on processed and unprocessed data.
5. Visualize-processed.ipynb : distilbert-base-uncased is trained on unprocessed data for 3 tasks.
6. Visualize.ipynb : distilbert-base-uncased is trained on processed data for 3 tasks.
7. *train_all_tasks.csv : Data on which models are trained*

**OUTLINE**
This project revolves around the task of "Explainable Detection of Online Sexism (EDOS)," focusing on predicting sexist sentiments in textual data sourced from platforms such as Reddit and Gab. The primary objective is to develop a robust Multi-Class Text Classification system capable of accurately flagging instances of sexism. 
Link: https://codalab.lisn.upsaclay.fr/competitions/7124

**1. OBJECTIVE**
This project aims to identify the most effective model for the task of detecting sexist sentiments in online content (specifically from Reddit and Gab) through Multi-Class Text Classification. This entails exploring and evaluating both traditional Machine Learning Models and Deep Learning Models to determine the optimal approach for our dataset.

The task contains three hierarchical subtasks:
TASK A:  Binary Sexism Detection: Sexist or Not Sexist
TASK B:  4-class classification of sexist posts: Threats, Derogation, Animosity, Prejudiced Discussion
TASK C: 11-class classification where systems have to predict one of 11 fine-grained vectors. These are the explainable features of Task A.

![Screenshot (252)](https://github.com/Lekhansh-cmd/Deep-Learning/assets/78807364/22a3d5ce-990d-421f-a5d4-3cbc5e4b1a76)

**PRE-PROCESSING DONE:**
We can not get any meaningful inferences from raw data, we pre-process the data. This step involves cleaning the data, balancing the data, and getting rid of unnecessary columns, etc.
Since we are low and biased on the dataset,  some of the techniques used:
1. Undersampling for heavily skewed data: Since the dataset was heavily skewed on some of the labels, it was very important to perform undersampling to remove the biases in our model.
2. Data Augmentation using Back Translation

![Screenshot (253)](https://github.com/Lekhansh-cmd/Deep-Learning/assets/78807364/fb6a7d9b-8eb8-4f6b-baf1-86109d7039cd)

**MACHINE LEARNING MODELLING**
Before we move to the deep learning model, we try traditional machine learning techniques as a benchmark on our dataset.
1. Logistic Regression
2. Support Vector Machines
3. Random Forest
We compared the results from these models using a confusion matrix and three indicators: Precision, Recall, and Accuracy.
Below are the best accuracy we achieved using the Machine Learning Models
Random Forest (Task A and B): Accuracy - 81% and 72% respectively
Logistic Regression (Task C): Accuracy - 69% 


**DEEP LEARNING:** Transformer based Pre-trained Model
Advantages:
1. Trained on millions of parameters
2. More Accuracy.
3. Process data in less time.
4. Works with any type of sequential data.

![Screenshot (256)](https://github.com/Lekhansh-cmd/Deep-Learning/assets/78807364/ba78fa62-5736-4691-bfb3-ede50670d5fe)

**BERT-BASE-UNCASED**	
- 12-layer, 768-hidden, 12-heads, 110M parameters. 
- Trained on lower-cased English text.
- It has this special ability to read in both directions simultaneously. Thus Bi-directional.
- It combines the Mask Language Model (MLM) and Next Sentence Prediction (NSP).
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

![Screenshot (254)](https://github.com/Lekhansh-cmd/Deep-Learning/assets/78807364/fec05544-a4db-4712-a1dc-f8e1d1870acc)

So we can conclude BERT was the best-performing model for all of the tasks, where Roberta showed a slightly better result than distilBERT for Task C.
We got the best accuracy with BERT-base-uncased which was approx. 89%
The poor results can be attributed to fewer data, poor fine-tuning, and hyper-parameter selection.
As for future work, we are trying to get rid of the fallacies in our approaches to get competitive results.
