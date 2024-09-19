# NRMS
This is a Pytorch Implementation of original NRMS algorithm (https://aclanthology.org/D19-1671.pdf). The algorithm is modified such that it uses Pre-trained Distil-BERT embeddings instead of GloVe. The improvement of performance is minor. After probing (logistic regression on final news embeddings to predict news topics), the improvement is likely from the model's improved ability to understand topics of news articles. We believe that the improvement can be larger if we are able to use the full news content instead of using only the news titles. To learn more, please check the presentation PDF.

![alt text](https://github.com/marktqh/NRMS/blob/BERT/project-nrms.png?raw=true)
