# LID-IndoMalay
Languange Identification (LID) for Indonesian-Malaysian

anguage Identification aims to guess or identify which language the text or sound is coming. Language identification tends to be easier in languages with different characteristics (e.g., Indonesian and English). But not for languages with similar characteristics (e.g., Indonesian and Malaysian). Similar languages can cause ambiguity that will be a bias for machine learning. Using the Support Vector Machine (SVM) technique, this research tried to identify the Indonesian-Malaysian language. The training data is taken from the Leipzig Corpora Collection, while the test data is from the Twitter data set. The feature representation technique uses TF-IDF, and the baseline testing uses Naive Bayes Multinomial. We used two training techniques: split (20:80) and 10-cross validation. Based on the test results, the difference in accuracy between the baseline and SVM is not too far. Both provide accuracy of around 90% and above. The results of this study indicate that the accuracy of identification of Indonesian and Malaysian languages is relatively high even though using simple techniques.
