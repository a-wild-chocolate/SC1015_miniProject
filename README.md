# SC1015_miniProject--Mushroom Identifier 3000

## Contributors:
- @irene-Lijinglin: Li Jing Lin  
- @a-wild-chocolate: Mao Yan Yu
- @eskimofishes: Ng Jun Yu

## About:
Contained within are the source codes for our SC1015 (Introduction to Data Science and Artificial intelligence) Mini-Project. Our mini-project utilizes the mushroom classification dataset corresponding to 23 species of gilled mushrooms drawn from “The Audubon Society Field Guide to North American Mushrooms (1981)” taken from kaggle and 400 images of random mushrooms with classification from Wiki and MushroomWorld.

## Files Included:
1) Slides
The Powerpoint slides for the video.
2) Jupyter Notebook #1: Data Extraction, Exploratory Analysis, Machine Learning, User input.
The code in this notebook uses the mushroom classification dataset sourced from Kaggle. The order of presentation is as follows: (The code has to be loaded in order)
1)	The dataset is extracted and cleaned before being having the proportions of poisonous and edible mushrooms for each characteristic calculated, visualized and compared. 
2)	The Kmode method is further used to supplement the exploratory data analysis. 
3)	The Machine Learning algorithms used are decision tree model and random forest classification model, both having the data tuned for the models to be able to work with the categorical variables.
4)	A prototype of how a user-input system can work for the machine learning algorithms along with a sample input.
3) Jupyter Notebook #2: Convolution Neural Network (CNN)
4) DataSets:
Contained within is the dataset sourced from kaggle and a repository of the images of mushrooms sourced from the internet.

## Problem Definition:
The objective of this project is to develop machine learning algorithms and AI models to investigate the identifiable characteristics of a poisonous mushroom and use ML models to predict whether they are edible.


## Models Used:  
1)	Decision Tree Model
2)	Random Forest Classification
3)	Convolution Neural Network (CNN) 

## Conclusion:

There are many characteristics which when analysed shows that they are unique to poisonous mushrooms.  

Both the machine learning algorithms which used the Kaggle dataset were similar in their accuracy rate, 99 and 100% respectively while the CNN which uses images of random mushrooms only returned an accuracy rate of 80%. However, CNN is a lot easier to use in application as users can simply take a picture of any mushroom while the machine learning algorithms would require users to analyse and input 22 characteristics.   

The **Kmode algorithm and proportion method** used to analyse the data also returned relatively accurate key characteristics of poisonous mushrooms.  

The dataset from Kaggle is based a hypothetical data set from only 2 families of mushrooms which may have made the machine learning models overfitting.  

While the high demand of running CNN on home computers limited the number of epochs we were able to run which may have reduced the accuracy rate.  

## What did we learn from this project?  
As our dataset happened to be **entire categorical** while most of what we learnt in class was for numerical datasets, we learnt many methods to process and evaluate categorical data. Of which methods include:  
1)	One-Hot Encoder  
2)	KMode  
3)	Fit-Transform  
4)	LabelEncoder  
5)	PCA  
6)	Keras Image Augmentation API  
New Models learnt include:  
1)	Random Forest Classification  
2)	Convolution Neural Network (CNN)  




## References:
Bonthu, H. (2023, April 18). KModes clustering algorithm for categorical data. *Analytics Vidhya.* https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/

Chollet, B. F. (n.d.). Building powerful image classification models using very little data. *The Keras Blog ATOM.* https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Koivisto, T., Nieminen, T., & Harjunpää, J. (2018). *Deep Shrooms: classifying mushroom images.* https://tuomonieminen.github.io/deep-shrooms/

Li, H., Zhang, Y., Zhang, H., Zhou, J., Liang, J., Yin, Y., He, Q., Jiang, S., Zhang, Y., Yuan, Y., Lang, N., Cheng, B., Wang, M., & Sun, C. (2023). Mushroom poisoning outbreaks — China, 2022. *China CDC Weekly*, 5(3), 45–50. https://doi.org/10.46234/ccdcw2023.009

Li, H., Zhang, H., Zhang, Y., Zhang, K., Zhou, J., Yin, Y., Jiang, S., Ma, P., He, Q., Zhang, Y., Wen, K., Yuan, Y., Lang, N., Lu, J., & Sun, C. (2020, January 1). Mushroom poisoning outbreaks - China, 2019. *China CDC Weekly.* https://weekly.chinacdc.cn/en/article/doi/10.46234/ccdcw2020.005#:~:text=At%20least%20100%20estimated%20people,China%20(2%2D5).

Nuse, I. P., & Christensen, A. (2016, September 13). Finding mushrooms with your mobile phone. *Sciencenorway.* https://sciencenorway.no/forskningno-norway-smartphone/finding-mushrooms-with-your-mobile-phone/1437329

Sandhyakrishnan02. (2022, March 1). Mushroom Classification - Decision Tree Classifier. *Kaggle.* https://www.kaggle.com/code/sandhyakrishnan02/mushroom-classification-decision-tree-classifier/notebook#11.-Decision-Tree-Creation

scikit-learn developers. (2023). *sklearn.preprocessing.LabelEncoder.* Retrieved April 22, 2023, from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

Some disturbing facets of the super mario universe. (n.d.). *VGJUNK.* Retrieved April 23, 2023, from http://retrovania-vgjunk.blogspot.com/2012/08/some-disturbing-facets-of-super-mario.html

Tran, J. (2021, December 14). Random Forest Classifier in Python - Towards Data Science. *Medium.* https://towardsdatascience.com/my-random-forest-classifier-cheat-sheet-in-python-fedb84f8cf4f

Ujjwalkarn. (2017, May 29). An Intuitive Explanation of Convolutional Neural Networks. *Ujjwal Karn.* https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

Wikipedia contributors. (2023). Convolutional neural network. *Wikipedia.* https://en.wikipedia.org/wiki/Convolutional_neural_network

Wild Food UK. (2023, February 13). *Wild UK mushrooms (fungi): Guide to identification & picking.* Retrieved April 23, 2023, from https://www.wildfooduk.com/mushroom-guide/


