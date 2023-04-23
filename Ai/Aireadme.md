This is an AI readme file records the process of CNN model construction  
# Preparation
We manually collected images from Wiki and Mushroomworld. There are 141 edible mushroom and 158 poisonous mushroom in total. Besides, we prepare 16 images for extra test. 
# Model 1.0 -- MushroomAI1.0.ipynb
We first create AI 1.0. This is a trial for CNN image recongition. We use **turicreate** library to train the model. Turicareate is a user-friendly library. We don't need to adjust the model itself. And it will automatically develope the custom machine learning models. However, this model's acuuracy is not very high. In the inner test case it's acuuracy is 79%, but in the outside test files it's accuracy is only 55%. So we give up the first model.  

# Model 3.0 -- MushroomAi3_0.ipynb
Next, we use **keras** to develop the second model. We use google colab with google drive to deal with our image data due to the size of file.  

Firstly, we **increased the number of image** due to the machine learning requirement. The purpose of this is trying to "cheat" ai by introducing small distortions to the images.  
We convert image to a NumPy array for the convenience of processing.`x = img_to_array(img)`
Then, we use the datagen.flow() method to generate augmented images and save it into the google drive. `for batch in datagen.flow(x, batch_size=1, save_to_dir='/content/drive/My Drive/Picture/TranEdible', save_prefix=Eprefix, save_format='jpeg'):  
            i += 1. 
            if i > 20:  
                break`   
We increased our image number to 5564 in total  

Next, we develop the core model of the CNN, below is our model's structure:
![/Ai/CNN Stucture.png](https://github.com/a-wild-chocolate/SC1015_miniProject/blob/main/Ai/CNN%20Stucture.png)

The 50th epoch test accuracy of our model is 0.7286. And the performance of our model in training processing is shown below:  
![Training Processing](https://github.com/a-wild-chocolate/SC1015_miniProject/blob/main/Ai/training_history.png)  
From this line chart，we can learn that As the train accuracy reaches a stage at **85%** , the testing accuracy waves around **80%**.   

To find out the best train times, we transfer the model performance into panda DataFrame. And we can get that the best training time (Epoch) is **45** with **86%** test accuracy.
This could help us improve the model by changing the epoch number.  

Furthermore, we create an **application** for whis model.
The **predict_and_draw** function would read one image at a time and generate the prediction as the title.  
We use the extra test case to verify this model as an example, hence we add the actual statue as a comparison. Below is one example it predicts:  
![Example of prediction](https://github.com/a-wild-chocolate/SC1015_miniProject/blob/main/Ai/prediction_example.jpg)

# Files Included:
 - Picture: the raw images we collected with classification.
 - testPicture: the extra test image
 - MushroomAI1.0.ipynb: the first AI model
 - MushroomAi3_0.ipynb: the final AI model
 - history: A binary text file records the final AI model
 - model.json: A json file records the final AI model
 - prediction_example.jpg: One example of the prediction generate by our AI model
 - training_history.png: The line chart of the performance change during the training of the AI
 - CNN Stucture.png: The structure of our CNN model  

# Reference
1. Chollet, F. (n.d.). The keras blog. The Keras Blog ATOM. Retrieved April 22, 2023, from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.htm
2. Koivisto, T., Nieminen, T., & Harjunpää, J. (2018). Deep Shrooms: classifying mushroom images. Retrieved April 22, 2023, from https://tuomonieminen.github.io/deep-shrooms/
3. Ujjwalkarn. (2017, May 29). An Intuitive Explanation of Convolutional Neural Networks. Ujjwal Karn. https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/ 


 
 
 -  
