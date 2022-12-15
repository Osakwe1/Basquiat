# Basquiat
We designed a Generative Adversial Network for image outpainting as our project for the Le Wagon Data Science #1050. 

## Setup
Our code is written in Python 3.10, and we used the Google Console a Vertex AI VM (TensorFlow Enterprise 2.10) with an NVIDIA T4 GPU, 4 vCPUs, and 15 GB of RAM.


## Model Architecture
In designing this, we used a GAN comprising a Generator and Discriminator. 

The Discriminator identifies real images from the images created by the Generator and classifies them accordingly. Using backpropagation, the model weights are adjusted by calculating the weight's impact on the output.

![Flowchart1 (2)](https://user-images.githubusercontent.com/42135459/207884696-c264280b-83bb-4954-87ca-5bbe242203f3.png)

## Model Training
Using the designed model architecture and adjusting parameters to improve performance, we produced a model which produced realstic expansions of masked images. Our results through steps of training are shown below:
![Screenshot 2022-12-13 at 21 02 26](https://user-images.githubusercontent.com/42135459/207443050-785caf12-4b7a-4a7c-873c-5e67dc67712a.png)

The Ground Truth image for comparison is shown below:
![00000003](https://user-images.githubusercontent.com/42135459/207891404-c009c99f-fb27-4095-aa1c-ee1424793320.jpg)



## Gallery
Here are some of our results, taken directly from our model!

(L-R: Original image, Outpainted image)

![combine_images](https://user-images.githubusercontent.com/42135459/207445184-bfe18405-a6d5-44f1-b533-cb81aeedb31a.jpg)
![combine_images (1)](https://user-images.githubusercontent.com/42135459/207445594-9664b888-baff-46aa-80b2-817d144b970c.jpg)
![combine_images](https://user-images.githubusercontent.com/42135459/207447971-4a186d78-e7ae-47fd-b128-aee0b4762b1c.png)


## FrontEnd

This site was built & hosted using [Streamlit](https://streamlit.io/)


