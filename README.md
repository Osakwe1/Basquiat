# Basquiat
We designed a Generative Adversarial Network for **image outpainting** as our project for the Le Wagon Data Science Bootcamp #1050. The primary inspiration for this was the recently introduced Outpainting feature by OpenAI's [DALL-E 2](https://openai.com/dall-e-2/).

## Introduction
Our code is written in Python 3.10, and we used the Google Console Vertex AI VM (TensorFlow Enterprise 2.10) with an NVIDIA T4 GPU, 4 vCPUs, and 15 GB of RAM. The training and test set for this project were the [Places365 Dataset](http://places2.csail.mit.edu/) provided by Bolei Zhou.

## Getting Started
If you need help setting up your copy of the project, click the drop down below. If not, skip ahead.

<details>
<summary>So how do I set it up?</summary>
<br>
These instructions will get you a copy of the project up and running on your local machine for testing and development purposes.

First of all, this project makes use of all the following python libraries and packages:

### Built With
In working on this project, the following libraries were utilized.
- Python 🐍 - Programming language
- Pandas 🐼 - Data manipulation library
- Numpy 🔢 - Scientific maths library
- scikit-learn 🗺️ - Machine Learning library
- Matplotlib 📊 - Data visualization library
- Seaborn 🌊 - Data visualization library
- Tensorflow 📈 - Machine Learning and AI library
- Pillow 🍃 - Image manipulation library
- Streamlit 📈 - App hosting site

  
* Clone the project repository to your local machine.
To set up your own local copy of this project, you will need to 'clone' this repo. To create a clone, run this in your terminal 
```
gh repo clone Osakwe1/Basquiat
```  
 
### Prerequisites
You will need to have Python 3 and the necessary libraries installed. You can install these libraries using pip by running the below :
```
pip install -r requirements.txt 
```
- Clone the project repository to your local machine.
- Open a command prompt or terminal window and navigate to the project directory.
- If you have VS Code or Jupyter Notebook, you can open up the folder by running:
```
code .  # for VS Code
```  
OR 
```
jupyter notebook  # for Jupyter Notebook
```  
If you do have either, I have linked the download link for [VS code](https://code.visualstudio.com/Download)   
</details>
Once in the root, you can run the site by running the below: 

```
streamlit run Outpainting/Streamlit/Input.py
```


## Model Architecture & Training
In designing this, we used a Conditional GAN comprising a Generator and Discriminator. The Generator produces outpaintings of masked images it deems to be 'realistic' based on the training set of images it has seen. The Discriminator identifies real images from the images created by the Generator and classifies them accordingly. The Discriminator returns feedback on the images it views as '1's and '0's, which is used to calculate the loss function.  
Using backpropagation, the model weights are then adjusted by calculating the weight's impact on the output. The training process is shown in detail below:

![Flowchart1 (2)](https://user-images.githubusercontent.com/42135459/207884696-c264280b-83bb-4954-87ca-5bbe242203f3.png)

Using the model architecture designed, and sufficient training, the model was capable of producing convincing recreations of test images:

***(L-R: After 1K steps, After 25K steps, After 50K steps, After 1M steps, After 2M Steps, Original Image)***

![image (1)](https://user-images.githubusercontent.com/42135459/208507107-d98454a4-c325-4920-8b8a-82e6d9069d96.png)

<!-- ![Screenshot 2022-12-13 at 21 02 26](https://user-images.githubusercontent.com/42135459/207443050-785caf12-4b7a-4a7c-873c-5e67dc67712a.png) -->


## Gallery
Here are some of our results, taken directly from our model!

***(L-R: Original image, Outpainted image)***

![combine_images](https://user-images.githubusercontent.com/42135459/207445184-bfe18405-a6d5-44f1-b533-cb81aeedb31a.jpg)
![combine_images (1)](https://user-images.githubusercontent.com/42135459/207445594-9664b888-baff-46aa-80b2-817d144b970c.jpg)
![combine_images](https://user-images.githubusercontent.com/42135459/207447971-4a186d78-e7ae-47fd-b128-aee0b4762b1c.png)


<!-- #### Special Thanks 
Special thanks are in order for Mark Botterill & Andrei Danila for their guidance and assistance throughout the project. I would also love to extend my gratitude to Catriona Beamish & Oliver Giles for their inspiration and encouragement throughout the pitch and feasibility process. -->

<!-- ## FrontEnd -->

<!-- This site was built & hosted using [Streamlit](https://streamlit.io/). -->
