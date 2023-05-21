import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
st.set_page_config(layout="wide", page_icon=":lower_left_paintbrush:", menu_items={
    'Report a bug': "mailto:OOsakwe1@icloud.com?subject=Bug%20found%20in%20Outpainting",
    'About': "# Built by Krishini, Louis, Hafiz & Olisa.",
    'Get Help': "https://github.com/Osakwe1/Outpainting"
    }
)

from preprocessing import preprocess,resize, convert_to_np
from load_model import load_model
from postprocessing import reduce_colors

def make_image():
    if option == 'left' or option == 'right':
        preprocessed_image,original_image = preprocess(uploaded_file,expand_side=option)

        filler = np.full((256,64,3),(255))

        model_path_gen_r = "Outpainting/Streamlit/weights.h5"
        model_uploaded = load_model(model_path_gen=model_path_gen_r)

        prediction = model_uploaded(preprocessed_image, training = True)

        output = reduce_colors(prediction[0],option_colours).astype(np.uint8)

        if option == 'left':

            image_input = np.hstack((filler,original_image))
            output = np.fliplr(output)

            output_full = np.hstack((output[:,:64,:],original_image))

        else:
            image_input = np.hstack((original_image,filler))
            output_full = np.hstack((original_image,output[:,192:,:]))

        with st.spinner('Outpainting...'):
            time.sleep(5)
        st.success('Done!')

        output_ready = Image.fromarray(output_full)
        output_ready.save('output_image.png')


        col1, col2, col3 = st.columns([1, 5, 1])
        col2.image(image_input, width = 600)

        col1, col2, col3 = st.columns([1, 5, 1])
        col2.image('output_image.png', width = 600)

        #now we need to do post-processing

    elif option == 'both sides':
        preprocessed_image_left,original_image = preprocess(uploaded_file,expand_side='left')
        preprocessed_image_right,original_image = preprocess(uploaded_file,expand_side='right')

        model_path_gen_r = "Outpainting/Streamlit/weights.h5"
        model_uploaded = load_model(model_path_gen=model_path_gen_r)

        prediction_left = model_uploaded(preprocessed_image_left, training = True)
        prediction_right = model_uploaded(preprocessed_image_right, training = True)

        output_left  = reduce_colors(prediction_left[0] ,option_colours).astype(np.uint8)
        output_right = reduce_colors(prediction_right[0],option_colours).astype(np.uint8)

        output_left = np.fliplr(output_left)

        output_full = np.hstack((output_left[:,:64,:],original_image,output_right[:,192:,:]))

        with st.spinner('Outpainting...'):
            time.sleep(5)
        st.success('Done!')

        output_ready = Image.fromarray(output_full)
        output_ready.save('output_image.png')

        filler = np.full((256,64,3),(255))
        image_input = np.hstack((filler, original_image,filler))

        col1, col2, col3 = st.columns([1, 5, 1])
        col2.image(image_input, width = 900)

        col1, col2, col3 = st.columns([1, 5, 1])
        col2.image('output_image.png', width = 900)

    return True

st.title("Basquiat")
st.subheader('Welcome to Basquiat, an Outpainting tool to expand images.')
st.markdown('This Conditional Generative Adversial Network was built by  by Krishini Patel, Louis Swynnerton, Abdulhafiz Alasa, & Olisa Osakwe')
st.markdown('This model is trained on 2 million images to produce realistic expansions of images. Try some sample images below or upload your own, have fun!' )


with st.expander('Check out our sample images', expanded=False):
    st.write("""
        Make a outpainting using one of the following sample images!
    """)

    with st.container():
   # You can call any Streamlit command, including custom components:
        col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Outpainting/Images/Places365_val_00000041.jpg")
        if st.button('1', key='1'):
            uploaded_file = st.image("Outpainting/Images/Places365_val_00000041.jpg")
        else:
            pass

        st.image("Outpainting/Images/Places365_val_00000041.jpg")
        if st.button('4', key='4'):
            uploaded_file = st.image("Outpainting/Images/Places365_val_00000348.jpg")
        else:
            pass

    with col2:
        st.image("Outpainting/Images/Places365_val_00000135.jpg")
        if st.button('2', key='2'):
            uploaded_file = st.image("Outpainting/Images/Places365_val_00000135.jpg")
        else:
            pass

        st.image("Outpainting/Images/Places365_val_00000996.jpg")
        if st.button('5', key='5'):
            uploaded_file = st.image("Outpainting/Images/Places365_val_00000996.jpg")
        else:
            pass

    with col3:
        st.image("Outpainting/Images/Places365_val_00000282.jpg")
        if st.button('3', key='3'):
            uploaded_file = st.image("Outpainting/Images/Places365_val_00000282.jpg")
        else:
            pass

        st.image("Outpainting/Images/Places365_val_00000748.jpg")
        if st.button('6', key='6'):
            uploaded_file = st.image("Outpainting/Images/Places365_val_00000748.jpg")
        else:
            pass
temp_uploaded = st.selectbox("Which sample image would you like to see?",(1,2,3,4,5,6))
option = "both sides"
option_colours = 100
if st.button('Generate Outpainted Sample'):
    if temp_uploaded == 1:
        uploaded_file = "Outpainting/Images/Places365_val_00000041.jpg"
    if temp_uploaded ==4:
        uploaded_file = "Outpainting/Images/Places365_val_00000348.jpg"
    if temp_uploaded==2:
        uploaded_file = "Outpainting/Images/Places365_val_00000135.jpg"
    if temp_uploaded==5:
        uploaded_file = "Outpainting/Images/Places365_val_00000996.jpg"
    if temp_uploaded==3:
        uploaded_file = "Outpainting/Images/Places365_val_00000282.jpg"
    if temp_uploaded==6:
        uploaded_file = "Outpainting/Images/Places365_val_00000748.jpg"
    make_image()


option = None
uploaded_file = None
uploaded_file = st.file_uploader("Upload Image here")
if uploaded_file is not None:
    option_colours =  st.slider(
    'Select a range of values for no. of colours', min_value = 10, max_value = 200,value = 80, step= 4)

    option = st.selectbox(
    'Which side would you like to outpaint',
    ('Select side', 'left', 'right','both sides'))

    st.write('You selected:', option)
    if option != "Select side":


        if st.button('Run',on_click=make_image):
            uploaded_file = 'output_image.png'
