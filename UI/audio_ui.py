import streamlit as st
import random
import time
import torch
from yamnet_inference import ESCYamnetInference
from model import NNModel,predict
from labels import ESC_CLASSES


# Title of the app
st.title("Audio classification")

# Uploading an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3","m4a"])

# Dummy classifier function that randomly returns "cat" or "dog"
def dummy_classifier():
    return random.choice(["cat", "dog"])

yamnet_inference = ESCYamnetInference(r'yamnet.onnx', device='cpu')
model = NNModel(2048,50)
print(model)
# checkpoint = torch.load(r'esc-model1_2024-10-16_08-27-24.pth',map_location=torch.device('cpu'))
# print(checkpoint.keys())  # Check the layer names and their shapes
model.load_state_dict(torch.load(r'esc-model1_2024-10-16_08-27-24.pth',map_location=torch.device('cpu')))
model.eval()

# If a file is uploaded, play the audio
if uploaded_file is not None:
    
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
        
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Display success message
    st.success("Audio file successfully uploaded and played!")
    
    # Add a button to run the classifier
    if st.button("Run classifier"):
        # Create a placeholder for the result
        # Create a placeholder for the result
        result_placeholder = st.empty()

        # Display a loading icon within the placeholder
        result_placeholder.markdown("<h3 style='text-align: center;'>⏳ Classifying...</h3>", unsafe_allow_html=True)
        embeddings = yamnet_inference.infer("temp_audio.wav")
        prediction = predict(model, embeddings.unsqueeze(0)).squeeze()
        # Simulate a delay (e.g., 0.5 seconds) to mimic the process
        # time.sleep(20)

        # Call the dummy classifier function to get a random prediction

        # Replace the loading icon with the actual result in the same placeholder
        result_placeholder.markdown(f"<h2 style='text-align: center; color: blue;'>Prediction: {ESC_CLASSES[int(prediction)]}</h2>", unsafe_allow_html=True)
