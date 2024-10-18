import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from yamnet_inference import ESCYamnetInference
from model import NNModel,predict,NNModel2
from labels import ESC_CLASSES

# Title of the app
st.title("Audio Classification")

# Uploading an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

# Function to plot waveform using matplotlib
def plot_waveform(signal, sr, window_start=0, window_end=None):
    """Plot the waveform of the signal."""
    time = torch.arange(window_start, window_end) / sr
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time.numpy(), signal[window_start:window_end].numpy(), color='blue')
    ax.set_title("Waveform")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    
    # Convert the plot to an image to display in Streamlit
    buf = BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    st.image(buf)
    
yamnet_inference = ESCYamnetInference(r'yamnet.onnx', device='cpu')
model = NNModel(2048,50)
print(model)
# checkpoint = torch.load(r'esc-model1_2024-10-16_08-27-24_inp2048.pth',map_location=torch.device('cpu'))
# print(checkpoint.keys())  # Check the layer names and their shapes
model.load_state_dict(torch.load(r'esc-model1_2024-10-16_08-27-24_inp2048.pth',map_location=torch.device('cpu')))
model.eval()

# If a file is uploaded, process and display the waveform
if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Load the audio signal
    signal, sr = torchaudio.load("temp_audio.wav")

    # Ensure the audio is mono
    if signal.shape[0] == 2:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # Determine the length of the signal (in samples and seconds)
    signal_length = signal.shape[1]
    signal_duration = signal_length / sr

    # Check if the audio is shorter than 5 seconds
    if signal_duration <= 6:
        # st.warning("Audio is shorter than 5 seconds, displaying full waveform.")
        plot_waveform(signal.squeeze(), sr, 0, signal_length)  # Display the full waveform
        
        # Save the whole signal and re-write as a smaller file for audio player
        selected_signal = signal.squeeze()
        torchaudio.save("selected_audio.wav", selected_signal.unsqueeze(0), sr)
    else:
        print(signal_duration)
        # Add a slider to allow the user to scroll through the waveform
        st.subheader("Select a 5-second window")

        # Slider for selecting the start time in seconds (frame is fixed at 5 seconds)
        start_time = st.slider("Start time (seconds)", 0, int(signal_duration - 5), 0)

        # Calculate the start and end sample indices
        window_start = int(start_time * sr)
        window_end = window_start + (5 * sr)  # Display a 5-second window

        # Plot the selected window
        plot_waveform(signal.squeeze(), sr, window_start, window_end)
        
        # Extract the selected 5-second window from the signal
        selected_signal = signal.squeeze()[window_start:window_end]

        # Save the selected portion of the signal as a new audio file for the audio player
        torchaudio.save("selected_audio.wav", selected_signal.unsqueeze(0), sr)

    # Display the audio player for the selected window (or full audio if short)
    st.audio("selected_audio.wav", format="audio/wav")

    # Display success message
    st.success("Audio file successfully uploaded and played")
    
    # Add a button to run the classifier
    if st.button("Run classifier"):
        # Create a placeholder for the result
        # Create a placeholder for the result
        result_placeholder = st.empty()

        # Display a loading icon within the placeholder
        result_placeholder.markdown("<h3 style='text-align: center;'>⏳ Classifying...</h3>", unsafe_allow_html=True)
        embeddings = yamnet_inference.infer("selected_audio.wav")
        prediction = predict(model, embeddings.unsqueeze(0)).squeeze()
        # Simulate a delay (e.g., 0.5 seconds) to mimic the process
        # time.sleep(20)

        # Call the dummy classifier function to get a random prediction

        # Replace the loading icon with the actual result in the same placeholder
        result_placeholder.markdown(f"<h2 style='text-align: center; color: blue;'>Prediction: {ESC_CLASSES[int(prediction)]}</h2>", unsafe_allow_html=True)

