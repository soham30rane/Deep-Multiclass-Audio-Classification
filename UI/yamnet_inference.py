import torch
import torchaudio
import onnxruntime as ort

SAMPLE_RATE = 16000

TARGET_LENGTH = 5  # 5 seconds
TARGET_LENGTH_SAMPLES = SAMPLE_RATE * TARGET_LENGTH


# Load YAMNet for embeddings
class ESCYamnetInference:
    def __init__(self, yamnet_onnx_path, target_sample_rate=SAMPLE_RATE, device='cpu',mode="max-avg"):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.ort_session = ort.InferenceSession(yamnet_onnx_path)
        self.mode = mode
    
    def infer(self, audio_file_path):
        # Load and process the audio file
        signal, sr = torchaudio.load(audio_file_path)
        signal = signal.to(self.device)
        signal = self._fix_sample_rate(signal, sr)
        signal = self._pad_signal(signal)
        signal = signal.squeeze()
        signal = self._min_max_scaling(signal)
        
        # Get YAMNet embeddings
        embeddings = self._get_yamnet_embeddings(signal)
        return embeddings
    
    def _get_yamnet_embeddings(self, signal):
        # Convert signal to numpy and get embeddings from ONNX model
        inputs = {self.ort_session.get_inputs()[0].name: signal.numpy()}
        outputs = self.ort_session.run(None, inputs)
        _, embeddings, _ = outputs
        embeddings = torch.tensor(embeddings)

        # Apply max and average pooling
        max_pooled_embedding = torch.max(embeddings, dim=0)[0]
        avg_pooled_embedding = torch.mean(embeddings, dim=0)
        concatenated_embedding = torch.cat((avg_pooled_embedding, max_pooled_embedding), dim=0)
        
        if self.mode == "max":
            return max_pooled_embedding
        elif self.mode == "avg":
            return avg_pooled_embedding
        else:
            return concatenated_embedding

    def _min_max_scaling(self, signal):
        min_amp = torch.min(signal)
        max_amp = torch.max(signal)
        if min_amp != max_amp:
            signal = (signal - min_amp) / (max_amp - min_amp)
            signal = (signal * 2) - 1
        return signal

    def _fix_sample_rate(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _pad_signal(self,signal):
        # Convert to mono by averaging the two channels (if stereo)
        print(f"old  : {signal.shape}")
        if signal.shape[0] == 2:
            signal = torch.mean(signal, dim=0, keepdim=True)  # Convert stereo to mono

        # Ensure the signal has the correct length
        signal_length = signal.shape[1]

        if signal_length < TARGET_LENGTH_SAMPLES:
            # Pad with zeros if the signal is shorter than 5 seconds (80000 samples)
            padding_length = TARGET_LENGTH_SAMPLES - signal_length
            signal = torch.nn.functional.pad(signal, (0, padding_length))
        elif signal_length > TARGET_LENGTH_SAMPLES:
            # Trim the signal to the target length if it's longer than 5 seconds
            signal = signal[:, :TARGET_LENGTH_SAMPLES]
        torchaudio.save("out.wav",signal,SAMPLE_RATE)
        print(f"new  : {signal.shape}")
        return signal