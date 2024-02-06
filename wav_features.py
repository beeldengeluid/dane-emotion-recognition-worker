import os
import torch
import torchaudio
from fairseq.models import wav2vec
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

def process_wav_file(wav_path, output_path, model, num_frames=800000):
    try:
        # Check if the output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {wav_path} as {output_path} already exists.")
            return

        # Load waveform in chunks to avoid loading the entire file into memory
        waveform, sample_rate = torchaudio.load(wav_path, num_frames=num_frames)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        with torch.no_grad():
            features = model(waveform)

        torch.save(features, output_path)
        print(f"Processed {wav_path}")
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    try:
        # Specify the path to the Wave2Vec model checkpoint
        model_checkpoint_path = '/home/kdeboer/sseft/pre-trained_models/wav2vec_large.pt'

        # Load the model checkpoint
        state = load_checkpoint_to_cpu(model_checkpoint_path)
        args = state['args']

        # Build the model
        model = wav2vec.Wav2VecModel.build_model(args, task=None)
        model.load_state_dict(state['model'], strict=True)
        model.eval()

        # Specify input and output directories
        input_directory = '/home/kdeboer/MELD_data/temp_audio'
        output_directory = '/home/kdeboer/MELD_data/temp_audio_feat'

        os.makedirs(output_directory, exist_ok=True)

        # Process WAV files one at a time
        files = sorted([f for f in os.listdir(input_directory) if f.endswith('.wav')])
        for wav_file in files:
            wav_path = os.path.join(input_directory, wav_file)
            output_path = os.path.join(output_directory, f"{os.path.splitext(wav_file)[0]}.pt")

            # Process current WAV file
            process_wav_file(wav_path, output_path, model)
    except Exception as e:
        print(f"Error: {e}")
