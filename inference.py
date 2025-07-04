import glob
import os
import argparse
import json
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import MambAttention
import soundfile as sf
import time

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
)

h = None
device = None

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = MambAttention(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ):
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(os.listdir( args.input_folder )):
            #print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)
            sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='VB-DemandEx/noisy_test')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='MambAttention/checkpoints/MambAttention_seed3441_VB-DemandEx.yaml')
    parser.add_argument('--checkpoint_file', default='MambAttention/checkpoints/seed3441.yaml', required=True)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        #device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    start = time.time()
    inference(args, device)
    print(time.time() - start)

if __name__ == '__main__':
    print("Initializing inference...")
    main()
