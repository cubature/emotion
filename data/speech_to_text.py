# -*- coding: UTF-8 -*-

import os
import sys

import deepspeech.client as deepspeech

# path_of_speech = "./Audio_Speech_Actors_01-24/"

def main():
    parser = deepspeech.argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=True,
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--alphabet', required=True,
                        help='Path to the configuration file specifying the alphabet used by the network')
    parser.add_argument('--lm', nargs='?',
                        help='Path to the language model binary file')
    parser.add_argument('--trie', nargs='?',
                        help='Path to the language model trie file created with native_client/generate_trie')
    parser.add_argument('--audio', required=True,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--version', action=deepspeech.VersionAction,
                        help='Print version and exits')
    args = parser.parse_args()

    print('Loading model from file {}'.format(args.model), file=sys.stderr)
    model_load_start = deepspeech.timer()
    ds = deepspeech.Model(args.model, deepspeech.N_FEATURES, deepspeech.N_CONTEXT, args.alphabet, deepspeech.BEAM_WIDTH)
    model_load_end = deepspeech.timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if args.lm and args.trie:
        print('Loading language model from files {} {}'.format(args.lm, args.trie), file=sys.stderr)
        lm_load_start = deepspeech.timer()
        ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, deepspeech.LM_ALPHA, deepspeech.LM_BETA)
        lm_load_end = deepspeech.timer() - lm_load_start
        print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)

    fin = deepspeech.wave.open(args.audio, 'rb')
    fs = fin.getframerate()
    if fs != 16000:
        print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
        fs, audio = deepspeech.convert_samplerate(args.audio)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/16000)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = deepspeech.timer()
    print(ds.stt(audio, fs))
    inference_end = deepspeech.timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

    return ds.stt(audio, fs)

if __name__ == '__main__':
    main()