""" arguments definition """

import argparse

def params_setup():
    """ arguments definition """

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--attention_mode', type=str, default='bahdanau', help='')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='')
    parser.add_argument('--init_scale', type=float, default=0.1, help='')
    parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='')
    parser.add_argument('--num_units', type=int, default=128, help='') # 128
    parser.add_argument('--num_layers', type=int, default=3, help='') # 3
    parser.add_argument('--batch_size', type=int, default=32, help='') # 32
    parser.add_argument('--encoder_vocab_size', type=int, default=30000, help='')
    parser.add_argument('--decoder_vocab_size', type=int, default=86000, help='')
    parser.add_argument('--embedding_size', type=int, default=128, help='') # 128
    parser.add_argument('--max_len', type=int, default=50, help='')
    parser.add_argument('--debug', type=int, default=0, help='')
    parser.add_argument('--beam_search', type=int, default=1, help='')
    parser.add_argument('--beam_width', type=int, default=2, help='')

    para = parser.parse_args()

    return para
