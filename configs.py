import argparse
import sys

def config_train():
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',            # data_dir
                        help='Processed data directory.')
    parser.add_argument('--log_dir', type=str, default='log/',              # log_dir
                        help='Directory to save the log/models.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,        # lr
                        help='Initial learning rate.')
    parser.add_argument('--epoch', type=int, default=1000,                  # epoch
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=80,               # batch_size
                        help='Initial learning rate.')
    parser.add_argument('--ctx_shape', type=list, default=[196, 512],       # context_shape
                        help='Context shape for attention.')
    parser.add_argument('--dim_ctx', type=int, default=512,                 # dim_ctx
                        help='Attention context dimension.')
    parser.add_argument('--dim_emb', type=int, default=256,                 # dim_emb
                        help='Embedded dimension.')
    parser.add_argument('--dim_hid', type=int, default=256,                 # dim_hid
                        help='Hidden layer dimension (LSTM).')
    parser.add_argument('--pre-trained-path', type=str, default=None,       # dim_hid
                        help='Given pretrained model path')
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))
    return FLAGS
