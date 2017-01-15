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
    parser.add_argument('--word_thrh', type=int, default=5,                 # word_thrh
                        help='Word counts threshold.')
    parser.add_argument('--batch_size', type=int, default=256,              # batch_size
                        help='Initial learning rate.')
    parser.add_argument('--ctx_shape', type=tuple, default=(196, 512),      # ctx_shape
                        help='Context shape for attention.')
    parser.add_argument('--dim_ctx', type=int, default=512,                 # dim_ctx
                        help='Context dimension for attention.')
    parser.add_argument('--dim_emb', type=int, default=256,                 # dim_emb
                        help='Embedded dimension.')
    parser.add_argument('--dim_hid', type=int, default=256,                 # dim_hid
                        help='Hidden layer dimension (LSTM).')
    parser.add_argument('--bias_init', type=bool, default=True,             # bias_init
                        help='Use bias init vector or not.')
    parser.add_argument('--pretrained_path', type=str, default=None,        # pretrained_path
                        help='Pretrained model path.')
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))
    return FLAGS

def config_test():
    FLAGS = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',                    # data_dir
                        help='Processed data directory.')
    parser.add_argument('--model_path', type=str, default='log/model-epoch-6',      # model_path
                        help='The model we are going to test.')
    parser.add_argument('--word_thrh', type=int, default=80,                        # word_thrh
                        help='Word counts threshold.')
    parser.add_argument('--ctx_shape', type=tuple, default=(196, 512),              # ctx_shape
                        help='Context shape for attention.')
    parser.add_argument('--dim_ctx', type=int, default=512,                         # dim_ctx
                        help='Context dimension for attention.')
    parser.add_argument('--dim_emb', type=int, default=256,                         # dim_emb
                        help='Embedded dimension.')
    parser.add_argument('--dim_hid', type=int, default=256,                         # dim_hid
                        help='Hidden layer dimension (LSTM).')
    parser.add_argument('--img_path', type=str, default=None,                       # img_path
                        help='The image we are going to generate caption of.')
    parser.add_argument('--eval_all', type=bool, default=False,                     # eval_all
                        help='Generate captions for all the modified test set.')
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))
    if not FLAGS.eval_all and FLAGS.img_path == None: sys.exit('If not eval_all, please set --img_path')
    return FLAGS
