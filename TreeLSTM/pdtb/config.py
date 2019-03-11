import socket

hostname = socket.gethostname()


class PathConfig:
    if hostname == "Precision-Tower-5810":  # linux 服务器
        pipe_data_dir = '/home/dejian/data/pdtb_v2/pipe'
        json_data_dir = '/home/dejian/data/pdtb_v2/pickle'

        embedding_path = '/home/dejian/data/glove/glove.6B.50d.txt'
    else:
        # pipe_data_dir = 'D:\\data\\pdtb_v2\\pipe'
        # json_data_dir = 'D:\\data\\pdtb_v2\\pickle'
        pipe_data_dir = 'D:\\mygit\\TagNN-PDTB\\data\\mysmall\\pipe'
        json_data_dir = 'D:\\mygit\\TagNN-PDTB\\data\\mysmall\\pickle'

        embedding_path = 'D:\\data\\glove.6B\\glove.6B.50d.txt'

    experiment_data_dir = '../data/pdtb'
    train_sections = set(list(range(2, 21)))  # {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    dev_sections = {0, 1}
    test_sections = {21, 22}
    vocab_path = experiment_data_dir + '/vocab.data'

    best_model_path = experiment_data_dir + '/model.info'


class ModelConfig:
    tag_embed_dim = 50
    # lstm_hidden_size = 250
    lstm_hidden_size = 50