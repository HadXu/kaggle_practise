class DefaultConfig(object):
    env = 'default'
    model = 'AlexNet'

    train_data_root = './data/train'
    val_data_root = './data/train'
    test_data_root = './data/test1'
    load_model_path = None

    batch_size = 32
    use_gpu = False
    num_workers = 4
    print_freq = 20

    max_epoch = 10

    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

