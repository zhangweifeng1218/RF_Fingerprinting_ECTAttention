cfg = {
    # data
    'train_data_dir': 'data/USRPN2932/',
    'h5_file': 'data/USRPN2932/data_1trainset.h5',
    'sample_len': 512,
    'sample_overlap_random': False,
    'sample_overlap': 256,
    'n_classes': 20,
    # model
    'model': 'ECSA_ResNet50',
    'channel_attention': True,
    'spatial_attention': True,
    'checkpoint_path': 'check_point/USRPN2932/',
    'batch_size': 128,
    'n_epoch': 5,
    'lr': 4e-3,
}

