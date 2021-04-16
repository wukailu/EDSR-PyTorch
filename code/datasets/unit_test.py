from datasets import DataProvider

if __name__ == "__main__":
    hparams = {
        "dataset": {
            'name': "DIV2K",
            'patch_size': 96,
            'ext': 'sep',
            'scale': 2,
            "batch_size": 16,
            'test_bz': 1,
            'repeat': 2,
        },
    }

    provider = DataProvider(hparams['dataset'])
    train_ds = provider.train_dl
    for x, y, _ in train_ds:
        print(x.shape, y.shape)

    for batch_idx, batch in enumerate(provider.test_dl):
        pass
