_base_ = '../default.py'

expname = 'dvgo_cube'
basedir = './logs/deepvoxels'

data = dict(
    datadir='/data/nerf_data/deepvoxels/',
    dataset_type='deepvoxels',
    scene='cube',
    white_bkgd=True,
)

