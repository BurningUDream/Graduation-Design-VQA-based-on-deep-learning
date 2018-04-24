import os
import glob
import numpy as np
import h5py
import progressbar

root='../data/vqa02/'

def main(root):
    for split in ['train2014','val2014']:
        path='{}_feature_2'.format(split)
        trans(os.path.join(root,path), root, split)


def trans(from_dir, to_dir, split):
    print("start transfer {}".format(split))
    pattern=os.path.join(from_dir,"*.npy")
    h5file = '{}/{}_img_feature.h5'.format(to_dir, split)
    bar = progressbar.ProgressBar()
    with h5py.File(h5file, 'w') as f:
        for i, filepath in enumerate(bar(glob.glob(pattern)), 1):
            id = os.path.basename(filepath).split('.')[0]
            feature = np.load(filepath)
            f.create_dataset(id, dtype='float32', data=feature)# Save an 3d ndarray (2048,7,7) id(12 string)->3d ndarray (2048,7,7)


if __name__=='__main__':
    main(root)