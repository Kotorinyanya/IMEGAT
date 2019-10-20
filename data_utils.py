import os.path as osp
import pandas as pd
import os, fnmatch

from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_destrieux_2009

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def read_fs_stats_file(stats_file):
    """
    read anatomical properties computed by FreeSurfer,
    for only one hemisphere
    """
    skip_rows = list(range(0, 53))
    names = ['StructName',
             'NumVert',
             'SurfArea',
             'GrayVol',
             'ThickAvg',
             'ThickStd',
             'MeanCurv',
             'GausCurv',
             'FoldInd',
             'CurvInd']
    df = pd.read_csv(stats_file, sep=' +', skiprows=skip_rows, names=names)
    return df


def read_fs_stats(root):
    """

    :param root:
    :return: res_dict {subject_id: (lh_df, rh_df)}
    """
    stats_files = find('*.stats', root)
    res_dict = {}
    for lh_file, rh_file in zip(stats_files[0::2], stats_files[1::2]):
        subject_id = lh_file.split('/')[-3]
        lh_df = read_fs_stats_file(lh_file)
        rh_df = read_fs_stats_file(rh_file)
        res_dict[subject_id] = (lh_df, rh_df)
    return res_dict

def extract_time_series(nii_file):
    destrieux_2009 = fetch_atlas_destrieux_2009()
    masker = NiftiLabelsMasker(labels_img=destrieux_2009.maps, standardize=True,
                               memory='nilearn_cache', verbose=5)
    time_series = masker.fit_transform(nii_file)
    return time_series

def resample_temporal(time_series, time_window):
    time_series_list = []
    length = time_series.shape[0]
    for start in range(0, length, time_window):
        end = start + time_window
        end = length - 1 if end >= length else end
        time_series_list.append(time_series[start:end])
    return time_series_list

def z_score_norm(tensor):
    """
    Normalize a tensor with mean and standard deviation.
    Args:
        tensor (Tensor): Tensor image of size [num_nodes, num_node_features] to be normalized.

    Returns:
        Tensor: Normalized tensor.
    """
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std == 0] = 1e-6
    normed_tensor = (tensor - mean) / std
    return normed_tensor

def z_score_norm_data(data):
    data.x = z_score_norm(data.x)
    return data


if __name__ == '__main__':
    read_fs_stats('datasets/NYU')