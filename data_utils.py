import os

import requests
import socket
import socks
import urllib.request as request
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_destrieux_2009
from scipy.stats import skew
from scipy.stats import kurtosis
from utils import nan_or_inf
import os.path as osp
import os


def find(pattern, path):
    import os, fnmatch
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def first_uncommented(stats_file):
    with open(stats_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line[0] != '#':
            return i


def read_fs_stats_file(stats_file, atlas):
    """
    read anatomical properties computed by FreeSurfer,
    for only one hemisphere
    """
    first_line = first_uncommented(stats_file)
    skip_rows = list(range(0, first_line))
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
    if atlas == 'HCPMMP1':
        df = df[1:]  # remove sub-crotical (not for legacy)
    return df


def read_fs_stats(root, atlas):
    """

    :param root:
    :return: res_dict {subject_id: (lh_df, rh_df)}
    """

    if atlas == 'HCPMMP1':
        patten = '*HCPMMP1.stats'
    elif atlas == 'destrieux':
        patten = '*aparc.a2009s.stats'
    stats_files = find(patten, root)
    res_dict = {}
    failed_subjects = []
    for lh_file, rh_file in tqdm(zip(stats_files[0::2], stats_files[1::2]), desc='read_fs_stats',
                                 total=len(stats_files) / 2):
        assert lh_file.split('/')[-3] == rh_file.split('/')[-3]  # same subject
        try:
            subject_id = lh_file.split('/')[-3]
            lh_df = read_fs_stats_file(lh_file, atlas)
            rh_df = read_fs_stats_file(rh_file, atlas)
            if lh_df.empty or rh_df.empty:
                failed_subjects.append(subject_id)
            else:
                res_dict[subject_id] = (lh_df, rh_df)
        except:
            failed_subjects.append(subject_id)
    # assert len(failed_subjects) == 0
    return res_dict


def extract_time_series(nii_file, atlas_nii_file=None):
    atlas_nii_file = fetch_atlas_destrieux_2009().maps if atlas_nii_file is None else atlas_nii_file
    masker = NiftiLabelsMasker(labels_img=atlas_nii_file, standardize=True,
                               memory='nilearn_cache', verbose=5)
    time_series = masker.fit_transform(nii_file)
    return time_series


def resample_temporal(time_series, n_split=2):
    return np.array_split(time_series, n_split)


def top_k_percent_adj(adj, k):
    # remove self-connection
    np.fill_diagonal(adj, 0)
    # sort and get threshold
    sorted_arr = np.sort(adj.flatten())
    idx = k if type(k) == int else int(k * len(sorted_arr))
    threshold = sorted_arr[-idx]
    # threshold
    adj[np.abs(adj) < threshold] = 0
    # add self-connection
    np.fill_diagonal(adj, 1)
    return adj


def repermute(data_list, target_order):
    target_order = np.asarray(target_order)
    new_data_list = []
    for subject_id in target_order:
        for data in data_list:
            if str(data.subject_id) == subject_id:
                new_data_list.append(data)

    return new_data_list


def get_adj_statistics(adj):
    mean = torch.tensor(adj.mean(-1))
    std = torch.tensor(adj.std(-1))
    skewness = torch.tensor(skew(adj, axis=-1))
    kurto = torch.tensor(kurtosis(adj, axis=-1))
    assert not nan_or_inf(kurto) and not nan_or_inf(skewness)
    additional_feature = torch.stack([mean, std, skewness, kurto], dim=-1)
    return additional_feature


def H_operration(arr):
    """
    :type arr: List[int]
    :rtype: int
    """
    if not arr.any():
        return 0
    return max([min(i + 1, c) for i, c in enumerate(sorted(arr, reverse=True))])


def get_hs(adj, level=5):
    """

    :param adj: np.array, shape (num_nodes, num_nodes)
    :return:
    """
    Hs = []
    Hs.append(adj.sum(0))
    for i in range(level):
        H_i = np.asarray([H_operration(Hs[-1][adj[n].nonzero()[0]]) for n in range(adj.shape[0])])
        Hs.append(H_i)
    return torch.tensor(np.asarray(Hs)).t()


def fetch_url(proxies, out_dir, s3_prefix, s3_path):
    rel_path = s3_path.lstrip(s3_prefix)
    download_file = os.path.join(out_dir, rel_path)
    download_dir = os.path.dirname(download_file)
    if not os.path.exists(download_dir):
        try:
            os.makedirs(download_dir)
        except:  # racing
            pass
    if not os.path.exists(download_file):
        print('Retrieving: {0}'.format(download_file))
        try:
            r = requests.get(s3_path, stream=True, proxies=proxies)
            if r.status_code == 200:
                with open(download_file, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
        except:  # time out
            pass
    return download_file


def download_abide(out_dir, site=None, subject_ids_to_download=None,
                   proxies=None):
    # proxies = {'http': "socks5://192.168.192.128:1080",
    #            'https': "socks5://192.168.192.128:1080"}
    # Init variables
    mean_fd_thresh = 0.2
    # mean_fd_thresh = 1e5
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
    s3_pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])
    fs_file_list_path = 'fs_file_list.txt'

    if not os.path.exists(out_dir):
        print('Could not find {0}, creating now...'.format(out_dir))
        os.makedirs(out_dir)

    # Load the phenotype file from S3
    # s3_pheno_file = request.urlopen(s3_pheno_path)
    # pheno_list = s3_pheno_file.readlines()
    s3_pheno_file = requests.get(s3_pheno_path, allow_redirects=True, proxies=proxies)
    pheno_list = s3_pheno_file.text.splitlines()

    # Load FreeSurfer file list
    with open(fs_file_list_path, 'r') as f:
        content = f.readlines()
    fs_file_list = [x.strip() for x in content]

    # Get header indices
    header = pheno_list[0].split(',')
    try:
        site_idx = header.index('SITE_ID')
        file_idx = header.index('FILE_ID')
        dx_idx = header.index('DX_GROUP')
        mean_fd_idx = header.index('func_mean_fd')
    except Exception as exc:
        err_msg = 'Unable to extract header information from the pheno file: {0}\nHeader should have pheno info:' \
                  ' {1}\nError: {2}'.format(s3_pheno_path, str(header), exc)
        raise Exception(err_msg)

    # Go through pheno file and build download paths
    print('Collecting images of interest...')
    s3_paths = []
    for pheno_row in pheno_list[1:]:

        # Comma separate the row
        cs_row = pheno_row.split(',')

        try:
            # See if it was preprocessed
            row_file_id = cs_row[file_idx]
            subject_id = row_file_id[-5:]
            # Read in participant info
            row_site = cs_row[site_idx]
            row_dx = cs_row[dx_idx]
            row_mean_fd = float(cs_row[mean_fd_idx])
        except Exception as e:
            err_msg = 'Error extracting info from phenotypic file, skipping...'
            print(err_msg)
            continue

        if subject_ids_to_download is not None:
            if subject_id not in subject_ids_to_download:
                continue

        # If the filename isn't specified, skip
        if row_file_id == 'no_filename':
            continue
        # If mean fd is too large, skip
        # if row_mean_fd >= mean_fd_thresh:
        #     continue

        # Test phenotypic criteria (three if's looks cleaner than one long if)
        # Test site
        if site != 'ALL' and site.lower() != row_site.lower():
            continue

        # add functional to download
        filename = row_file_id + '_' + 'func_preproc' + '.nii.gz'
        s3_path = '/'.join([s3_prefix, 'Outputs', 'ccs', 'filt_noglobal', 'func_preproc', filename])
        # print('Adding {0} to download queue...'.format(s3_path))
        s3_paths.append(s3_path)

        # add structural to download
        for fs_file in fs_file_list:
            s3_path = '/'.join([s3_prefix, 'Outputs', 'freesurfer/5.1', row_file_id, fs_file])
            # print('Adding {0} to download queue...'.format(s3_path))
            s3_paths.append(s3_path)

    # And download the items
    from multiprocessing.pool import ThreadPool, Pool
    from functools import partial
    download_func = partial(fetch_url, proxies, out_dir, s3_prefix)
    with Pool(os.cpu_count()) as pool:
        results = list(pool.imap_unordered(download_func, s3_paths, chunksize=100))

    # Print all done
    print('Done!')


def label_from_pheno(pheno_df, subject):
    y = pheno_df[pheno_df['FILE_ID'] == subject]['DX_GROUP'].values.item() - 1
    sex = pheno_df[pheno_df['FILE_ID'] == subject]['SEX'].values.item() - 1

    iq = pheno_df[pheno_df['FILE_ID'] == subject]['FIQ'].values.item()
    iq = np.nan if iq == -9999 or np.isnan(iq) else iq

    site_id_string = pheno_df[pheno_df['FILE_ID'] == subject]['SITE_ID'].values.item()
    site_id = np.where(pheno_df.SITE_ID.unique() == site_id_string)[0].item()

    subject_id = pheno_df[pheno_df['FILE_ID'] == subject]['subject'].values.item()
    return y, sex, iq, site_id, subject_id


def chunks(l, n):
    """Yield successive n chunks from l."""
    size = int((len(l) + 1) / n) + 1
    for i in range(0, len(l), size):
        yield l[i:(i + size) if i + size < len(l) else len(l)]


def process_fs_output(fs_subject_dir, sh_script_path):
    only_dirs = [f for f in os.listdir(fs_subject_dir) if osp.isdir(osp.join(fs_subject_dir, f))]
    subject_ids = [osp.basename(p) for p in only_dirs]
    subject_ids_file_path = osp.join(fs_subject_dir, 'all_subject_list')
    with open(subject_ids_file_path, 'w') as f:
        for item in subject_ids:
            if item[-1].isdigit() and item[0].isalpha():
                f.write("%s\n" % item)

    cp_cmd = 'cd {} && cp -r $SUBJECTS_DIR/fsaverage ./'.format(fs_subject_dir)
    # os.system(cp_cmd)
    # os.environ["SUBJECT_DIR"] = fs_subject_dir
    # # run the script in parallel
    cmd = 'cd {} &&'.format(fs_subject_dir) + \
          'bash ' + sh_script_path + \
          ' -L {} -a HCPMMP1 -d all_output'.format(subject_ids_file_path)  # `all_output` hardcoded
    # os.system(cmd)  # it's dangerous to run parallel bash script in this way, require manual launching the script

    if osp.isdir(osp.join(fs_subject_dir, 'all_output')):
        return
    else:
        raise Exception("\n\nPlease run processing manually from here\n" + cp_cmd + '\n' + cmd + '\n')


if __name__ == '__main__':
    download_abide('datasets/ALL')
    # process_fs_output('/data_57/huze/projects/IMEGAT/datasets/ALL/raw/Outputs/freesurfer/5.1', None)
