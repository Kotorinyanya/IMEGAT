import os

import requests


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
    import pandas as pd
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
    from tqdm import tqdm

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
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.datasets import fetch_atlas_destrieux_2009
    atlas_nii_file = fetch_atlas_destrieux_2009().maps if atlas_nii_file is None else atlas_nii_file
    masker = NiftiLabelsMasker(labels_img=atlas_nii_file, standardize=True,
                               memory='nilearn_cache', verbose=5)
    time_series = masker.fit_transform(nii_file)
    return time_series


def resample_temporal(time_series, time_window=30):
    time_series_list = []
    length = time_series.shape[0]
    for start in range(0, length, time_window):
        end = start + time_window
        next_end = end + time_window
        if next_end >= length:
            end = length - 1
            time_series_list.append(time_series[start:end])
            break
        time_series_list.append(time_series[start:end])
    return time_series_list


def top_k_percent_adj(adj, k):
    import numpy as np
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


def fetch_url(out_dir, s3_prefix, s3_path):
    rel_path = s3_path.lstrip(s3_prefix)
    download_file = os.path.join(out_dir, rel_path)
    download_dir = os.path.dirname(download_file)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    if not os.path.exists(download_file):
        r = requests.get(s3_path, stream=True)
        if r.status_code == 200:
            with open(download_file, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return download_file


def download_abide(out_dir, site=None):
    # Import packages
    import os
    import urllib.request as request
    # Init variables
    mean_fd_thresh = 0.2
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
    s3_pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])
    fs_file_list_path = 'fs_file_list.txt'

    if not os.path.exists(out_dir):
        print('Could not find {0}, creating now...'.format(out_dir))
        os.makedirs(out_dir)

    # Load the phenotype file from S3
    s3_pheno_file = request.urlopen(s3_pheno_path)
    pheno_list = s3_pheno_file.readlines()

    # Load FreeSurfer file list
    with open(fs_file_list_path, 'r') as f:
        content = f.readlines()
    fs_file_list = [x.strip() for x in content]

    # Get header indices
    header = pheno_list[0].decode().split(',')
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
        cs_row = pheno_row.decode().split(',')

        try:
            # See if it was preprocessed
            row_file_id = cs_row[file_idx]
            # Read in participant info
            row_site = cs_row[site_idx]
            row_dx = cs_row[dx_idx]
            row_mean_fd = float(cs_row[mean_fd_idx])
        except Exception as e:
            err_msg = 'Error extracting info from phenotypic file, skipping...'
            print(err_msg)
            continue

        # If the filename isn't specified, skip
        if row_file_id == 'no_filename':
            continue
        # If mean fd is too large, skip
        if row_mean_fd >= mean_fd_thresh:
            continue

        # Test phenotypic criteria (three if's looks cleaner than one long if)
        # Test site
        if site is not None and site.lower() != row_site.lower():
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
    total_num_files = len(s3_paths)
    for path_idx, s3_path in enumerate(s3_paths):
        rel_path = s3_path.lstrip(s3_prefix)
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        try:
            if not os.path.exists(download_file):
                print('Retrieving: {0}'.format(download_file))
                request.urlretrieve(s3_path, download_file)
                print('{0:3f}% percent complete'.format(100 * (float(path_idx + 1) / total_num_files)))
            else:
                print('File {0} already exists, skipping...'.format(download_file))
        except Exception as exc:
            print('There was a problem downloading {0}.\n Check input arguments and try again.'.format(s3_path))

    # Print all done
    print('Done!')


def label_from_pheno(pheno_df, subject):
    import torch
    y = torch.from_numpy(
        pheno_df[pheno_df['FILE_ID'] == subject]['DX_GROUP'].values)
    # for class label
    y[y == 1] = 0
    y[y == 2] = 1
    return y


def chunks(l, n):
    """Yield successive n chunks from l."""
    size = int((len(l) + 1) / n) + 1
    for i in range(0, len(l), size):
        yield l[i:(i + size) if i + size < len(l) else len(l)]


def process_fs_output(fs_subject_dir, sh_script_path):
    import os.path as osp
    import os
    only_dirs = [f for f in os.listdir(fs_subject_dir) if osp.isdir(osp.join(fs_subject_dir, f))]
    subject_ids = [osp.basename(p) for p in only_dirs]
    num_workers = os.cpu_count()
    subject_chunks = chunks(subject_ids, num_workers)

    # # save subject list chunks to file
    # all_file_names = []
    # for i, chunk in enumerate(subject_chunks):
    #     file_basename = 'subject_list_{}'.format(i)
    #     file_path = osp.join(fs_subject_dir, file_basename)
    #     with open(file_path, 'w') as f:
    #         for item in chunk:
    #             f.write("%s\n" % item)
    #     all_file_names.append(file_basename)
    # with open(osp.join(fs_subject_dir, 'all_subject_list'), 'w') as f:
    #     for item in all_file_names:
    #         f.write("%s\n" % item)
    with open(osp.join(fs_subject_dir, 'all_subject_list'), 'w') as f:
        for item in subject_ids:
            if item[-1].isdigit() and item[0].isalpha():
                f.write("%s\n" % item)

    cp_cmd = 'cd {} && cp -r $SUBJECTS_DIR/fsaverage ./'.format(fs_subject_dir)
    # os.system(cp_cmd)
    # os.environ["SUBJECT_DIR"] = fs_subject_dir
    # # run the script in parallel
    cmd = 'cd {} &&'.format(fs_subject_dir) + \
          'cat all_subject_list | parallel bash ' + sh_script_path + \
          ' -L {} -a HCPMMP1 -d {}_output -t YES "&>" {}.log'
    # os.system(cmd)
    merge_cmd = 'cd {};'.format(fs_subject_dir) + \
                'mkdir all_output;' + \
                'find -name "subject_list_*_output" -type d -print0 | xargs -0 -n 1 -I {} mv "{}" "all_output/{}";' + \
                "find -name 'NYU*' -type d -exec sh -c 'mv {} ./$(basename {})' \;" + \
                "find -name 'subject_list_*' -type d -exec sh -c 'rm -rf {}' \;"

    if osp.isdir(osp.join(fs_subject_dir, 'all_output')):
        return
    else:
        raise Exception("\n\nPlease run processing manually from here\n" + cp_cmd + '\n' + cmd + '\n' + merge_cmd)


if __name__ == '__main__':
    # download_abide('new_datasets/ALL')
    process_fs_output('/data_57/huze/projects/IMEGAT/datasets/ALL/raw/Outputs/freesurfer/5.1', None)
