def find(pattern, path):
    import os, fnmatch
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def read_fs_stats_file(stats_file):
    import pandas as pd
    """
    read anatomical properties computed by FreeSurfer,
    for only one hemisphere
    """
    skip_rows = list(range(0, 59)) # 53 for legacy
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


def extract_time_series(nii_file, atlas_nii_file=None):
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.datasets import fetch_atlas_destrieux_2009
    atlas_nii_file = fetch_atlas_destrieux_2009().maps if atlas_nii_file is None else atlas_nii_file
    masker = NiftiLabelsMasker(labels_img=atlas_nii_file, standardize=True,
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


def download_freesurfer_output(site, out_dir):
    # Import packages
    import os
    import urllib.request as request
    # Init variables
    mean_fd_thresh = 0.2
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/' \
                'ABIDE_Initiative'
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

        # functional
        filename = row_file_id + '_' + 'func_preproc' + '.nii.gz'
        s3_path = '/'.join([s3_prefix, 'Outputs', 'ccs', 'filt_noglobal', 'func_preproc', filename])
        print('Adding {0} to download queue...'.format(s3_path))
        s3_paths.append(s3_path)

        for fs_file in fs_file_list:
            s3_path = '/'.join([s3_prefix, 'Outputs', 'freesurfer/5.1', row_file_id, fs_file])
            print('Adding {0} to download queue...'.format(s3_path))
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

    # save subject list chunks to file
    all_file_names = []
    for i, chunk in enumerate(subject_chunks):
        file_basename = 'subject_list_{}'.format(i)
        file_path = osp.join(fs_subject_dir, file_basename)
        with open(file_path, 'w') as f:
            for item in chunk:
                f.write("%s\n" % item)
        all_file_names.append(file_basename)
    with open(osp.join(fs_subject_dir, 'all_subject_list'), 'w') as f:
        for item in all_file_names:
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
    # read_fs_stats('datasets/NYU')
    download_freesurfer_output('NYU', 'datasets/NYU')
    # process_fs_output('')
