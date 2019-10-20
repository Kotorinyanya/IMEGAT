import torch_geometric
from nilearn.connectome import ConnectivityMeasure
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import torch
import os.path as osp
from os.path import join

from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm
import os
import urllib.request as request

from data_utils import read_fs_stats, extract_time_series, z_score_norm_data


class ABIDE(InMemoryDataset):
    def __init__(self, root, name='ABIDE', transform=None, pre_transform=None, site='NYU',
                 derivative='func_preproc', pipeline='ccs', strategy='filt_noglobal', extension='.nii.gz',
                 mean_fd_thresh=0.2):
        self.mean_fd_thresh = mean_fd_thresh
        self.extension = extension
        self.strategy = strategy
        self.pipeline = pipeline
        self.derivative = derivative
        self.name = name + '_' + site
        self.site = site
        self.anatomical_feature_names = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg',
                                         'ThickStd', 'MeanCurv', 'GausCurv']
        super(ABIDE, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Outputs/', 'Phenotypic_V1_0b_preprocessed1.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):

        # Init variables
        s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/' \
                    'ABIDE_Initiative'
        s3_pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])
        out_dir = '/'.join([self.root, 'raw'])

        if not os.path.exists(out_dir):
            print('Could not find {0}, creating now...'.format(out_dir))
            os.makedirs(out_dir)

        # Load the phenotype file from S3
        s3_pheno_file = request.urlopen(s3_pheno_path)
        phenot_file = osp.join(out_dir, self.raw_file_names[1])
        with open(phenot_file) as f:
            f.write(s3_pheno_file.read())
        pheno_list = s3_pheno_file.readlines()
        print(pheno_list[0])
        # Get header indices
        header = pheno_list[0].decode().split(',')
        try:
            site_idx = header.index('SITE_ID')
            file_idx = header.index('FILE_ID')
            age_idx = header.index('AGE_AT_SCAN')
            sex_idx = header.index('SEX')
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
                row_age = float(cs_row[age_idx])
                row_sex = cs_row[sex_idx]
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
            if row_mean_fd >= self.mean_fd_thresh:
                continue

            # Test phenotypic criteria (three if's looks cleaner than one long if)
            # Test site
            if self.site is not None and self.site.lower() != row_site.lower():
                continue

            # functional
            filename = row_file_id + '_' + self.derivative + self.extension
            s3_path = '/'.join([s3_prefix, 'Outputs', self.pipeline, self.strategy, self.derivative, filename])
            print('Adding {0} to download queue...'.format(s3_path))
            s3_paths.append(s3_path)

            # structural
            filename = row_file_id
            # for freesurfer Destrieux Atlas
            sub_files = ['lh.aparc.a2009s.stats', 'rh.aparc.a2009s.stats']
            for file in sub_files:
                s3_path = '/'.join([s3_prefix, 'Outputs', 'freesurfer', '5.1', filename, 'stats', file])
                print('Adding {0} to download queue...'.format(s3_path))
                s3_paths.append(s3_path)

        # download the items
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

    def process(self):
        s3_pheno_path = '/'.join([self.root, 'raw', self.raw_file_names[1]])
        pheno_df = pd.read_csv(s3_pheno_path)

        correlation_measure = ConnectivityMeasure(kind='correlation')

        anatomical_features_dict = read_fs_stats(self.root)
        subject_ids = list(anatomical_features_dict.keys())

        data_list = []
        for subject in tqdm(subject_ids):
            # phenotypic (for label only)
            y = torch.from_numpy(
                pheno_df[pheno_df['FILE_ID'] == subject]['DX_GROUP'].values)
            # structural
            lh_df, rh_df = anatomical_features_dict[subject]
            node_features = torch.from_numpy(
                np.concatenate([lh_df[self.anatomical_feature_names].values,
                                rh_df[self.anatomical_feature_names].values]))
            # functional
            fmri_nii_file = '/'.join([self.root, 'raw', 'Outputs', self.pipeline, self.strategy, self.derivative,
                                      "{}_func_preproc.nii.gz".format(subject)])

            time_series = extract_time_series(fmri_nii_file)
            time_series_list = self.pre_transform(time_series) if self.pre_transform else [time_series]
            connectivity_matrix_list = correlation_measure.fit_transform(time_series_list)

            for conn in connectivity_matrix_list:
                edge_index, edge_weight = from_scipy_sparse_matrix(coo_matrix(conn))
                data = Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_weight,
                            y=y)
                data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


if __name__ == '__main__':
    abide = ABIDE(root='datasets/NYU', transform=z_score_norm_data)
    pass
