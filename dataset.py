import torch_geometric
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import torch
import os.path as osp
from os.path import join

from nilearn import image

from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm
import os
import urllib.request as request

from scipy.stats import kurtosis, skew

from utils import z_score_norm_data, positive_transform, nan_or_inf

from data_utils import read_fs_stats, extract_time_series, download_abide, \
    process_fs_output, resample_temporal, top_k_percent_adj, label_from_pheno

from nilearn.plotting import plot_matrix


class ABIDE(InMemoryDataset):
    def __init__(self, root,
                 name='ABIDE',
                 transform=None,
                 resample_ts=False,
                 transform_edge=False,
                 use_edge_weight_as_node_feature=True,
                 threshold=None,
                 atlas='HCPMMP1',
                 site='NYU',
                 derivative='func_preproc', pipeline='ccs', strategy='filt_noglobal',
                 extension='.nii.gz',
                 mean_fd_thresh=0.2):
        """

        :param root: data directory
        :param name: ABIDE
        :param transform: transform at run
        :param resample_ts: bool, for data augmentation
        :param transform_edge: bool, positive transform of edges
        :param use_edge_weight_as_node_feature: bool
        :param threshold: float or int
        :param atlas: str
        :param site: str
        :param derivative: str
        :param pipeline: str
        :param strategy: str
        :param extension: str
        :param mean_fd_thresh: float
        """
        self.threshold = threshold
        self.use_edge_weight_as_node_feature = use_edge_weight_as_node_feature
        self.atlas = atlas
        self.transform_edge = transform_edge
        self.resample_ts = resample_ts
        self.mean_fd_thresh = mean_fd_thresh
        self.extension = extension
        self.strategy = strategy
        self.pipeline = pipeline
        self.derivative = derivative
        self.name = name + '_' + site
        self.site = site
        self.anatomical_feature_names = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg',
                                         'ThickStd', 'MeanCurv', 'GausCurv']
        super(ABIDE, self).__init__(root, transform)
        self.data, self.slices, self.group_vector, self.site_vector = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Outputs/', 'Phenotypic_V1_0b_preprocessed1.csv', 'lh.HCPMMP1.annot', 'rh.HCPMMP1.annot',
                'HCPMMP1_on_MNI152_ICBM2009a_nlin.nii.gz']

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

        # download entire freesurfer ouput
        download_abide(site=self.site, out_dir=out_dir)

        # download .annot
        lh_url = 'https://ndownloader.figshare.com/files/5528816'
        rh_url = 'https://ndownloader.figshare.com/files/5528819'
        os.system('wget {} -p {}'.format(lh_url, out_dir))
        os.system('wget {} -p {}'.format(rh_url, out_dir))

        # download HCPMMP1.nii.gz
        url = 'https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5594363/HCPMMP1_on_MNI152_ICBM2009a_nlin.nii.gz'
        os.system('wget {} -p {}'.format(url, out_dir))

        # Load the phenotype file from S3
        s3_pheno_file = request.urlopen(s3_pheno_path)
        phenot_file = osp.join(out_dir, self.raw_file_names[1])
        # download csv
        with open(phenot_file, 'wb') as f:
            f.write(s3_pheno_file.read())

        # download images
        s3_pheno_file = request.urlopen(s3_pheno_path)
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

            # # structural
            # filename = row_file_id
            # # for freesurfer Destrieux Atlas
            # sub_files = ['lh.aparc.a2009s.stats', 'rh.aparc.a2009s.stats']
            # for file in sub_files:
            #     s3_path = '/'.join([s3_prefix, 'Outputs', 'freesurfer', '5.1', filename, 'stats', file])
            #     print('Adding {0} to download queue...'.format(s3_path))
            #     s3_paths.append(s3_path)

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
        raw_dir = '/'.join([self.root, 'raw'])

        # copy .annot to SUBJECT_DIR
        fs_subject_dir = osp.join(raw_dir, self.raw_file_names[0], 'freesurfer/5.1')
        os.system('cp {} {}'.format(osp.join(raw_dir, self.raw_file_names[2]), fs_subject_dir))
        os.system('cp {} {}'.format(osp.join(raw_dir, self.raw_file_names[3]), fs_subject_dir))

        # process fs outputs
        shell_script_path = osp.join(os.getcwd(), 'create_subj_volume_parcellation.sh')
        process_fs_output(fs_subject_dir, shell_script_path)

        # this is dirty, but who cares?
        if self.atlas == 'HCPMMP1':
            fs_subject_dir = osp.join(fs_subject_dir, 'all_output')

        # phenotypic for label
        s3_pheno_path = '/'.join([self.root, 'raw', self.raw_file_names[1]])
        pheno_df = pd.read_csv(s3_pheno_path)

        # for optional atlas
        if self.atlas == 'HCPMMP1':
            # load and transform atlas
            atlas_nii_file = '/'.join([self.root, 'raw', self.raw_file_names[4]])
            atlas_img = image.load_img(atlas_nii_file)
            atlas_img.get_data()[atlas_img.get_data()[:int(atlas_img.shape[0] / 2 + 1), :, :].nonzero()] += \
                atlas_img.get_data().max()
            num_nodes = 360
        elif self.atlas == 'destrieux':
            from nilearn.datasets import fetch_atlas_destrieux_2009
            atlas_nii_file = fetch_atlas_destrieux_2009().maps
            atlas_img = image.load_img(atlas_nii_file)
            num_nodes = 148

        # read FreeSurfer output
        anatomical_features_dict = read_fs_stats(fs_subject_dir, self.atlas)
        subject_ids = list(anatomical_features_dict.keys())

        data_list = []
        group_vector, current_group = [], 0
        site_vector = []
        failed_subject_list = []
        for subject in tqdm(subject_ids, desc='subject_list'):
            try:
                y = label_from_pheno(pheno_df, subject)

                # read anatomical features from dict
                lh_df, rh_df = anatomical_features_dict[subject]
                node_features = torch.from_numpy(
                    np.concatenate([lh_df[self.anatomical_feature_names].values,
                                    rh_df[self.anatomical_feature_names].values])).float()
                if node_features.shape[0] != num_nodes:
                    # check missing nodes, for 'destrieux'
                    continue

                # path for preprocessed functional MRI
                fmri_nii_file = '/'.join([self.root, 'raw', 'Outputs', self.pipeline, self.strategy, self.derivative,
                                          "{}_func_preproc.nii.gz".format(subject)])

                # nilearn masker and corr
                masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True,
                                           memory='nilearn_cache', verbose=5)
                correlation_measure = ConnectivityMeasure(kind='correlation')
                time_series = masker.fit_transform(fmri_nii_file)

                # optional data augmentation
                time_series_list = resample_temporal(time_series) if self.resample_ts else [time_series]
                # group vector for cross validation shuffling
                group_vector += [current_group] * len(time_series_list)
                current_group += 1
                # site vector for multi-site cv
                site_id = subject.split('_')[0]
                site_vector += [site_id] * len(time_series_list)

                # correlation form time series
                connectivity_matrix_list = correlation_measure.fit_transform(time_series_list)

                for adj in connectivity_matrix_list:
                    # concat statistics of adj to node feature
                    if self.use_edge_weight_as_node_feature:
                        mean = torch.tensor(adj.mean(-1))
                        std = torch.tensor(adj.std(-1))
                        skewness = torch.tensor(skew(adj, axis=-1))
                        kurto = torch.tensor(kurtosis(adj, axis=-1))
                        # assert nan_or_inf(kurto)
                        additional_feature = torch.stack([mean, std, skewness, kurto], dim=-1)
                        node_features = torch.cat([node_features, additional_feature], dim=-1)
                    # positive transform (to distance)
                    # adj = 1 - np.sqrt((1 - adj) / 2) if self.transform_edge else adj
                    adj = np.abs(adj)
                    # set a threshold for adj
                    if self.threshold is not None:
                        adj = top_k_percent_adj(adj, self.threshold)

                    # create torch_geometric Data
                    edge_index, edge_weight = from_scipy_sparse_matrix(coo_matrix(adj))
                    data = Data(x=node_features,
                                edge_index=edge_index,
                                edge_attr=edge_weight,
                                y=y)
                    data.num_nodes = data.x.shape[0]
                    data_list.append(data)
            except:
                failed_subject_list.append(subject)
        print("failed_subject_list", failed_subject_list)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, group_vector, site_vector), self.processed_paths[0])


if __name__ == '__main__':
    abide = ABIDE(root='datasets/ALL', transform=z_score_norm_data,
                  resample_ts=False, transform_edge=True,
                  use_edge_weight_as_node_feature=False,
                  threshold=360 * 11,
                  atlas='HCPMMP1')
    pass
