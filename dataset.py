import logging

import nilearn
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

from utils import z_score_norm_data, positive_transform, nan_or_inf, check_strongly_connected, fisher_z, drop_negative

from data_utils import read_fs_stats, extract_time_series, download_abide, \
    process_fs_output, resample_temporal, top_k_percent_adj, label_from_pheno, repermute

from nilearn.plotting import plot_matrix
from sir import sir_score_of_adj


class ABIDE(InMemoryDataset):
    def __init__(self, root,
                 name='ABIDE',
                 transform=None,
                 resample_ts=False,
                 transform_edge=None,
                 additional_node_feature_func=None,
                 threshold=None,
                 atlas='HCPMMP1',
                 site='ALL',
                 derivative='func_preproc', pipeline='ccs', strategy='filt_noglobal',
                 extension='.nii.gz',
                 mean_fd_thresh=0.2):
        """

        :param root: data directory
        :param name: ABIDE
        :param transform: transform at run
        :param resample_ts: bool, for data augmentation
        :param transform_edge: function
        :param additional_node_feature_func: function
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
        self.additional_node_feature_func = additional_node_feature_func
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
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Outputs/',
                'Phenotypic_V1_0b_preprocessed1.csv',
                'lh.HCPMMP1.annot',
                'rh.HCPMMP1.annot',
                'HCPMMP1_on_MNI152_ICBM2009a_nlin.nii.gz']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):

        out_dir = '/'.join([self.root, 'raw'])
        download_abide(out_dir, self.site)

        # download .annot
        lh_url = 'https://ndownloader.figshare.com/files/5528816'
        rh_url = 'https://ndownloader.figshare.com/files/5528819'
        os.system('wget {} -p {}'.format(lh_url, out_dir))
        os.system('wget {} -p {}'.format(rh_url, out_dir))

        # download HCPMMP1.nii.gz
        url = 'https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/5594363/HCPMMP1_on_MNI152_ICBM2009a_nlin.nii.gz'
        os.system('wget {} -p {}'.format(url, out_dir))

        # Load the phenotype file from S3
        s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
        s3_pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])
        s3_pheno_file = request.urlopen(s3_pheno_path)
        phenot_file = osp.join(out_dir, self.raw_file_names[1])
        # download phenotype.csv
        with open(phenot_file, 'wb') as f:
            f.write(s3_pheno_file.read())

    def process(self):
        raw_dir = '/'.join([self.root, 'raw'])
        fs_subject_dir = osp.join(raw_dir, self.raw_file_names[0], 'freesurfer/5.1')

        # copy .annot to SUBJECT_DIR
        os.system('cp {} {}'.format(osp.join(raw_dir, self.raw_file_names[2]), fs_subject_dir))
        os.system('cp {} {}'.format(osp.join(raw_dir, self.raw_file_names[3]), fs_subject_dir))

        # process fs outputs
        shell_script_path = osp.join(os.getcwd(), 'fs_preproc.sh')
        process_fs_output(fs_subject_dir, shell_script_path)

        if self.atlas == 'HCPMMP1':
            fs_subject_dir = osp.join(fs_subject_dir, 'all_output')  # `all_output` hardcoded

        # phenotypic for label
        s3_pheno_path = '/'.join([self.root, 'raw', self.raw_file_names[1]])
        pheno_df = pd.read_csv(s3_pheno_path)

        # load atlas for fmri
        if self.atlas == 'HCPMMP1':
            # load and transform atlas
            atlas_nii_file = '/'.join([self.root, 'raw', self.raw_file_names[4]])
            atlas_img = image.load_img(atlas_nii_file)
            # split left and right hemisphere
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

        # make subject_ids
        subject_ids = list(anatomical_features_dict.keys())
        if self.site == 'ALL':
            import urllib
            all_subject_ids_path = 'https://raw.githubusercontent.com/parisots/population-gcn/master/subject_IDs.txt'
            response = urllib.request.urlopen(all_subject_ids_path)
            all_subject_ids = [s.decode() for s in response.read().splitlines()]
            subject_ids = [s for s in subject_ids if s[-5:] in all_subject_ids]
            assert len(subject_ids) == 871

        # process the data
        data_list = []
        failed_subject_list = []
        for subject in tqdm(subject_ids, desc='subject_list'):
            try:
                y, sex, iq, site_id, subject_id = label_from_pheno(pheno_df, subject)

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

                # handle broken file in ABIDE preprocessed
                if subject == 'UM_1_0050302':
                    time_series = nilearn.signal.clean(time_series.transpose(), low_pass=0.1, high_pass=0.01, t_r=2
                                                       ).transpose()
                if subject == 'Leuven_2_0050730':
                    time_series = nilearn.signal.clean(time_series.transpose(), low_pass=0.1, high_pass=0.01, t_r=1.6667
                                                       ).transpose()
                # optional data augmentation
                time_series_list = resample_temporal(time_series) if self.resample_ts else [time_series]
                # correlation form time series
                connectivity_matrix_list = correlation_measure.fit_transform(time_series_list)
                for adj in connectivity_matrix_list:
                    if self.additional_node_feature_func is not None:
                        additional_feature = self.additional_node_feature_func(adj)
                        node_features = torch.cat([node_features, additional_feature], dim=-1)
                    # transform adj
                    # np.fill_diagonal(adj, 0)  # remove self-loop for transform
                    adj = self.transform_edge(adj) if self.transform_edge is not None else adj
                    # set a threshold for adj
                    if self.threshold is not None:
                        adj = top_k_percent_adj(adj, self.threshold)
                        assert check_strongly_connected(adj) == True
                    # create torch_geometric Data
                    edge_index, edge_weight = from_scipy_sparse_matrix(coo_matrix(adj))

                    data = Data(x=node_features,
                                edge_index=edge_index,
                                edge_attr=edge_weight,
                                y=y)
                    data.sex, data.iq, data.site_id, data.subject_id = sex, iq, site_id, subject_id
                    data.num_nodes = data.x.shape[0]
                    data_list.append(data)
            except Exception as e:
                print(e)
                logging.warning("failed at subject {}".format(subject))
                failed_subject_list.append(subject)
        # with open('failed_fmri_subject_list', 'w') as f:
        #     f.write("\n".join(failed_subject_list))
        print("failed_subject_list", failed_subject_list)
        if self.site == 'ALL':
            data_list = repermute(data_list, all_subject_ids)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])


if __name__ == '__main__':
    abide = ABIDE(root='datasets/ALL',
                  resample_ts=False,
                  transform_edge=None,
                  # additional_node_feature_func=sir_score_of_adj,
                  threshold=None,
                  site='ALL',
                  atlas='HCPMMP1')
    pass
