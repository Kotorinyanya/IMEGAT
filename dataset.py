import torch_geometric

from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import torch
import os.path as osp
from os.path import join
from tqdm import tqdm
import os
import urllib.request as request

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
        super(ABIDE, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Outputs/']

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
        pass


if __name__ == '__main__':
    abide = ABIDE(root='datasets/NYU')