import os
import numpy as np
import nibabel as nib
from surfer import Brain
import torch


def get_brain(hemi, subject_id="fsaverage"):
    surf = "inflated"
    view = ['lateral', 'medial']
    # view = ['lateral']
    brain = Brain(subject_id, hemi, surf, cortex='bone',
                  background="white", alpha=1, views=view)

    return brain


def get_labels_and_ctab(S, annot_path):
    """
    params S: shape (n, 5)
    """
    color_reduce = np.asarray([[255., 0., 0.],
                               [0., 255., 0.],
                               [0., 0., 255.],
                               [255., 255., 255.],
                               [0., 0., 0.]])
    S = S @ color_reduce

    labels, ctab, names = nib.freesurfer.read_annot(annot_path)

    # index for color
    #     ctab[:, 4] = np.arange(len(ctab))
    ctab[1:, :3] = S  # ingonore 0
    print(ctab)

    nib.freesurfer.write_annot('tmp.annot', labels, ctab, names)
    return labels, ctab


def plot_surf(S, hemi='lh', subject_id="fsaverage", annot_name='.HCPMMP1.annot'):
    """
    params S: shape (n, 5), single hemi
    """
    annot_path = os.path.join(os.environ["SUBJECTS_DIR"],
                              subject_id, "label",
                              hemi + annot_name)

    labels, ctab = get_labels_and_ctab(S, annot_path)

    brain = get_brain(hemi=hemi)

    #     brain.add_annotation((labels, ctab), borders=False,)
    brain.add_annotation('tmp.annot', borders=False, )
    #     brain.add_annotation((labels, ctab), borders=1, remove_existing=False)
    brain.toggle_toolbars()

    return brain


S = torch.load('tensor').cpu().numpy()
lh_brain = plot_surf(S[4, :180, :], hemi='lh')
rh_brain = plot_surf(S[4, 180:, :], hemi='rh')

while True:
    pass