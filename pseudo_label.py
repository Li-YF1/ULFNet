import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from DINO import DINOExtractor
from dataloader import test_dataset2
from scipy.linalg import eigh
from  scipy import ndimage
from segment_anything import sam_model_registry, SamPredictor

image_size=256
model = DINOExtractor()
model.cuda()
model.eval()

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0, 1) @ feats).cpu().numpy()
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], \
        bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def detect_box(bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]

    if principle_object:
        mask = np.where(objects == cc)
        # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError

def Ncut(feats):

    tau=0.2
    dims=[32,32]
    scales=[8,8]
    init_image_size=[image_size,image_size]

    # construct the affinity matrix
    A, D = get_affinity_matrix(feats[0], tau)
    # get the second smallest eigenvector
    eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
    # get salient area
    bipartition = get_salient_areas(second_smallest_vec)
    seed = np.argmax(np.abs(second_smallest_vec))
    nc = check_num_fg_corners(bipartition, dims)
    if nc >= 3:
        reverse = True
    else:
        reverse = bipartition[seed] != 1
    if reverse:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
        seed = np.argmax(eigenvec)
    else:
        seed = np.argmax(second_smallest_vec)

    # get pxiels corresponding to the seed
    bipartition = bipartition.reshape(dims).astype(float)
    _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
    pseudo_mask = np.zeros(dims)
    pseudo_mask[cc[0], cc[1]] = 1
    pseudo_mask = torch.from_numpy(pseudo_mask)
    pseudo_mask=pseudo_mask.cpu().numpy()
    pseudo_mask=cv2.resize(pseudo_mask,(256,256))
    return pseudo_mask


def Prompt_SAM(name,prompt):
    sam_path = 'sam_vit_h_4b8939.pth'
    net_sam = sam_model_registry['vit_h'](checkpoint=sam_path)
    net_sam.cuda()
    net_sam.eval()
    predictor = SamPredictor(net_sam)
    img = cv2.imread( name)
    predictor.set_image(img)
    masks, score, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=prompt[None, :]
    )
    index = np.argsort(score)
    score = score[index]
    masks = masks[index]
    return mask[0],score[0]


test_datasets=['']
for dataset in test_datasets:
    save_path =''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root=''
    depth_root = ''

    test_loader = test_dataset2(image_root,  image_root,depth_root, image_size, dataset)
    for i in range(test_loader.size):
        image,  depth, name, image_for_post = test_loader.load_data(dataset)
        image = image.cuda()
        depth = depth.cuda()
        res= model.forward_step(image)
        res = Ncut(res)

        rows, cols = np.nonzero(res)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        prompt = np.array([min_col, min_row, max_col, max_row])
        mask,score = Prompt_SAM(os.path.join(image_root, name), prompt)
        root=''
        if not os.path.exists(root):
            os.makedirs(root)
        path=os.path.join(root, name)
        cv2.imwrite(path,mask)
    print('Test Done!')
