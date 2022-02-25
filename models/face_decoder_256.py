from models.bfm import BfmExtend
import torch
from torch import nn
from utils.simple_renderer import SimpleRenderer


class Face3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.facemodel = BfmExtend()
        self.keypoints = self.facemodel.keypoints

    def Split_coeff(self, coeff):
        shp_coeff = coeff[:, :144]  # id_coeff :80, ex_coeff 80:144
        tex_coeff = coeff[:, 144:224]
        angles = coeff[:, 224:227]
        gamma = coeff[:, 227:254]
        translation = coeff[:, 254:257]

        return shp_coeff, tex_coeff, angles, translation, gamma

    def Compute_rotation_matrix(self, angles):
        N = angles.shape[0]
        device = angles.device
        x = angles[:, 0]
        y = angles[:, 1]
        z = angles[:, 2]
        cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
        sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
        rotation = torch.zeros(N, 3, 3).to(device)
        rotation[:, 0, 0] = cz * cy
        rotation[:, 0, 1] = sx * sy * cz - cx * sz
        rotation[:, 0, 2] = cx * sy * cz + sx * sz
        rotation[:, 1, 0] = cy * sz
        rotation[:, 1, 1] = sx * sy * sz + cx * cz
        rotation[:, 1, 2] = cx * sy * sz - sx * cz
        rotation[:, 2, 0] = -sy
        rotation[:, 2, 1] = sx * cy
        rotation[:, 2, 2] = cx * cy
        rotation = torch.transpose(rotation, 1, 2)
        return rotation

    def Rigid_transform_block(self, face_shape, rotation, translation):
        face_shape_r = torch.bmm(face_shape, rotation)
        face_shape_t = face_shape_r + translation.unsqueeze(1)
        return face_shape_t

    def Orthogonal_projection_block(self, face_shape, focal=1015.0):
        # the reconstructed coordinates are from -112 to 112
        div = torch.ones_like(face_shape)  # *10
        div[:, :, 0] = 10 - face_shape[:, :, 2]
        div[:, :, 1] = 10 - face_shape[:, :, 2]
        div[:, :, 2] = 10
        return face_shape * focal / div

    def forward(self, coeff, mat_inverse, src_sz=256):
        # mat_reverse = None
        shp_coeff, tex_coeff, angles, translation, gamma = self.Split_coeff(coeff)
        shape, tex = self.facemodel(shp_coeff, params_tex=tex_coeff) # tex_coeff
        rotation = self.Compute_rotation_matrix(angles)
        verts = self.Rigid_transform_block(shape, rotation, translation)
        verts = self.Orthogonal_projection_block(verts)
        # print(mat_reverse[:, 2])

        # to image coordinates
        bz = mat_inverse.shape[0]
        mat_img = torch.Tensor([[1, 0, 0, 112],
                                [0, -1, 0, 112],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).to(coeff.device).repeat(bz, 1, 1)

        mat_inverse = torch.bmm(mat_inverse, mat_img)
        verts = torch.bmm(torch.cat((verts, torch.ones_like(verts)[:, :, 0:1]), dim=2), mat_inverse.transpose(1,2))
        keypoints = verts[:, self.keypoints, :2]
        # to NDC
        # mat_ndc = torch.Tensor([[1 / 128., 0, 0, -1],
        #                         [0, -1 / 128., 0, 255 / 128 - 1],
        #                         [0, 0, 1 / 128., 0]]).to(self.device).repeat(bz, 1, 1)

        mat_ndc = torch.Tensor([[2/src_sz, 0, 0, -1],
                                [0, -2/src_sz, 0, 1-2/src_sz],
                                [0, 0, 2/src_sz, 0]]).to(coeff.device).repeat(bz, 1, 1)

        verts = torch.bmm(verts, mat_ndc.transpose(1, 2))

        return verts, tex, gamma, keypoints # verts*1.2 for figure1


if __name__ == '__main__':
    from utils.cv import tensor2img
    from utils.microsoft_align import Preprocess
    import cv2
    import numpy as np
    import pickle
    import os


    def get_ldmk(ldmk_pth, fv_point=True):
        with open(ldmk_pth, 'rb') as f:
            ldmk = pickle.load(f)
        if not fv_point:
            lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
            ldmk = np.stack(
                [ldmk[lm_idx[0], :], np.mean(ldmk[lm_idx[[1, 2]], :], 0), np.mean(ldmk[lm_idx[[3, 4]], :], 0),
                 ldmk[lm_idx[5], :], ldmk[lm_idx[6], :]], axis=0)
            ldmk = ldmk[[1, 2, 0, 3, 4], :]
        return ldmk


    device = 'cuda:0'
    reconstructor = Face3D()
    pth = '/media/xn/1TDisk/CelebAMask-HQ/celebahq256/'
    ldmk_pth = '/media/xn/1TDisk/CelebAMask-HQ/ldmk256/'
    coeff_pth = '/media/xn/SSD1T/CelebAMask-HQ/microsoft_coeff/'
    for name in os.listdir(pth):
        coeff_pth_i = os.path.join(coeff_pth, name.split('.')[0] + '.pkl')
        with open(coeff_pth_i, 'rb') as f:
            coeff = pickle.load(f)
            coeff = torch.from_numpy(coeff).type(torch.float32).to(device)

        ldmk_pth_i = os.path.join(ldmk_pth, name.split('.')[0] + '.pkl')
        with open(ldmk_pth_i, 'rb') as f:
            ldmk = pickle.load(f)

        I = cv2.imread(os.path.join(pth, name))
        I_new, new_ldmk, mat_inverse = Preprocess(I, ldmk)

        rgb_bfm, light, keypoints, mask = reconstructor.Reconstruction_block(coeff, torch.from_numpy(mat_inverse).type(
            torch.float32).to(device))
        # rgb_bfm_show = tensor2img(rgb_bfm).copy()
        # for pt in keypoints[0]:
        #     cv2.circle(rgb_bfm_show, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        # rgb_bfm_show = rgb_bfm_show[:, :, ::-1]
        # loss = ((I - rgb_bfm_show) ** 2).mean()
        # print(loss)

        # merge = (I * 0.5 + rgb_bfm_show * 0.5).astype('uint8')
        # light = tensor2img(light)
        # show = np.concatenate((I, rgb_bfm_show, merge, light), axis=1)
        # cv2.imshow('show', show)
        # for pt in keypoints[0]:
        #     cv2.circle(I_new, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        # cv2.imshow('show', I_new)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     break

    cv2.destroyAllWindows()