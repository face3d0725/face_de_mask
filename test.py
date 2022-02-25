from models.face_encoder_res50 import FaceEncoder
from models.face_decoder import Face3D
from utils.cv import tensor2img
import torch
from utils.simple_renderer import SimpleRenderer
from utils.cv import img2tensor
from utils.seg_util import tensor_close, tensor_dilate
import numpy as np
import cv2
import os
import pathlib
from scipy.io import loadmat
import pickle
from models.networks_inpaint import Generator
import torch.nn.functional as F


def load_lm3d():
    pth = './BFM/similarity_Lm3D_all.mat'
    Lm3D = loadmat(pth)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


lm3D = load_lm3d()


def POS(xp, x):
    npts = xp.shape[0]
    if npts == 68:
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        xp = np.stack([xp[lm_idx[0], :], np.mean(xp[lm_idx[[1, 2]], :], 0), np.mean(xp[lm_idx[[3, 4]], :], 0),
                       xp[lm_idx[5], :], xp[lm_idx[6], :]], axis=0)
        xp = xp[[1, 2, 0, 3, 4], :]
        npts = 5
    if npts == 29:
        lm_idx = np.array([20, 8, 10, 9, 11, 22, 23])
        xp = np.stack([xp[lm_idx[0], :], np.mean(xp[lm_idx[[1, 2]], :], 0), np.mean(xp[lm_idx[[3, 4]], :], 0),
                       xp[lm_idx[5], :], xp[lm_idx[6], :]], axis=0)
        xp = xp[[1, 2, 0, 3, 4], :]
        npts = 5

    A = np.zeros([2 * npts, 8])
    x = np.concatenate((x, np.ones((npts, 1))), axis=1)
    A[0:2 * npts - 1:2, 0:4] = x

    A[1:2 * npts:2, 4:] = x

    b = np.reshape(xp, [-1, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def process_img(img, t, s, target_size=256):
    h0, w0 = img.shape[:2]
    scale = 116. / s
    dx = -(t[0, 0] * scale - target_size / 2)
    dy = -((h0 - t[1, 0]) * scale - target_size / 2)
    mat = np.array([[scale, 0, dx],
                    [0, scale, dy]])

    corners = np.array([[0, 0, 1], [w0 - 1, h0 - 1, 1]])
    new_corners = (corners @ mat.T).astype('int32')
    pad_left = max(new_corners[0, 0], 0)
    pad_top = max(new_corners[0, 1], 0)
    pad_right = min(new_corners[1, 0], target_size - 1)
    pad_bottom = min(new_corners[1, 1], target_size - 1)
    mask = np.zeros((target_size, target_size, 3))
    mask[:pad_top, :, :] = 1
    mask[pad_bottom:, :, :] = 1
    mask[:, :pad_left, :] = 1
    mask[:, pad_right:, :] = 1
    img_affine = cv2.warpAffine(img, mat, (target_size, target_size), borderMode=cv2.BORDER_REFLECT_101)
    img_affine = img_affine.astype('float32') * (1 - mask) + cv2.blur(img_affine, (10, 10)) * mask
    img_affine = img_affine.astype('uint8')
    return img_affine, mat


def show_ldmk(img, lm):
    for pt in lm:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    return img


def Preprocess(img, lm):
    h0, w0 = img.shape[:2]
    lm_ = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)
    t, s = POS(lm_, lm3D)
    img_new, mat = process_img(img, t, s)
    lm_affine = np.concatenate((lm[:, :2], np.ones((lm.shape[0], 1))), axis=1)
    lm_affine = lm_affine @ mat.T
    return img_new, lm_affine, mat


class Test():
    def __init__(self):
        device = 'cuda:0'
        self.img_root = './input/'
        self.ldmk_root = './ldmk/'
        self.img_lst = os.listdir(self.img_root)
        self.renderer = torch.nn.DataParallel(SimpleRenderer(device).to(device), [0, 1])
        self.face_decoder = torch.nn.DataParallel(Face3D(device).to(device), [0, 1])
        self.generator = torch.nn.DataParallel(Generator().to(device), [0, 1])
        self.tri = self.face_decoder.module.facemodel.tri.unsqueeze(0)
        self.face_encoder = torch.nn.DataParallel(FaceEncoder().to(device), [0, 1])
        state_dict = torch.load('./ckpts/3D_it_500000.pkl')
        self.face_encoder.load_state_dict(state_dict)
        self.face_encoder.eval()
        state_dict = torch.load('./ckpts/inpaint_it_200000.pkl')
        self.generator.load_state_dict(state_dict['gen'])

        self.data = {}
        self.title = 'show'
        cv2.namedWindow(self.title)
        with open('./ckpts/expression_mu_std.pkl', 'rb') as f:
            exp = pickle.load(f)
        self.std = exp['std']
        self.std[2] *= 4
        self.std[5] *= 4
        self.std[50]*=5
        self.std[51]*=5
        self.std[81]*=4
        self.std[83] *=1.5
        self.std[84] = self.std[83]
        self.std[89] *= 4
        self.mu = exp['mean']
        # self.top_idx = exp['idx']
        self.top_idx = np.array([0, 1, 2, 3, 4, 5, 8, 16, 33, 50, 51, 80, 81, 82, 83, 84, 85, 89, 144, 145, 146])
        self.step_sz = self.std * 6 / 20
        self.coeff_clone = torch.zeros(1, 257)
        self.stop = False
        self.noise = torch.rand(1, 1, 256, 256)
        # self.generator#.eval()

    def create_trackbars(self):
        def create_callback(idx):
            def callback(val):
                value = self.coeff_clone[0, idx]
                self.data['coeff'][0, idx] = value + (val - 10) * self.step_sz[idx]
                self.de_mask()

            return callback

        for idx in self.top_idx[: 30]:
            value = self.data['coeff'][0, idx]
            location = (value - self.mu[idx] + 3 * self.std[idx]) / self.step_sz[idx]
            location = int(location)
            cv2.createTrackbar('{}'.format(idx), self.title, location, 20, create_callback(idx))

    def load_data(self, name):
        img_pth = os.path.join(self.img_root, name)
        img_init = cv2.imread(img_pth)[..., ::-1]
        ldmk_name = name.split('.')[0] + '.pkl'
        ldmk_pth = os.path.join(self.ldmk_root, ldmk_name)
        with open(ldmk_pth, 'rb') as f:
            ldmk = pickle.load(f)

        img, _, mat = Preprocess(img_init, ldmk)
        mat_inv = np.array([[1 / mat[0, 0], 0, -mat[0, 2] / mat[0, 0]],
                            [0, 1 / mat[1, 1], -mat[1, 2] / mat[1, 1]]])

        img_tensor = img2tensor(img).cuda()
        coeff, seg1, seg2, seg3 = self.face_encoder(img_tensor)
        # verts, tex, id_coeff, ex_coeff, tex_coeff, gamma, ldmk_pred = self.face_decoder(coeff)
        mask = (torch.sigmoid(seg1) > 0.1).type(torch.float32)
        mask = tensor_dilate(mask, 5)
        mask = tensor_dilate(mask, 3)
        seg = tensor2img(mask) // 255
        # seg = cv2.dilate(seg, np.ones((3, 3), np.uint8))
        seg_pth = os.path.join('mask/{}'.format(name))
        seg = cv2.warpAffine(seg, mat_inv, (256, 256)) * 255

        if not os.path.exists(seg_pth):
            cv2.imwrite(seg_pth, seg)


        return img_init, img, img_tensor, seg, coeff, mask, mat_inv

    def seamless(self, src, dst, mask):
        x, y, W, H = cv2.boundingRect(mask)
        mask_crop = mask[y:y + H, x:x + W]
        src_crop = src[y:y + H, x:x + W]
        ctr = (x + W // 2, y + H // 2)
        out = cv2.seamlessClone(src_crop, dst, mask_crop, ctr, cv2.NORMAL_CLONE)
        return out

    def de_mask(self):
        img, img_tensor, mask, seg, coeff, img_init, mat = \
            self.data['I'], self.data['I_t'], self.data['mask'],\
            self.data['seg'], self.data['coeff'], self.data['img_init'], self.data['mat']

        verts, tex, id_coeff, ex_coeff, tex_coeff, gamma, ldmk_pred = self.face_decoder(coeff)
        out = self.renderer(verts, self.tri, size=256, colors=tex, gamma=gamma, front_mask=None)
        recon = out['rgb']
        ldmk_np = ldmk_pred.squeeze().cpu().detach().numpy()
        ldmk_np = np.concatenate((ldmk_np, np.ones((68, 1))), axis=1).dot(mat.T)
        self.data['ldmk'] = ldmk_np

        recon_show = tensor2img(torch.clamp(recon+1-out['mask'], 0, 1))
        I_de_occ = (img_tensor * (1 - mask) + self.noise).cuda()
        occ_recon = torch.cat((I_de_occ, recon, mask), dim=1)
        inpaint = self.generator(occ_recon)
        inpaint_show = tensor2img(inpaint)
        inpaint_show = cv2.warpAffine(inpaint_show, mat, (256, 256),borderValue=(255,255,255))
        recon_show = cv2.warpAffine(recon_show, mat, (256, 256), borderValue=(255,255,255))
        inpaint_show = img_init*(1-seg[...,np.newaxis]//255) + inpaint_show*(seg[...,np.newaxis]//255)
        inpaint_show[-1,:,:] = np.array([255, 255, 255])
        # inpaint_show = self.seamless(inpaint_show, img_init, seg)
        self.data['inpaint_show'] = inpaint_show
        self.data['recon_show'] = recon_show
        self.data['show'] = np.concatenate((img_init, recon_show, inpaint_show), axis=1)

    def forward(self):
        for name in self.img_lst:
            cv2.namedWindow(self.title)
            img_init, img, img_tensor, seg, coeff, mask, mat = self.load_data(name)
            self.data = {'I': img, 'I_t': img_tensor, 'mask': mask, 'seg': seg, 'coeff': coeff, 'img_init': img_init, 'mat': mat}
            self.noise = torch.rand_like(mask)
            self.coeff_clone = coeff.clone()
            self.create_trackbars()
            self.de_mask()
            count = 0
            while True:
                show = self.data['show'][..., ::-1]
                cv2.imshow(self.title, show)
                key = cv2.waitKey(5)
                if key == ord('q'):
                    self.stop = True
                    break
                elif key == ord('n'):
                    cv2.destroyWindow(self.title)
                    break
                if key == ord('r'):
                    self.noise = torch.rand_like(mask)
                    self.de_mask()
                if key == ord('d'):
                    self.data['coeff']= self.coeff_clone.clone()
                    cv2.destroyWindow(self.title)
                    cv2.namedWindow(self.title)
                    self.create_trackbars()
                    self.de_mask()
                if key == ord('s'):
                    sv_name = name.split('.')[0]+'_{}.png'.format(count)
                    img_pth = os.path.join('animation/res/', sv_name)
                    cv2.imwrite(img_pth, self.data['inpaint_show'][...,::-1])
                    rec_pth = os.path.join('animation/3D/', sv_name)
                    cv2.imwrite(rec_pth, self.data['recon_show'][...,::-1])

                    ldmk_sv_name = name.split('.')[0]+'_{}.txt'.format(count)
                    ldmk_pth = os.path.join('animation/ldmk/', ldmk_sv_name)
                    np.savetxt(ldmk_pth, self.data['ldmk'])
                    count += 1
                    # cv2.imwrite('output_ours/{}'.format(name), self.data['inpaint_show'][...,::-1])

            if self.stop:
                break

            # if key == ord('s'):


if __name__ == '__main__':
    tester = Test()
    tester.forward()
