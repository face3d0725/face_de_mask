from models.bfm import BfmExtend
import torch
from torch import nn
from utils.simple_renderer import SimpleRenderer


class Face3D(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.facemodel = BfmExtend().to(device)
        self.keypoints = self.facemodel.keypoints

    def Split_coeff(self, coeff):
        # shp_coeff = coeff[:, :144]  # id_coeff :80, ex_coeff 80:144
        id_coeff = coeff[:, :80]
        ex_coeff = coeff[:, 80:144]
        tex_coeff = coeff[:, 144:224]
        angles = coeff[:, 224:227]
        gamma = coeff[:, 227:254]
        translation = coeff[:, 254:257]

        return id_coeff, ex_coeff, tex_coeff, angles, translation, gamma

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

    # def Projection_block(self, face_shape, focal=1015.0): 1015/224*256
    def Projection_block(self, face_shape, focal=1160.0):
        # the reconstructed coordinates are from -128 to 128
        div = torch.ones_like(face_shape)  # *10
        div[:, :, 0] = 10 - face_shape[:, :, 2]
        div[:, :, 1] = 10 - face_shape[:, :, 2]
        div[:, :, 2] = 10
        return face_shape * focal / div

    def forward(self, coeff, size=256):
        # mat_reverse = None
        id_coeff, ex_coeff, tex_coeff, angles, translation, gamma = self.Split_coeff(coeff)
        shp_coeff = torch.cat((id_coeff, ex_coeff), dim=1)
        shape, tex = self.facemodel(shp_coeff, tex_coeff)
        rotation = self.Compute_rotation_matrix(angles)
        verts = self.Rigid_transform_block(shape, rotation, translation)
        focal = 1015/224*size
        verts = self.Projection_block(verts, focal)
        # print(mat_reverse[:, 2])

        # get keypoints under image coordinates
        half_sz = size/2
        bz = coeff.shape[0]
        keypoints = verts[:, self.keypoints, :2]
        keypoints = keypoints * torch.tensor([[[1., -1.]]]).to(keypoints.device)
        keypoints = keypoints + torch.tensor([[[half_sz, half_sz]]]).to(keypoints.device)

        # to NDC

        mat_ndc = torch.Tensor([[1 / half_sz, 0, 0],
                                [0, 1 / half_sz, 0],
                                [0, 0, 1 / half_sz]]).to(verts.device).repeat(bz, 1, 1)

        verts = torch.bmm(verts, mat_ndc.transpose(1, 2))
        return verts, tex, id_coeff, ex_coeff, tex_coeff, gamma, keypoints


if __name__ == '__main__':
    from utils.cv import tensor2img
    import cv2
    from utils.simple_renderer import SimpleRenderer

    device = 'cuda:0'
    renderer = SimpleRenderer(device).to(device)

    model = Face3D().to(device)
    tri = model.facemodel.tri.unsqueeze(0)

    x = torch.zeros(1, 257).to(device)
    verts, tex, id_coeff, ex_coeff, tex_coeff, gamma, landmarks = model(x)
    # verts, faces, size=128, colors=None, gamma=None, eros=7
    out = renderer(verts, tri, size=256, colors=tex, gamma=None)
    rgb = out['rgb']
    rgb = tensor2img(rgb).copy()
    landmarks = landmarks.squeeze()
    for pt in landmarks:
        cv2.circle(rgb, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

    cv2.imshow('bfm', rgb[...,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()