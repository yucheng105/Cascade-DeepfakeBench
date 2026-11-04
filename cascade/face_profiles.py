"""Model-specific face geometry, following the pinned upstream inference paths.

DeepfakeBench f188b1c1 (dlib) and SelfBlendedImages d6d9a351 (SBI).
The RetinaFace inference implementation is the unmodified retinaface_pytorch package.
"""
from pathlib import Path

import cv2
import numpy as np


def dlib_align(rgb, points):
    from skimage.transform import SimilarityTransform
    dst=np.array([[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],
                  [33.5493,92.3655],[62.7299,92.2041]],dtype=np.float32)
    dst[:,0]+=8.
    dst[:,0]=dst[:,0]*256/112
    dst[:,1]=dst[:,1]*256/112
    margin=256*(1.3-1)/2
    dst[:,0]+=margin;dst[:,1]+=margin
    dst[:,0]*=256/(256+2*margin);dst[:,1]*=256/(256+2*margin)
    transform=SimilarityTransform()
    if not transform.estimate(np.asarray(points,dtype=np.float32),dst):
        raise ValueError('Degenerate dlib landmarks')
    crop=cv2.warpAffine(rgb,transform.params[:2,:],(256,256))
    crop=cv2.resize(crop,(256,256))
    return cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),transform.params[:2,:].tolist()


def sbi_crop(rgb,bbox):
    x0,y0,x1,y1=bbox;h,w=rgb.shape[:2]
    dx=(x1-x0)/8;dy=(y1-y0)/8
    left=max(0,int(x0-dx));right=min(w,int(x1+dx)+1)
    top=max(0,int(y0-dy));bottom=min(h,int(y1+dy)+1)
    crop=rgb[top:bottom,left:right]
    if not crop.size:raise ValueError('Empty SBI crop')
    return cv2.cvtColor(cv2.resize(crop,(380,380)),cv2.COLOR_RGB2BGR)


class DlibFaces:
    def __init__(self,predictor):
        import dlib
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor(str(predictor))

    def __call__(self,image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        faces=self.detector(rgb,1)
        if not faces:
            self.last_rejection_reason='no_face';return []
        face=max(faces,key=lambda r:r.width()*r.height())
        shape=self.predictor(rgb,face)
        points=[[shape.part(i).x,shape.part(i).y] for i in (37,44,30,49,55)]
        crop,matrix=dlib_align(rgb,points)
        # The reference's second detector/predictor deliberately receives BGR.
        aligned=self.detector(crop,1)
        if not aligned:
            self.last_rejection_reason='post_alignment_no_face';return []
        landmark=self.predictor(crop,aligned[0])
        all_points=[[landmark.part(i).x,landmark.part(i).y] for i in range(landmark.num_parts)]
        return [(crop,dict(bbox=[face.left(),face.top(),face.right(),face.bottom()],
                          landmarks=points,aligned_landmarks=all_points,alignment_matrix=matrix))]


class RetinaFaces:
    def __init__(self,checkpoint,device):
        import torch
        from retinaface.predict_single import Model
        torch.set_num_threads(4)
        torch.manual_seed(1)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        self.detector=Model(max_size=2048,device=device)
        self.detector.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        self.detector.eval()

    def __call__(self,image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        annotations=self.detector.predict_jsons(rgb)
        valid=[a for a in annotations if len(a.get('bbox',[]))==4 and np.isfinite(a['bbox']).all()]
        if not valid:
            self.last_rejection_reason='no_face';return []
        sizes=[(a['bbox'][2]-a['bbox'][0])*(a['bbox'][3]-a['bbox'][1]) for a in valid]
        # The reference crops every detected face before filtering by area.
        try:crops=[sbi_crop(rgb,a['bbox']) for a in valid]
        except ValueError:
            self.last_rejection_reason='invalid_bbox_crop';return []
        maximum=max(sizes)
        return [(crop,dict(bbox=a['bbox'],landmarks=a['landmarks'],detection_score=float(a['score'])))
                for crop,a,size in zip(crops,valid,sizes) if size>=maximum/2]


def build_faces(profile,assets,device):
    if profile=='dlib81':return DlibFaces(Path(assets['predictor']['path']))
    if profile=='retinaface':return RetinaFaces(Path(assets['retinaface']['path']),device)
    if profile=='yunet_legacy':
        from .video_input import YuNetFaces
        detector=YuNetFaces(Path(assets['yunet']['path']))
        def faces(image):
            crop,details=detector.crop(image)
            setattr(faces,'last_rejection_reason',details.get('reason','no_valid_face'))
            return [] if crop is None else [(crop,details)]
        return faces
    raise ValueError('Unknown face profile')
