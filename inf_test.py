
import math
import os
from PIL import Image
#import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import glob
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
#torch.set_grad_enabled(False);


# COCO classes
CLASSES = [
   'person',"traffic light","train","traffic sign","rider",'car','bike','mortor','truck','bus'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def plot_results(pil_img, prob, boxes,image_id,path):
    fig=plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    fig.savefig(f"{path}/{image_id}")
    #plt.show()

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size,device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(torch.device(device))

    return b
    #tensor = tensor.to(device=torch.device("cuda:0"))


def inf(model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    model.to(device)

    model.eval()
    path='**********************************'
    folder='val_output'

    for imgfile in sorted(glob.glob(path+'/*')):
        

        im=Image.open(imgfile)
        img = transform(im).unsqueeze(0)
        img = img.to(device)

        # propagate through the model
        outputs = model(img)
        #print(outputs)
        #exit()

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.1

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size,device)

        file_path=f'./{folder}'
        if not os.path.exists(file_path):  # ディレクトリが存在しない場合、作成する。
            os.makedirs(file_path)
            

        image_id=imgfile.split('\\')[-1]
        plot_results(im, probas[keep], bboxes_scaled,image_id,file_path)
    
    #from detr.makegif import make_gif
    #make_gif(file_path)




