
def print_path():
    import os
    print(os.getcwd())

def load_models():
    # load detect and crop model
    from insightface_func.face_detect_crop_single import Face_detect_crop

    app = Face_detect_crop(name='antelope', root='code/SimSwap/insightface_func/models/')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode="None")

    # load simswap model
    from models.models import create_model
    from options.test_options import TestOptions

    opt = TestOptions().parse()
    opt.Arc_path = 'code/Simswap/arcface_model/arcface_checkpoint.tar'

    model = create_model(opt)
    model.eval()

    return app, model


def get_embedding(app, model):
    # embedding vector
    import cv2
    from PIL import Image
    import torch.nn.functional as F
    from torchvision import transforms

    transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = cv2.imread("serving/data/example/id.jpg")
    img_crop, _ = app.get(img, 224)
    img_crop = Image.fromarray(cv2.cvtColor(img_crop[0],cv2.COLOR_BGR2RGB))

    img_a = transformer_Arcface(img_crop)

    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
    img_id = img_id.cuda()

    img_down = F.interpolate(img_id, size=(112,112))
    latent_id = model.netArc(img_down)

    return latent_id


