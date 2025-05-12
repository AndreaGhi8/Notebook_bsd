# Ghiotto Andrea   2118418

from Classes_and_functions import imports
from Classes_and_functions.Dataloader import load_poses

class SonarDescriptorRealDataset(imports.Dataset):
    def __init__(self, datapath, database4val=None):
        self.img_source = imports.glob.glob("Datasets/placerec_trieste_updated/imgs/*")
        self.img_labels = imports.glob.glob("Datasets/placerec_trieste_updated/pose/*")

        self.img_source.sort()
        self.img_labels.sort()
        self.img_source = imports.np.array(self.img_source)
        self.img_labels = imports.np.array(self.img_labels)

        self.imgs = self.img_source
        self.pose_paths = self.img_labels
        self.poses = imports.np.zeros((len(self.img_source), 3))

        self.synth = len(self.img_source)

        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            self.poses[i] = load_poses.Pose(lab_path)()

        self.pad = imports.nn.ZeroPad2d((0, 0, 28, 28))
        self.img_size = (256, 200)
        self.min_dx, self.min_dy = 335, -458
        self.poses[:, 0]-=self.min_dx
        self.poses[:, 1]-=self.min_dy
        self.poses[:, :2]*=10

        self.poses = imports.torch.Tensor(self.poses)
        
    def __len__(self):
        return len(self.imgs)

    def crop_and_resize_image(self, image:imports.np.ndarray, rotation:float) -> imports.np.ndarray:
        shift = int(1536*rotation / 360)
        image = image[:, (512-shift):(1024-shift)]
        return image

    def __getitem__(self, idx):

        img_path = self.imgs[idx]

        image = imports.cv2.cvtColor(imports.cv2.imread(img_path), imports.cv2.COLOR_BGR2GRAY)
        image = imports.cv2.flip(image, 0)
       
        pose = imports.np.copy(self.poses[idx])
        
        image = imports.cv2.resize(image, self.img_size)
        image = self.pad(imports.torch.Tensor(image))
        image = ( image / 255.0 ) - 0.5
        image_ = image[None] * imports.np.pi
        sin, cos = imports.torch.sin(image_), imports.torch.cos(image_)
        
        return imports.torch.cat([sin, cos]), imports.torch.Tensor(image)[None], pose, img_path, self.img_labels[idx]