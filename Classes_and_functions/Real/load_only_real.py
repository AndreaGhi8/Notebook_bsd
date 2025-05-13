# Ghiotto Andrea   2118418

from Classes_and_functions import imports
from Classes_and_functions.Dataloader import load_poses

class SonarDescriptorOnlyRealDataset(imports.Dataset):
    def __init__(self, database4val=None):
        self.training = database4val is None
        
        self.img_source = imports.glob.glob("Datasets/placerec_trieste_updated/imgs/*")
        self.img_labels = imports.glob.glob("Datasets/placerec_trieste_updated/pose/*")

        self.img_source.sort()
        self.img_labels.sort()
        if self.training:
            self.img_source = imports.np.array(self.img_source)[:710]
            self.img_labels = imports.np.array(self.img_labels)[:710]
        else:
            self.img_source = imports.np.array(self.img_source)[715:1500]
            self.img_labels = imports.np.array(self.img_labels)[715:1500]

        self.imgs = self.img_source
        self.pose_paths = self.img_labels
        self.poses = imports.np.zeros((len(self.img_source), 3))

        self.synth = len(self.img_source)

        cont=0
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

        if not self.training:
            self.closest_poses = self.correlate_poses(database4val)
          
    def __len__(self):
        return len(self.imgs)

    def correlate_poses(self, database4val):
        self.closest_indices = imports.np.zeros(self.poses.shape[0])
        for idx in range(self.poses.shape[0]):
            self.closest_indices[idx] = database4val.gtquery_real(self.poses[idx])
        self.closest_indices = self.closest_indices.astype(int)

    def crop_and_resize_image(self, image:imports.np.ndarray, rotation:float) -> imports.np.ndarray:
        shift = int(1536*rotation / 360)
        image = image[:, (512-shift):(1024-shift)]
        return image

    def computeDescriptors(self, net):
        self.descriptors=[]
        print("computing dataset descriptors")
        net.eval()
        if not self.training:
            self.shifts=imports.np.array([0])
        with imports.torch.no_grad():
            for idx in imports.tqdm(range(self.synth)):
                img_path = self.imgs[idx]
                image = imports.cv2.cvtColor(imports.cv2.imread(img_path), imports.cv2.COLOR_BGR2GRAY)
                image_ = imports.np.copy(image)
                image_ = self.pad(imports.torch.Tensor(image_))
                image_ = imports.torch.Tensor(image_)
                image_ = ( image_ / 255.0 ) - 0.5
                image_ = image_[None] * imports.np.pi
                sin, cos = imports.torch.sin(image_), imports.torch.cos(image_)
                image_ = imports.torch.cat([sin, cos]).cuda()[None]
                descriptor = net(image_, reco=False)[0, :].detach().cpu().numpy()
                self.descriptors.append(descriptor)
        print("descriptors computed!")

    def gtquery_real(self, realpose):
        x,y,yaw_deg = realpose
        yaw_deg = (90+yaw_deg)%360
        return self.gtquery(x, y, yaw_deg)

    def gtquery(self, x, y, yaw_deg):
        
        dist_matrix = imports.torch.cdist(imports.torch.Tensor([x,y]).unsqueeze(0), self.poses[:self.synth, :2].unsqueeze(0)).squeeze()
    
        closest_index = imports.torch.argmin(dist_matrix, dim=-1)
    
        return closest_index
   
    def query(self, query_descriptor):
        self.norms = imports.np.zeros(len(self.descriptors))
        for i in range(len(self.descriptors)):
            self.norms[i] = imports.np.sum((self.descriptors[i] - query_descriptor)**2)
        return self.norms.argmin()

    def __getitem__(self, idx):

        img_path = self.imgs[idx]
        image = imports.cv2.cvtColor(imports.cv2.imread(img_path), imports.cv2.COLOR_BGR2GRAY)      
        pose = imports.np.copy(self.poses[idx])    
        image = imports.cv2.resize(image, self.img_size)

        image = self.pad(imports.torch.Tensor(image))
        image = ( image / 255.0 ) - 0.5

        image_ = image[None] * imports.np.pi
        sin, cos = imports.torch.sin(image_), imports.torch.cos(image_)
        
        return imports.torch.cat([sin, cos]), imports.torch.Tensor(image)[None], pose, img_path, self.img_labels[idx]