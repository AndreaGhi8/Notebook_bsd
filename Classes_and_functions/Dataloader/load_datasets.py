# Ghiotto Andrea   2118418

from Classes_and_functions import imports
from Classes_and_functions.Dataloader import load_poses
from Classes_and_functions.Dataloader import functions

class SonarDescriptorDatasetFull(imports.Dataset):
    def __init__(self, datapath, database4val=None):
        self.img_source = imports.glob.glob(imports.os.path.join(datapath, "imgs", "*"))
        self.img_labels = imports.glob.glob(imports.os.path.join(datapath, "poses", "*"))
        self.img_source.sort()
        self.img_labels.sort()
        self.img_source = imports.np.array(self.img_source)
        self.img_labels = imports.np.array(self.img_labels)
        
        self.training = database4val is None

        if self.training:
            self.idxs = imports.np.arange(0, len(self.img_source), 1, dtype=int)
            imports.np.random.shuffle(self.idxs)
            self.train_idxs_num = self.idxs.shape[0]
            self.train_idxs = self.idxs[:self.train_idxs_num]
    
            self.img_source = self.img_source[self.train_idxs]
            self.img_labels = self.img_labels[self.train_idxs]
        else:
            self.idxs = imports.np.arange(0, len(self.img_source), 1, dtype=int)
            imports.np.random.shuffle(self.idxs)
            self.valid_idxs_num = self.idxs.shape[0]
            self.valid_idxs = self.idxs[:self.valid_idxs_num]
            
            self.img_source = self.img_source[self.valid_idxs]
            self.img_labels = self.img_labels[self.valid_idxs]

        if False and self.training:     # REMOVE "False and" WHEN NEED TO TRAIN ALSO REAL
            idxs = imports.np.arange(0, 1700, 1, dtype=int)
            imports.np.random.shuffle(idxs)
            idxs = idxs[:1700]
            
            self.realimg_source = imports.glob.glob("Datasets/placerec_trieste_updated/imgs/*")
            self.realimg_source.sort()
            self.realimg_source = imports.np.array(self.realimg_source)[idxs]

            self.realimg_labels = imports.glob.glob("Datasets/placerec_trieste_updated/pose/*")
            self.realimg_labels.sort()
            self.realimg_labels = imports.np.array(self.realimg_labels)[idxs]
            
            self.imgs       = imports.np.concatenate((self.img_source, self.realimg_source))
            self.pose_paths = imports.np.concatenate((self.img_labels, self.realimg_labels))
            
            self.descriptors=[]
            
            self.poses = imports.np.zeros((len(self.img_source)+len(self.realimg_source), 3))
            
        else:
            self.imgs = self.img_source
            self.poses = imports.np.zeros((len(self.img_source), 3))
            self.pose_paths = self.img_labels
        
        self.synth = len(self.img_source)

        if not self.training:
            self.rotations = imports.np.zeros(len(self.img_labels))
        
        cont=0
        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            pose = load_poses.Pose(lab_path)()
            self.poses[i] = pose

        self.pad = imports.nn.ZeroPad2d((0, 0, 28, 28))
        self.img_size = (256, 200)
        self.min_dx, self.min_dy = 335, -458
        self.poses[:, 0]-=self.min_dx
        self.poses[:, 1]-=self.min_dy
        self.poses[:, :2]*=10
        
        if self.training:
            self.poses = imports.torch.Tensor(self.poses)
        else:
            self.closest_poses = self.correlate_poses(database4val)
            
    def __len__(self):
        return len(self.imgs)

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

    def correlate_poses(self, database4val):
        self.closest_indices = imports.np.zeros(self.poses.shape[0])
        for idx in range(self.poses.shape[0]):
            self.closest_indices[idx] = database4val.gtquery_synth(self.poses[idx])
        self.closest_indices = self.closest_indices.astype(int)

    def gtquery_synth(self, synthpose):
        x,y,yaw_deg = synthpose
        # yaw_deg = (90 + yaw_deg) % 360        # REMOVE THE # AND CHECK WHEN NEED TO TRAIN ALSO REAL
        #print("synthpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg)
    
    def gtquery_real(self, realpose):
        x,y,yaw_deg = realpose
        #yaw_deg = (90+yaw_deg)%360
        #print("realpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg) 

    def gtquery(self, x, y, yaw_deg):
        
        dist_matrix = imports.torch.cdist(imports.torch.Tensor([x,y]).unsqueeze(0), self.poses[:self.synth, :2].unsqueeze(0)).squeeze()

        _, cand_indx = imports.torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)
        
        candidates = self.poses[:self.synth, 2][cand_indx]
        candidates = imports.torch.Tensor([functions.parse_pose([0,0,cand])[3] for cand in candidates])

        diff_yaw = imports.torch.min(abs(candidates-yaw_deg), abs(360-abs(candidates-yaw_deg)))

        min_yaw_idx = imports.torch.argmin(diff_yaw, dim=-1)

        closest_index = cand_indx[min_yaw_idx]
        closest_index = closest_index.item()
        
        return closest_index
   
    def query(self, query_descriptor):
        self.norms = imports.np.zeros(len(self.descriptors))
        for i in range(len(self.descriptors)):
            self.norms[i] = imports.np.sum((self.descriptors[i] - query_descriptor)**2)
        return self.norms.argmin()

    def crop_and_resize_image(self, image:imports.np.ndarray, rotation:float) -> imports.np.ndarray:
        image=image
        return image

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = imports.cv2.cvtColor(imports.cv2.imread(img_path), imports.cv2.COLOR_BGR2GRAY)
       
        pose = imports.np.copy(self.poses[idx])
        
        image = imports.cv2.resize(image, self.img_size)

        image = self.pad(imports.torch.Tensor(image))
        image = imports.torch.Tensor(image)
        image = (image / 255.0) - 0.5

        image_ = image[None] * imports.np.pi
        sin, cos = imports.torch.sin(image_), imports.torch.cos(image_)
        
        return imports.torch.cat([sin, cos]), imports.torch.Tensor(image)[None], pose, img_path, self.img_labels[idx] if idx<self.synth else "aaa", 1 if idx<self.synth else 0