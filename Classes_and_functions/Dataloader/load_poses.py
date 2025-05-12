# Ghiotto Andrea   2118418

from Classes_and_functions import imports

class Pose:
    def __init__(self, label_path):
        with open(label_path, "r") as file:
            line = file.readline()[:-1].split()
            self.x, self.y, self.z, self.r, self.p,  self.yaw, = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])
            self.yaw = -self.yaw * 180 / imports.math.pi
    def __call__(self):
        return imports.torch.Tensor([self.x, self.y, self.yaw])