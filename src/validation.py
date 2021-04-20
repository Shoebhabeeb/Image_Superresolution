from torch.utils import data as data

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".fits"])
    def __len__(self):
        return len(self.image_filenames)
class SuperResolve(data.Dataset):
    def __init__(self, data_path, input_transform, target_transform):
        self.data_path = data_path
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.img_files = make_dataset(data_path)
        
    def __getitem__(self, index):
        _img, _target = default_fits_loader(self.img_files[index], (2048,2048))
#        _img = _img[1024:3072, 1024:3072]
        _img = np.clip(_img,-1500,1500)
        _img = resize(_img, (1024, 1024),preserve_range=True,anti_aliasing=True)
        _target = np.clip(_target,-1500,1500)
        if self.input_transform:
            _img = self.input_transform(_img)
            _img[_img != _img] = -1
        if self.target_transform:
            _target = self.target_transform(_target)
            _target[_target != _target] = -1
        _data = (_img, _target)
        
        return _data
    def __len__(self):
        return len(self.img_files)    
