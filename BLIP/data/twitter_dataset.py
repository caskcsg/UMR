import json
import os
from torch.utils.data import Dataset
from PIL import Image
from data.utils import pre_caption


class twitter_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=512):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        #self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        #此处进行改进jpg->png
        #hateful
        #image_path = os.path.join(self.image_root,'%s.png'%ann['image']) 
        # twitter  为jpg结尾     
        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)   

        sentence = ann['sentence']
        if len(ann['sentence']) > self.max_words:          
            sentence = ann['sentence'][:self.max_words]
        #sentence = pre_caption(ann['sentence'], self.max_words)

        return image, sentence, ann['label']
    