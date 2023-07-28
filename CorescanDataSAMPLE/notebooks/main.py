from dataset import create_dataset
from dataset import create_patch_dataloaders
from train import train

path = '../img-rgb-50u'
label_path = '../img-clm-phy'
# cm_path = '../img-clm-phy'

class_min_dict = {"amphibole":(52,82,52),
              "apophyllite":(0,255,0),
              "aspectral":(209,209,209),
              "biotite":(128,0,0),
              "carbonate":(0,255,255),
              "carbonate-actinolite":(44,109,0),
              "chlorite":(0,192,0),
              "clinochlore":(45,95,45),
              "dickite":(148,138,84),
              "epidote":(188,255,55),
              "iron carbonate":(185,255,255),
              "iron oxide":(255,154,0),
              "gypsum":(213,87,171),
              "kaolinite":(191,183,143),
              "montmorillonite":(175,175,255),
                "NA":(0,0,0),
              "nontronite":(105,105,255),
              "phlogopite":(88, 0, 0),
              "prehnite":(70, 70, 220),
              "sericite":(58,102,156),
              "silica":(166,166,166),
              "tourmaline":(255,0,0),
                "UNK1":(83, 141, 213),
                "UNK2":(155, 187, 89),
                "UNK3":(0, 108, 105),
              "vermiculite":(95, 100, 200)
             }

train_loader, val_loader = create_dataset(path, label_path, 'chlorite',1) #Returns data loaders
model = train(train_loader, val_loader, 'chlorite', class_min_dict) #Train model
