import argparse
from functools import partial
from os import path
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import math
import random
import models
from dataset.dataset import SegmentationDataset, segmentation_collate, RDDFPredictDataset #, TrainingSegmentationDataset
from dataset.transform import SegmentationTransform
from modules.deeplab import DeeplabV3
#from modules.bn import InPlaceABN
from inplace_abn import InPlaceABN
import pdb
import cv2

batch_size = 2

#parser = argparse.ArgumentParser(description="Testing script for the Vistas segmentation model")
#parser.add_argument("--scales", metavar="LIST", type=str, default="[0.7, 1, 1.2]", help="List of scales")
#parser.add_argument("--flip", action="store_true", help="Use horizontal flipping")
#parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
#                    help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
#parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
#                    default="final",
#                    help="How the output files are formatted."
#                         " -- palette: color coded predictions"
#                         " -- raw: gray-scale predictions"
#                         " -- prob: gray-scale predictions plus probabilities")
#parser.add_argument("snapshot", metavar="SNAPSHOT_FILE", type=str, help="Snapshot file to load")
#parser.add_argument("data", metavar="IN_DIR", type=str, help="Path to dataset")
#parser.add_argument("output", metavar="OUT_DIR", type=str, help="Path to output folder")
#parser.add_argument("--world-size", metavar="WS", type=int, default=1, help="Number of GPUs")
#parser.add_argument("--rank", metavar="RANK", type=int, default=0, help="GPU id")

random.seed(42)

def save_lossfunction(log_time, log_string, dirName):
    f = open(dirName+"/lossfunction_"+ log_time +".txt","a+")
    f.write(logstring + '\n')
    f.close()    


def save_log_output(logstring, log_time, dirName):
    f = open(dirName+"/output_"+log_time+".txt","a+")
    f.write(logstring+"\n")
    f.close()

def save_preds(log_time, preds, img_timestamp, dirName):
    f= open(dirName+"/preds_output_" +log_time+ ".txt","a+")
#    preds_array = preds[0].data.cpu().numpy()
    preds_array = preds.squeeze().tolist()
    for i in range(len(preds_array)):
        for j in range(len(preds_array[i])):
            f.write(str(preds_array[i][j]) +" " )
        f.write(str(img_timestamp[i])+"\n")
    f.close()
    del preds_array


def get_data(txt_rddf, image_folder, arg_random):
    dados_rddf = []
    if('/' == image_folder[-1]):
        image_folder = image_folder[:-1]
    with open(txt_rddf, 'r') as f:
        dados_rddf = [line.strip().split(" ") for line in f]
    images_path = []
    for i in range(len(dados_rddf)):
        images_path.append(image_folder + '/' + dados_rddf[i][5] + '-r.png')

#            print(image_folder + '/' + dados_rddf[i][5] + '-r.png')
    if(arg_random):
        lista_temp = []
        while(len(lista_temp) < 100):
            rand_number = random.randint(0,len(dados_rddf)-1)
#            print(len(dados_rddf), rand_number)
            if(dados_rddf[rand_number] not in lista_temp):
                lista_temp.append(dados_rddf[rand_number])
        dados_rddf = lista_temp
    return  np.array(dados_rddf), image_folder, images_path


#def flip(x, dim):
#    indices = [slice(None)] * x.dim()
#    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
#                                dtype=torch.long, device=x.device)
#    return x[tuple(indices)]

#def weighted_mse_loss(input, target, weight):
#    return torch.sum(weight * (input - target) ** 2)

class SegmentationModule(nn.Module):
    _IGNORE_INDEX = 255
    
    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return cls #zprobs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(x.size(0), x.size(2), x.size(3), dtype=torch.long)
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, body, head, head_channels, classes, fusion_mode="mean"):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
#        self.cls = nn.Conv2d(head_channels, classes, 1)
        # self.cls = nn.Conv2d(head_channels, classes, 3) #3x3 conv layer
        self.out_vector=nn.Linear(1228800, 4)
#        self.out_vector=nn.Sequential(
#            nn.ReLU(),
#            nn.Linear(1228800, 5)
#            nn.Linear(50 , 5)
#        )

        self.classes = classes
#        if fusion_mode == "mean":
#            self.fusion_cls = SegmentationModule._MeanFusion
#        elif fusion_mode == "voting":
#            self.fusion_cls = SegmentationModule._VotingFusion
#        elif fusion_mode == "max":
#            self.fusion_cls = SegmentationModule._MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [round(s * scale) for s in x.shape[-2:]]
            x_up = functional.upsample(x, size=scaled_size, mode="bilinear")
        else:
            x_up = x

        x_up = self.body(x_up)
        x_up = self.head(x_up)

        #pdb.set_trace()

        x_up = x_up.reshape(x_up.size(0), -1)
        sem_logits = self.out_vector(x_up)

        del x_up
        return sem_logits

    def forward(self, x, scales, do_flip=True):
        return self._network(x, 1)


def main():
    # Load configuration
#    args = parser.parse_args()


    # Train = 0, Eval = 1
    mode = 1
    # Checkpoint path
    chk_path = "weights_batch/ckpoint_1576703507.3473513.pt"
    # Checkpoint save quantity
    chk_qtd = 6
    chk_count = 0

    # Torch stuff
    #torch.cuda.set_device(args.rank)
    torch.cuda.set_device(0) # To get this to run on free RAAMAC GPU - Dominic
    cudnn.benchmark = True

    # Create model by loading a snapshot
    body, head, cls_state = load_snapshot('/home/sabrina/Documents/Inplace_ABN/wide_resnet38_deeplab_vistas.pth.tar')
    model = SegmentationModule(body, head, 256, 5, "mean") # this changes
                                                                      # number of classes
                                                                      # in final model.cls layer
    arg_random = True

#     Create data loader
    transformation = SegmentationTransform(     # Only applied to RGB
        640,
        (0.41738699, 0.45732192, 0.46886091), # rgb mean and std - would this affect training at all?
        (0.25685097, 0.26509955, 0.29067996),
    )
#    my_dataset = RDDFPredictDataset('/dados/rddf_predict/listen_2019-11-29_11:32:36', '/dados/log_png_1003/', transform=transformation)
    my_dataset = RDDFPredictDataset('/dados/rddf_predict/listen_no_dtheta.txt', '/dados/log_png_1003/', transform=transformation)
#    random_dataset, image_folder, images_path = get_data('/dados/rddf_predict/listen_2019-11-29_11:32:36', '/dados/log_png_1003/', arg_random)
#    my_dataset = RDDFPredictDataset(random_dataset, '/dados/log_png_1003/', transform=transformation)
#    print(data_target)
    ####################
    #   TRAIN
    ####################

    train_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1)
    device = torch.device("cuda:0")
    model.to(device)
   
    if(mode == 1):
        model.eval()
        if(chk_path != ""):
           data = torch.load(chk_path)
           model.load_state_dict(data)
        with torch.no_grad():
            for batch_i, batch  in enumerate(train_loader):
                img = batch['image'].to(device)
                target = batch['params'].to(device)
                preds_eval = model(img, scales, False)
                print("Eval: " ,preds_eval, "\n")
        exit()
        



    # Run fine-tuning (of modified class layers)   
    for p in model.body.parameters():
        p.requires_grad = False
    for q in model.head.parameters():
        q.requires_grad = False
    for q in model.out_vector.parameters():
        q.requires_grad = True

#    for m in model.modules(): 
#        if isinstance(m, InPlaceABN):
#            m.affine = False


    #no_epochs = 50
    LR = 1e-7
#    momentum = 0.98
    epochs = 2000

    model.train()
    # Am definitely training on the right parameters.

    #optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                    lr=LR,momentum = momentum) 

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                        lr=LR)   
    #n

    oname = 'Adam'
    
#    scales = eval(args.scales)
    lossfunction = nn.MSELoss().to(device)
    #pdb.set_trace()
    
    log_time = str(time.time()) 
#    logforloss = open('output_batch_train/lossfunction_'+ log_time +'.txt','a')
	
    # Create target Directory if don't exist
    dirName = "output_train_batch/"+log_time
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

    for epoch in range(epochs):

        #if epoch == 200:
        #    LR *= 0.1

        if epoch % 20 == 0:
            LR *= 0.1
 #       if epoch == 100:
 #          LR *= 0.1

        for batch_i, batch  in enumerate(train_loader):
            img = batch['image'].to(device)
            target = batch['params'].to(device)

            preds = model(img, scales, False)
            preds_eval = preds
            loss = lossfunction(preds.float(),target.float())

            log_output = "Desejado: "+str(target)+ "\nPrevisto: "+ str(preds) + "\n"
            print("Desejado: ",target , "\nPrevisto: " , preds)

            save_preds(log_time, preds, batch['img_timestamp'], dirName)

#            with torch.no_grad():
#                model.eval()
#                preds_eval = model(img, scales, False)
#                log_output += "Eval: "+str(preds_eval)+"\n"
#                print("Eval: " ,preds_eval, "\n")
#                model.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #torch.save(model.state_dict,'ckpoint_{}_{}.pt'.format(batch_i,epoch))
            logstring =  'Train Epoch: {} [/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch , len(my_dataset),
                    100. * batch_i / len(my_dataset), loss.item())
            log_output += logstring
            print(logstring)

            save_lossfunction(log_output, log_time, dirName)
            save_log(log_time, logstring, dirName)

            del preds, target, img, preds_eval
            # Overwrite salvando a cada 20 interacoes
            if(batch_i % 20 == 0):
                torch.save(model.state_dict(), dirName+'/ckpoint_{}_{}.pt'.format(log_time, chk_count))
                chk_count += 1

            if(chk_count >= 6):
                chk_count = 0


 #       torch.save(model.state_dict(),'weights_batch/ckpoint_{}_{}_{}.pt'.format(0, LR, log_time))
        # Overwrite checkpoint
#        torch.save(model.state_dict(),'weights_batch/ckpoint_{}.pt'.format(log_time))


 #   torch.save(model.state_dict(),'ckpoint_{}_{}_{}.pt'.format(epoch, LR, time.time()))
 #   logforloss.close()
    

def load_snapshot(snapshot_file):
    """Load a training snapshot"""
    print("--- Loading model from snapshot")

    # Create network
#    norm_act = partial(InPlaceABN, activation="leaky_relu", slope=.01)
    norm_act = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    body = models.__dict__["net_wider_resnet38_a2"](norm_act=norm_act, dilation=(1, 2, 4, 4))
    head = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    # Load snapshot and recover network state
    data = torch.load(snapshot_file)
    body.load_state_dict(data["state_dict"]["body"])
    head.load_state_dict(data["state_dict"]["head"])

    return body, head, data["state_dict"]["cls"]


if __name__ == "__main__":
    main()
