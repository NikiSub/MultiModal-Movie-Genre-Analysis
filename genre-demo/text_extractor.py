import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
sys.path.insert(0, '../CRAFT-pytorch')
from craft import CRAFT
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils

sys.path.insert(0, '../deep-text-recognition-benchmark')
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate

class TextExtractor():
    def __init__(self, image_folder, extract_text_file,split):
        self.i_folder = image_folder
        #print(image_folder)
        #print("aaaaaaa test")
        self.extract_text_file = extract_text_file
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.show_time = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cuda = torch.cuda.is_available()
        self.net = CRAFT() #(1st model) model to detect words in images
        if self.cuda:
            self.net.load_state_dict(self.copyStateDict(torch.load('../CRAFT-pytorch/craft_mlt_25k.pth')))
        else:
            self.net.load_state_dict(self.copyStateDict(torch.load('../CRAFT-pytorch/craft_mlt_25k.pth', map_location='cpu')))
        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        self.net.eval()
        self.refine_net = None
        
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.poly = False

        self.result_folder = './'+split+'_'+'intermediate_result/'


        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)

        #Parameters for image to text model (2nd model)
        self.parser = argparse.ArgumentParser()
        #Data processing
        self.parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        self.parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        self.parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        self.parser.add_argument('--rgb', default=False, action='store_true', help='use rgb input')
        self.parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        self.parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        self.parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        #Model Architecture
        self.parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
        self.parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
        self.parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
        self.parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
        self.parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        self.parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        self.parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
        self.parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
        #self.opt = self.parser.parse_args()
        self.opt, unknown = self.parser.parse_known_args()
        #self.opt, unknown = self.parser.parse_known_args()

        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)
        #print(opt.rgb)
        if self.opt.rgb:
            self.opt.input_channel = 3
        self.opt.num_gpu = torch.cuda.device_count()
        self.opt.batch_size = 192
        #self.opt.batch_size = 3
        self.opt.workers = 0
        self.model = Model(self.opt) #image to text model (2nd model)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.load_state_dict(torch.load('../deep-text-recognition-benchmark/TPS-ResNet-BiLSTM-Attn.pth', map_location=self.device))
        self.model.eval()
    def copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    def test_net(self, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text
    def extract_text(self):
        l = sorted(os.listdir(self.i_folder))
        img_to_index = {}
        count = 0
        for full_file in l:
            split_file = full_file.split(".")
            filename = split_file[0] 
            img_to_index[count] = filename
            #print(count, filename)
            count+=1
            #print(filename)
            file_extension = "."+split_file[1]
            #print(filename, file_extension)
            image = imgproc.loadImage(self.i_folder+full_file)
            bboxes, polys, score_text = self.test_net(self.net, image, self.text_threshold, self.link_threshold, self.low_text, self.cuda, self.poly, self.refine_net)
            img=cv2.imread(self.i_folder+filename+file_extension)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            points = []
            order = []
            for i in range(0,len(bboxes)):
                sample_bbox = bboxes[i]
                min_point = sample_bbox[0]
                max_point = sample_bbox[2]
                for j,p in enumerate(sample_bbox):
                    if(p[0]<=min_point[0]):
                        min_point = (p[0],min_point[1])
                    if(p[1]<=min_point[1]):
                        min_point = (min_point[0],p[1])
                    if(p[0]>=max_point[0]):
                        max_point = (p[0],max_point[1])
                    if(p[1]>=max_point[1]):
                        max_point = (max_point[0],p[1])
                min_point = (max(min(len(rgb_img[0]),min_point[0]),0), max(min(len(rgb_img),min_point[1]),0))
                max_point = (max(min(len(rgb_img[0]),max_point[0]),0), max(min(len(rgb_img),max_point[1]),0))
                points.append((min_point, max_point))
                order.append(0)
            num_ordered = 0
            rows_ordered = 0
            points_sorted =[]
            ordered_points_index = 0
            order_sorted=[]
            while(num_ordered<len(points)):
                #find lowest-y that is unordered
                min_y = len(rgb_img)
                min_y_index = -1
                for i in range(0,len(points)):
                    if(order[i]==0):
                        if(points[i][0][1]<=min_y):
                            min_y = points[i][0][1]
                            min_y_index = i
                rows_ordered+=1
                order[min_y_index] = rows_ordered
                num_ordered+=1
                points_sorted.append(points[min_y_index])
                order_sorted.append(rows_ordered)
                ordered_points_index = len(points_sorted)-1
                
                # Group bboxes that are on the same row
                max_y = points[min_y_index][1][1]
                range_y = max_y-min_y
                for i in range(0,len(points)):
                    if(order[i]==0):
                        min_y_i = points[i][0][1]
                        max_y_i = points[i][1][1]
                        range_y_i = max_y_i-min_y_i
                        if(max_y_i>=min_y and min_y_i<=max_y):
                            overlap = (min(max_y_i,max_y)-max(min_y_i,min_y))/(max(1,min(range_y,range_y_i)))
                            if(overlap>=0.30):
                                order[i] = rows_ordered
                                num_ordered+=1
                                min_x_i = points[i][0][0]
                                for j in range(ordered_points_index, len(points_sorted)+1):
                                    if(j<len(points_sorted)): #insert before
                                        min_x_j = points_sorted[j][0][0]
                                        if(min_x_i<min_x_j):
                                            points_sorted.insert(j, points[i])
                                            order_sorted.insert(j, rows_ordered)
                                            break
                                    else: #insert at the end of array
                                        points_sorted.insert(j, points[i])
                                        order_sorted.insert(j, rows_ordered)
                                        break
            for i in range(0,len(points_sorted)):
                min_point = points_sorted[i][0]
                max_point = points_sorted[i][1]
                mask_file = self.result_folder + filename+"_" + str(order_sorted[i])+"_"+str(i) + file_extension
                crop_image = rgb_img[int(min_point[1]):int(max_point[1]),int(min_point[0]):int(max_point[0])]
                #print(filename, min_point, max_point, len(rgb_img), len(rgb_img[0]))
                cv2.imwrite(mask_file, crop_image)
        AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        demo_data = RawDataset(root=self.result_folder, opt=self.opt)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
        f = open(self.extract_text_file, "w")
        count = -1
        curr_order = 1
        curr_filename = ""
        output_string = ""
        end_line = "[SEP] "
        #print("image to index")
        #print(img_to_index)
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                #image = (torch.from_numpy(crop_image).unsqueeze(0)).to(device)
                #print(image_path_list)
                #print(image.size())
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)
                preds = self.model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
                for path,p in zip(image_path_list, preds_str):
                    #print(path)
                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = p.find('[s]')
                        p = p[:pred_EOS]  # prune after "end of sentence" token ([s])
                    path_info = path[len(self.result_folder):].split(".")[0].split("_") #ASSUMES FILE EXTENSION OF SIZE 4 (.PNG, .JPG, ETC)
                    #print(curr_filename)
                    #print(path_info[0])
                    #print("PATHINFO: ",path_info[0])
                    #print("CURRFILE: ", curr_filename)
                    if(not (curr_filename==path_info[0])):
                        if(not (curr_filename=="")):
                            f.write(str(count)+"\n")
                            f.write(curr_filename+"\n")
                            f.write(output_string+"\n\n")
                        count+=1
                        curr_filename = img_to_index[count]#path_info[0]
                        #print("CURRFILE: ", curr_filename)
                        while(not (curr_filename==path_info[0])):
                            f.write(str(count)+"\n")
                            f.write(curr_filename+"\n")
                            f.write("\n\n")
                            count+=1
                            curr_filename = img_to_index[count]#path_info[0]
                            #print("CURRFILE: ", curr_filename)
                        output_string = ""
                        curr_order = 1
                    if(int(path_info[1])>curr_order):
                        curr_order+=1
                        output_string+=end_line
                    output_string+=p+" "
            f.write(str(count)+"\n")
            f.write(curr_filename+"\n")
            f.write(output_string+"\n\n")
        f.close()

        #Go through each image in the i_folder and crop out text
        
        #generate text and write to text file

    def get_item(self, index):
        f = open(self.extract_text_file, "r")
        Lines = f.readlines()
        return (Lines[4*index+2][:-1])
        # read text file

#TEST
#t_e = TextExtractor("data/mmimdb-256/dataset-resized-256max/dev_n/images/","text_extract_output.txt")
#t_e.extract_text()
#text = t_e.get_item(1)
#print(text)