import torch
import torch.nn as nn
import sys
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from util.utils import get_val_data, perform_val
from IPython import embed
import sklearn
import cv2
import numpy as np
from image_iter import FaceDataset
import torch.utils.data as data
import argparse
import os
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import torch.nn.functional as F

if __name__ == '__main__':

    network = "VIT"
    NUM_CLASS = 93431
    w,h =112,112
    device = torch.device("cuda:0") # 디바이스 설정
    model_root = '/home/leo/Desktop/Face-Transformer/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'


    if network == 'VIT' :
        model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type='ArcFace',
            GPU_ID= device,
            num_class=NUM_CLASS,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif network == 'VITs':
        model = ViTs_face(
            loss_type='ArcFace',
            GPU_ID=device,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    model.load_state_dict(torch.load(model_root))
    model.to(device)
    model.eval()


    #debug
    w = torch.load(model_root)
    for x in w.keys():
        print(x, w[x].shape)

    submission = pd.read_csv("/home/leo/Desktop/inha_challenge/inha_data/sample_submission.csv")

    left_test_paths = list()
    right_test_paths = list()

    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])

    # 이미지 데이터 전처리 정의

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Left Side Image Processing

    left_test = list()
    for left_test_path in left_test_paths:
        img = Image.open("/home/leo/Desktop/inha_challenge/inha_data/test/" + left_test_path + '.jpg').convert("RGB")# 경로 설정 유의(ex .inha/test)
        img = data_transform(img) # 이미지 데이터 전처리
        left_test.append(img) 
    left_test = torch.stack(left_test)
    #print(left_test.size()) # torch.Size([6000, 3, 112, 112])

    left_infer_result_list = list()
    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 500
        for i in range(0, 12):
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            left_infer_result = model(tmp_left_input.to(device))
            #print(left_infer_result.size()) # torch.Size([1000, 512])
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)
        #print(left_infer_result_list.size()) # torch.Size([6000, 512])


    # Right Side Image Processing

    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open("/home/leo/Desktop/inha_challenge/inha_data/test/" + right_test_path + '.jpg').convert("RGB") # 경로 설정 유의 (ex. inha/test)
        img = data_transform(img)# 이미지 데이터 전처리
        right_test.append(img)
    right_test = torch.stack(right_test)
    #print(right_test.size()) # torch.Size([6000, 3, 112, 112])

    right_infer_result_list = list()
    with torch.no_grad():
        '''
        메모리 부족으로 6,000개 (배치) 한번에 입력으로 넣지 않고 1,000개 씩 입력으로 줌
        '''
        batch_size = 500
        for i in range(0, 12):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            #print(tmp_input.size()) # torch.Size([1000, 3, 112, 112])
            right_infer_result = model(tmp_right_input.to(device))
            #print(left_infer_result.size()) # torch.Size([1000, 512])
            right_infer_result_list.append(right_infer_result)

        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)
        #print(right_infer_result_list.size()) # torch.Size([6000, 512])


    def cos_sim(a, b):
        return F.cosine_similarity(a, b)

    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)


    submission = pd.read_csv("/home/leo/Desktop/inha_challenge/inha_data/sample_submission.csv") 
    submission['answer'] = cosin_similarity.tolist()
    #submission.loc['answer'] = submission['answer']
    submission.to_csv('/home/leo/Desktop/Face-Transformer/VIT_arc_submission.csv', index=False)




