"""
@author: <nktoan163@gmail.com>
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
from model import CNN
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser('Pesplanus training test script')
    parser.add_argument('-s', '--size', type= int, default=224, help='image size (default:640)')
    parser.add_argument('-i', '--root_path', type= str, default='Pesplanus.v2i.multiclass/test', help='the root folder of test images')
    parser.add_argument('-c', '--checkpoint_path', type= str, default='trained_model/best.pt', help='the')
    args= parser.parse_args()
    return args

def test(args):
    classes = [' Pesplanus', ' Notpresplanus']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(classes)).to(device)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print('no checkpoint')
        exit(0)
    if not args.root_path:
        print('no root path')
        exit(0)

    images_paths = []
    csv_path = os.path.join(args.root_path, '_classes.csv')
    classes_df = pd.read_csv(csv_path)
    for index, row in classes_df.iterrows():
        image_name = row['filename']
        image_paths = os.path.join(args.root_path, image_name)
        images_paths.append(image_paths)

    if not os.path.exists("probability_results"):
        os.makedirs("probability_results")

    for image_path in images_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (args.size, args.size))
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device).float()

        softmax = nn.Softmax()
        with torch.no_grad():
            prediction = model(image)

        probs = softmax(prediction)
        max_value, max_index = torch.max(probs, dim=1)
        print("This image is about {} with a probability of {:.4f}".format(classes[max_index], max_value[0].item()))

        plt.figure(figsize=(10, 6))  # Increase figure size
        plt.bar(classes, probs[0].cpu().numpy(), color='skyblue')
        plt.xlabel("Class", fontsize=14)
        plt.ylabel("Probability", fontsize=14)
        plt.title(classes[max_index], fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join("probability_results", f"result_{os.path.basename(image_path)}.png"))
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    test(args)