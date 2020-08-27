import cv2
import os
import glob
import numpy as np
import torch

from habitat.utils.visualizations.utils import observations_to_image
from aihabitat.rl.models.simple_cnn import SimpleCNN

print("OpenCV version : %s"%(cv2.__version__))

img_dir = "./data/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*.jpg')
files = glob.glob(data_path)


def get_frames(env, batch_size=30):
    rgb_frames = []
    observations = env.reset()

    frame = observations_to_image(observations[0],[])
    rgb_frames.append(frame)
    dones = [False]
    while dones[0] == False:
        outputs = env.step([env.action_spaces[0].sample()])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        frame = observations_to_image(observations[0], [])
        rgb_frames.append(frame)
    return rgb_frames


def train_cnn(env, inp_cnn, steps=100):

    tmp_cnn = SimpleCNN(env.observation_spaces[0], 512)
    tmp_cnn = tmp_cnn.cnn[:7]
    tmp_cnn[-2] = torch.nn.ConvTranspose2d(32, 1, (256,512), stride=1)
    tmp_cnn[-1] = torch.nn.Upsample(size=(256,512))
    optimizer = torch.optim.Adam(tmp_cnn.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for i in range(steps):
        print("Transfer learning step: %d"%i)
        labels = []
        img_frames = get_frames(env)
        for frame in img_frames:
            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
            edged = cv2.Canny(frame.astype(np.uint8), 50, 255)
            labels.append(edged)
        depth = np.zeros((len(img_frames), img_frames[0].shape[0], img_frames[0].shape[1]))
        depth = np.expand_dims(depth, axis=3)
        inp_frames = np.concatenate((np.array(img_frames), depth), axis=-1)
        inp_tensor = torch.Tensor(inp_frames)
        pred = tmp_cnn.forward(inp_tensor.permute(0, 3, 1, 2))

        pred = pred.reshape(-1,256,512)
        labels = np.array(labels)
        loss = loss_fn(pred, torch.Tensor(labels))
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    # inp_cnn[:5] = tmp_cnn[:5]
    print("Transfer Learning completed!!!!!!")
    return tmp_cnn[:5]
