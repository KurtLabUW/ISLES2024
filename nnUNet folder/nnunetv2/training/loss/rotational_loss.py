
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from monai.losses.dice import DiceLoss as DiceLossMONAI


class Rotational_loss(nn.Module):
    def __init__(self, weight=None, size_average=True, criterion=nn.MSELoss()):
        super(Rotational_loss, self).__init__()
        self.criterion = criterion

    def centroid(self, output, x, y, z):
        # device = "cuda:1"
        output.to(device)
        output = torch.abs(torch.round(torch.tanh(output))).to(device)
        total_mass = torch.sum(output, dim=(1, 2, 3)).to(device)   
        # print(f' output size = {output[0].size()} ')
        wZ = output * z.to(device)
        wX = output * x.to(device)
        wY = output * y.to(device)
        # print(f' x = {x}')
        # print(f' y = {y}')
        # print(f' z = {z}')
        
        sz = torch.sum(wZ, dim=(1, 2, 3))
        sy = torch.sum(wY, dim=(1, 2, 3))
        sx = torch.sum(wX, dim=(1, 2, 3))
        # Centroids
        z_centroid = sz / (total_mass + torch.rand(1).item() * (1e-7 - 1e-8) + 1e-8) # Shape: (batch_size, 1, 1, 1) divided by (batch_size, 1, 1, 1)
        y_centroid = sy / (total_mass + torch.rand(1).item() * (1e-7 - 1e-8) + 1e-8)
        x_centroid = sx / (total_mass + torch.rand(1).item() * (1e-7 - 1e-8) + 1e-8)
        # print(x_centroid)
        # print(y_centroid)
        return total_mass, torch.mean(x_centroid), torch.mean(y_centroid), torch.mean(z_centroid)
    
    def rotational_loss(self, output, x, y, z):
        # device = "cuda:1" 
        output.to(device)
        mass, cx, cy, cz = self.centroid(output, x, y, z)
        # Shift coordinates by the centroid
        z_shifted = z.to(device) - cz
        y_shifted = y.to(device) - cy
        x_shifted = x.to(device) - cx
        # print(f'z_shifted = {z_shifted}')    
        distances_squared_z = y_shifted**2 + x_shifted**2
        # print(f' z_squared val = {(distances_squared_z)}')
        distances_squared_y = z_shifted**2 + x_shifted**2
        distances_squared_x = z_shifted**2 + y_shifted**2
        # Compute the rotational inertia
        inertia_z = torch.sum((output.to(device)) * distances_squared_z.to(device))
        # print(f' z_inertia = {(inertia_z)}')
        inertia_x = torch.sum((output.to(device)) * distances_squared_x.to(device))
        inertia_y = torch.sum((output.to(device)) * distances_squared_y.to(device))
        inertia_fin = inertia_x + inertia_x + inertia_z
        # print(inertia_y)
       # Calculate the bounding box dimensions
        max_distance = ((output.shape[1] ** 2) + (output.shape[2] ** 2) + (output.shape[3] ** 2))**(0.5)
        # Normalize by maximum possible rotational inertia
        max_ri = (max_distance ** 2)
        cat = torch.cat((inertia_z.unsqueeze(0), inertia_x.unsqueeze(0), inertia_y.unsqueeze(0))).to(device)
        # iri1 = (mass * max_ri) / (2 * torch.pi * torch.mean(torch.cat((inertia_z.unsqueeze(0), inertia_x.unsqueeze(0), inertia_y.unsqueeze(0)))) + 1e-8)
        # iri2 = (mass * max_ri) / (2 * torch.pi * inertia_fin + 1e-8)
        # iri3 = (((mass ** 2) / (2 * torch.pi * torch.sum(cat) + 1e-8)) / max_ri)
        # iri = (mass ** 2) / (2 * torch.pi * torch.sum(cat) + 1e-9)
        print(f' cat: {cat}')
        print(f' Inverse rotational intertia output = {((mass ** 2) / (2 * torch.pi * torch.sum(cat) + 1e-8)) / max_ri}')
        print(f' Mass/2pi output = {(mass) / (2 * torch.pi)}')
        print(f' Mass = {(mass)}')
        print(f'iri: {iri}')
        return  (mass ** 2)  / (2 * torch.pi * torch.sum(cat) + 1e-8) 
    
    def forward(self, ya: torch.Tensor, yp: torch.Tensor):
        # device = "cuda:1"
        x, y, z = torch.meshgrid(torch.arange(128, requires_grad=True, dtype=torch.float32), 
                        torch.arange(192, requires_grad=True, dtype=torch.float32), 
                        torch.arange(128, requires_grad=True,  dtype=torch.float32))
        actual = self.rotational_loss(ya, x, y, z)
        predicted = self.rotational_loss(yp, x, y, z)
        print(f'actual rotational loss = {actual}')
        print(f'pred rotational loss = {predicted}')
        rotIn_error = self.criterion(actual, predicted)
        return rotIn_error