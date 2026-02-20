import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import ka_models
import utils
from spline_utils import set_spline_args
import math
# import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import h5py
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from datasets import build_dataset
import skimage.io
from skimage.measure import find_contours
# import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle

# from losscape.compute_loss import compute_loss
# from losscape.create_directions import create_random_direction, create_random_directions

#todo : plot anim la traj d'une optim avec PCA
#todo : losscape avec le test loss
#todo pour la lib : possiblité de tout foutre dans un fichier, et il fait les exps automatiquement ? (genre on met model + dataloader + optim + loss et il loop sur les models + optims)

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_weights(model):
    return [p.data for p in model.parameters()]

def _get_random_weights(weights):
    return [torch.randn(w.size()).to(device) for w in weights]

def make_directions(model, w1, w2):
    """
    Return two random directions in the model's weights space.
    These vectors are normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    w1, w2 : the directions.

    Returns
    ----------
    directions : list of two tensors, which correspond to the two sampled directions.


    Notes
    ----------
    Inspired from https://github.com/tomgoldstein/loss-landscape.

    """

    assert isinstance(w1, dict)
    assert isinstance(w2, dict)

    weights = _get_weights(model)

    x_direction = [p.to(device) for p in w1.values()]
    y_direction = [p.to(device) for p in w2.values()]

    _normalize_directions_for_weights(x_direction, weights)
    _normalize_directions_for_weights(y_direction, weights)

    return [x_direction, y_direction]

def create_random_directions(model):
    """
    Return two random directions in the model's weights space.
    These vectors are normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    model : the torch model whose weights will be used to create and normalize the directions.

    Returns
    ----------
    directions : list of two tensors, which correspond to the two sampled directions.


    Notes
    ----------
    Inspired from https://github.com/tomgoldstein/loss-landscape.

    """

    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)

    return [x_direction, y_direction]

def create_random_direction(model):
    """
    Return a random direction in the model's weights space.
    This vector is normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    model : the torch model whose weights will be used to create and normalize the direction

    Returns
    ----------
    direction : a tensor, which correspond to the sampled direction.


    Notes
    ----------
    Inspired from https://github.com/tomgoldstein/loss-landscape.

    """

    weights = _get_weights(model)
    direction = _get_random_weights(weights)
    _normalize_directions_for_weights(direction, weights)

    return direction

def _normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0) 
        d.mul_(w.norm() / (d.norm() + 1e-10))

#####################################################################################

def compute_loss(model, train_loader_unshuffled, get_batch, criterion = F.cross_entropy, num_batches:int = 8,closure = None):
    """
    Compute and return the loss over the first num_batches batches given by the train_loader_unshuffled, using the criterion provided.

    Parameters
    ----------
    model : the torch model which will be evaluated.
    train_loader_unshuffled : the torch dataloader. It is supposed to be fixed so that all the calls to this function will use the same data.
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    num_batches : number of batches to evaluate the model with. (default to 8)
    closure : An optional closure that can be passed in. This can be used if your model takes non-standard inputs or provides non-standard outputs.
    Returns
    ----------
    loss : loss computed

    """

    if criterion is None:
        criterion = F.cross_entropy

    loss = 0

    if train_loader_unshuffled is not None:
        with torch.no_grad():
            if closure is not None:
                loss, batch_idx = closure(train_loader_unshuffled,num_batches)
            else:
                for batch_idx, (Xb, Yb) in enumerate(train_loader_unshuffled):
                    Xb, Yb = Xb.to(device), Yb.to(device)
    
                    logits = model(Xb)
                    loss += criterion(logits, Yb).item()

                    if batch_idx + 1 >= num_batches:
                        break
    
        loss = loss / (batch_idx + 1)
    else:
        with torch.no_grad():
            for _ in range(num_batches):
                Xb, Yb = get_batch('train', 512)
                logits, l = model(Xb, Yb)
                loss += l.item()
        
        loss = loss / num_batches

    return loss

#simple, light weight and modular neural newtork loss landscape viz lib

# adv:
# -1D plot
# -torch.no_grad
# -easy plug and play

#####################################################################################

def create_2D_losscape(model, train_loader_unshuffled=None, get_batch=None, direction=None, criterion = None, closure = None, num_batches:int = 8, save_only:bool = False, output_path:str = '', x_min:float=-1., x_max:float=1., num_points:int=50):
    """
    Create a 2D losscape of the given model.

    Parameters
    ----------
    model : the torch model which will be used to create the losscape.
    train_loader_unshuffled : the torch dataloader. It is supposed to be fixed so that all the calls to this function will use the same data.
    optimizer : the optimizer used for training (should follow the same API as torch optimizers).(default to Adam)
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    closure : an optional closure that will replace the default compute_loss internals
    num_batches : number of batches to evaluate the model with. (default to 8)
    save_only : only save the plot and don't display it. (default to False)
    output_path : path where the plot will be saved. (default to '2d_losscape.png')
    x_min : min x value (that multiply the sampled direction). (default to -1.) 
    x_max : max x value (that multiply the sampled direction). (default to 1.)
    num_points : number of points to evaluate the loss, from x_min to x_max. (default to 50)

    Returns
    ----------
    coords : numpy array containing the x coords used to create the landscape
    losses : list of the losses computed

    """

    model.to(device)

    if direction is None:
        direction = [create_random_direction(model)]

    init_weights = [p.data for p in model.parameters()]

    coords = np.linspace(x_min, x_max, num_points)
    losses = []

    for x in coords:
        _set_weights(model, init_weights, direction, x)

        loss = compute_loss(model, train_loader_unshuffled, get_batch, criterion, num_batches, closure = closure)
        losses.append(loss)

    _reset_weights(model, init_weights)
    
    plt.plot(coords, losses)
    plt.savefig(os.path.join(output_path, '2d_losscape.png'), dpi=300)

    if not save_only:
        plt.show()
    
    plt.clf()

    return coords, losses

def create_3D_losscape(model, train_loader_unshuffled=None, get_batch=None, directions=None, criterion = None, closure = None, num_batches:int = 8, save_only:bool = False, output_path:str = '', output_vtp:bool = True, output_h5:bool = True, x_min:float=-1., x_max:float=1., y_min:float=-1., y_max:float=1., num_points:int=50):
    """
    Create a 3D losscape of the given model.

    Parameters
    ----------
    model : the torch model which will be used to create the losscape.
    train_loader_unshuffled : the torch dataloader. It is supposed to be fixed so that all the calls to this function will use the same data.
    optimizer : the optimizer used for training (should follow the same API as torch optimizers).(default to Adam)
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    closure : an optional closure that will replace the default compute_loss internals
    num_batches : number of batches to evaluate the model with. (default to 8)
    save_only : only save the plot and don't display it. (default to False)
    output_path : path where the plot will be saved. (default to '3d_losscape.png')
    output_vpt : whether or not to also create a .vtp file, used to 3D visualize the losscape. (default to False)
    output_h5 : whether or not to also create a .h5 file, containing the data generated by this function (default to True)
    x_min : min x value (that multiply the first sampled direction). (default to -1.) 
    x_max : max x value (that multiply the first sampled direction). (default to 1.)
    y_min : min x value (that multiply the second sampled direction). (default to -1.) 
    y_max : max x value (that multiply the second sampled direction). (default to 1.)
    num_points : number of points to evaluate the loss, from x_min to x_max and y_min to y_max. (default to 50)

    Returns
    ----------
    X : a (num_points, num_points) numpy array, the X meshgrid
    Y : a (num_points, num_points) numpy array, the Y meshgrid
    losses : a (num_points, num_points) numpy array containing all the losses computed

    Notes
    ----------
    The h5 files is structured as follows :

    """

    model.to(device)

    if directions is None:
        directions = create_random_directions(model)

    init_weights = [p.data for p in model.parameters()]

    X, Y = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))
    losses = np.empty_like(X)

    count = 0
    total = X.shape[0] * X.shape[1]

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            _set_weights(model, init_weights, directions, np.array([X[i, j], Y[i, j]]))

            loss = compute_loss(model, train_loader_unshuffled, get_batch, criterion, num_batches, closure = closure)
            losses[i, j] = loss

            count += 1
            print("LOSS FOR x={} AND y={} IS : {}. Done : {}/{} ({}%)".format(X[i, j], Y[i, j], loss, count, total, count/total*100.))

    _reset_weights(model, init_weights)

    cp = plt.contour(X, Y, losses, cmap='summer')
    plt.clabel(cp, inline=1, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(os.path.join(output_path, '3d_losscape.pdf'), bbox_inches='tight', dpi=300)
    
    if not save_only:
        plt.show()
    
    plt.clf()

    if output_vtp:
        _create_vtp(X, Y, losses, log=True, output_path=output_path)
        _create_vtp(X, Y, losses, log=False, output_path=output_path)

    if output_h5:
        with h5py.File(os.path.join(output_path, 'data.h5'), 'w') as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("Y", data=Y)
            hf.create_dataset("losses", data=losses)

    return X, Y, losses

def _set_weights(model, weights, directions, step):
    if len(directions) == 2:
        dx = directions[0]
        dy = directions[1]
        changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]

    else:
        changes = [d*step for d in directions[0]]

    for (p, w, d) in zip(model.parameters(), weights, changes):
        p.data = w + d

def _reset_weights(model, weights):
    for (p, w) in zip(model.parameters(), weights):
        p.data.copy_(w.type(type(p.data)))

# as in https://github.com/tomgoldstein/loss-landscape
def _create_vtp(X, Y, losses, log=False, zmax=-1, interp=-1, output_path=''):
    #set this to True to generate points
    show_points = False
    #set this to True to generate polygons
    show_polys = True

    xcoordinates = X
    ycoordinates = Y
    vals = losses

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    # Interpolate the resolution up to the desired amount
    if interp > 0:
        m = interpolate.interp2d(xcoordinates[0,:], ycoordinates[:,0], vals, kind='cubic')
        x_array = np.linspace(min(x_array), max(x_array), interp)
        y_array = np.linspace(min(y_array), max(y_array), interp)
        z_array = m(x_array, y_array).ravel()

        x_array, y_array = np.meshgrid(x_array, y_array)
        x_array = x_array.ravel()
        y_array = y_array.ravel()

    vtp_file = os.path.join(output_path, 'losscape')
    if zmax > 0:
        z_array[z_array > zmax] = zmax
        vtp_file +=  "_zmax=" + str(zmax)

    if log:
        z_array = np.log(z_array + 0.1)
        vtp_file +=  "_log"
    vtp_file +=  ".vtp"
    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (z_array[temp_index] + z_array[temp_index + 1] +
                                z_array[temp_index + matrix_size]  +
                                z_array[temp_index + matrix_size + 1]) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_file, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('  <PolyData>\n')

    if (show_points and show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_polys))
    else:
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(number_points, number_points))

    # <PointData>
    output_file.write('      <PointData>\n')
    output_file.write('        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) == 0:
            output_file.write('          ')
        output_file.write('{}'.format(z_array[vertexcount]))
        if (vertexcount % 6) == 5:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 6) != 5:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </PointData>\n')

    # <CellData>
    output_file.write('      <CellData>\n')
    if (show_polys and not show_points):
        output_file.write('        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
        output_file.write('        </DataArray>\n')
    output_file.write('      </CellData>\n')

    # <Points>
    output_file.write('      <Points>\n')
    output_file.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) == 0:
            output_file.write('          ')
        output_file.write('{} {} {}'.format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) == 1:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 2) != 1:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Points>\n')

    # <Verts>
    output_file.write('      <Verts>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_points - 1))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_points))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount + 1))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Verts>\n')

    # <Lines>
    output_file.write('      <Lines>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Lines>\n')

    # <Strips>
    output_file.write('      <Strips>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Strips>\n')

    # <Polys>
    output_file.write('      <Polys>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) == 0:
                    output_file.write('          ')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1), (temp_index + matrix_size)))
                if (polycount % 2) == 1:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
                polycount += 1
        if (polycount % 2) == 1:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (polycount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Polys>\n')

    output_file.write('    </Piece>\n')
    output_file.write('  </PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    print("Done with file:{}".format(vtp_file))

##################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--ckpt', default=None, help='Load checkpoint')
    # parser.add_argument('--arch', default='vit_small', type=str,
    #     choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--pretrained_weights', default='', type=str,
    #     help="Path to pretrained weights to load.")
    # parser.add_argument("--checkpoint_key", default="teacher", type=str,
    #     help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    # parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    # parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    # parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
    #     obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='vit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
    #                     help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    # parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "cosine"')
    # parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    # parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    # parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
    #                     help='epoch interval to decay LR')
    # parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    # parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    # parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    # parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    # parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #########################################
    # Kolmogorov Arnold Attention Parameters
    parser.add_argument('--grid', default=10, type=int, help='Number of grids for the Spline')
    parser.add_argument('--order', default=10, type=int, help='Order of the Spline')
    parser.add_argument('--grid_range', default=[-4,4], type=int, nargs=2, help='Range of grids for the Spline')
    parser.add_argument('--base_fun', default='nn.SiLU', type=str, help='Base Function Custom: ZeroModule')
    parser.add_argument('--spline_type', default='BasisSpline', type=str, choices=['BasisSpline', 'BasisSpline_Eff', 'FourierSpline'], help='Type of Spline Function to use')
    parser.add_argument('--depth', default=2, type=int, help='Depth/No. of Layers')
    parser.add_argument('--hidden_dim', default=32, type=int, help='Hidden Dimension to use between input/output')
    parser.add_argument('--num_attention', default=None, type=int, help='Number of KA Attention for MixedHeads')
    parser.add_argument('--mixed_head', action='store_true', default=False, help='Enable Mixed Heads')
    parser.add_argument('--uni_head', action='store_true', default=False, help='Enable Single KAN Activation for each heads throughout the network')
    parser.add_argument('--hybrid_act', action='store_true', default=False, help='Enable MLP+KAN layered activation')
    parser.add_argument('--hybrid_mode', default='w1_phi2', type=str, choices=['w1_phi2', 'phi1_w2'], help='MLP+KAN Mode')
    parser.add_argument('--project_l1', action='store_true', default=False, help='Enable L1 Ball projection')
    parser.add_argument('--sp_not_trainable', action='store_false', default=True, help='Disable Spline Training')
    parser.add_argument('--sb_not_trainable', action='store_false', default=True, help='Disable Base Function Training')
    parser.add_argument('--alt_lr', type=float, default=None, help='Use a different LR for KAN Layers')
    #########################################
    # WandB Parameters
    # parser.add_argument('--run_name', type=str, help='Name of Run to log with WandB')
    parser.add_argument('--direction_files', type=str, nargs=2, help='Name of weight files to be used for directions')
    parser.add_argument('--grad_file', type=str, help='Name of grad path files to be used for directions')
    
    args = parser.parse_args()
    set_spline_args(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # if args.data_set == 'CIFAR10':
    #     args.nb_classes = 10
    # elif args.data_set == 'CIFAR100':
    #     args.nb_classes = 100
    # elif args.data_set == 'IMNET':
    #     args.nb_classes = 1000
    # else:
    #     raise NotImplementedError('Only CIFAR10, CIFAR100, ILSVRC are implemented')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # build model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    if args.ckpt:
        if args.ckpt.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.ckpt, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)

    dirX = torch.load(args.direction_files[0])
    dirY = torch.load(args.direction_files[1])

    directions = make_directions(model=model, w1=dirX, w2=dirY)
    X, Y, losses = create_3D_losscape(model, train_loader_unshuffled=data_loader_val, directions=directions, num_batches=8, output_path=args.output_dir, num_points=100, save_only=True)

    with open(os.path.join(args.output_dir, 'data.pkl'), 'wb') as fpkl:
        pickle.dump({'X':X, 'Y':Y, 'losses':losses}, fpkl)

    grad_path = torch.load(args.grad_file, map_location='cpu')['FinalXY']
    # grad_path = torch.cat((grad_path, torch.load(args.grad_file, map_location='cpu')['opt_XY']), dim=-1)

    cp = plt.contour(X, Y, losses, cmap='summer')

    plt.plot(grad_path[0].cpu().numpy(), grad_path[1].cpu().numpy(), linestyle='-', marker='o', color='#377eb8')

    plt.clabel(cp, inline=True, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(args.output_dir, '3d_losscape_grad_path.pdf'), bbox_inches='tight', dpi=300)