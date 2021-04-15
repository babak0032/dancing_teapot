import argparse
import subprocess
import utils
import numpy as np
from PIL import Image
from math import *

def render_teapot(alpha,beta):
    FN = "teapot_small.obj"
    #FN = "entire_shape.obj"
    xs, ys = 128,128    # image size
    M = 15               # number of sampled points per face,
                        #     increase if there are black dots
    lx,ly,lz=1,-1,-1    # light source direction
    AMB = 70            # ambient light intensity
    LIGHT = 130         # direct light intensity
                        #   Note that AMB + LIGHT must be less than 256
    PFAC = 200          # perspective factor,
                        #     a higher value e.g. 1000 will produce less distortion

    im = Image.new("RGB", (xs, ys))
    zbuf = (xs) * (ys) * [1e6]      # Z buffer array

    s = 2       # object scaling factor
    xoff = 0    # X and Y
    yoff = 0    #         offsets

    # vertices/faces/normals lists
    V, F, N = [], [], []

    # read input file
    for l in open(FN):
        if l[0] == "v" and l[1] != "n":
            d, x, y, z = l.split()
            V.append([float(x)*0.3, float(y)*0.3, float(z)*0.3])
        if l[0] == "v" and l[1] == "n":
            d, x, y, z = l.split()
            N.append([float(x), float(y), float(z)])
        if l[0] == "f":
            d, x, y, z = l.split()
            x = x.split("/")[0]
            y = y.split("/")[0]
            z = z.split("/")[0]
            F.append([int(x), int(y), int(z)])

    print(f"{len(V)} vertices, {len(N)} normals, {len(F)} faces")

    w1 = sin(alpha) #w1 = 1 
    w2 = cos(alpha) #w2 = 0
    w3 = sin(beta)
    w4 = cos(beta)

    # transformed vertices/vertex normals and average Z distance for each face
    TV, TVN, FD = [], [], []

    # apply rotation to vertices
    for x,y,z in V:
        px = x*w1 + y*w2      
        py = x*w2*w4 - y*w1*w4 + z*w3
        pz = x*w3*w2 - y*w3*w1 - z*w4
        #px = x       
        #py = - y*w4 + z*w3
        #pz = - y*w3 - z*w4
        TV.append([px,py,pz])
        # plot each vertex:
        #im.putpixel((int(s*px+xs/2), int(s*py+ys/2)), (255,255,255))

    # apply rotation to vertex normals
    for x,y,z in N:
        px = x*w1 + y*w2
        py = x*w2*w4 - y*w1*w4 + z*w3
        pz = x*w3*w2 - y*w3*w1 - z*w4
        TVN.append([px,py,pz])

    # sort faces by depth
    for a,b,c in F:
        z1 = TV[a-1][2]
        z2 = TV[b-1][2]
        z3 = TV[c-1][2]
        za = (z1+z2+z3) / 3
        FD.append(za)

    FL = list(zip(FD, F))
    FL.sort(key = lambda x: -x[0])

    FLIST = FL
    FLEN = len(FLIST)
    ind = 0

    def vlen(x,y,z):
        return sqrt(x*x + y*y + z*z)

    # get light intensity based on surface normal and light direction
    def getcol(n1, n2, n3):
        ang = (lx*n1 + ly*n2 + lz*n3) / (vlen(lx,ly,lz) * vlen(n1,n2,n3))
        c = max(AMB, AMB + LIGHT * ang)
        return int(c)

    # loop over all faces
    for dd, ff in FLIST:
        ind += 1
        x1,y1,z1 = TV[ff[0]-1][0], TV[ff[0]-1][1], TV[ff[0]-1][2]
        x2,y2,z2 = TV[ff[1]-1][0], TV[ff[1]-1][1], TV[ff[1]-1][2]
        x3,y3,z3 = TV[ff[2]-1][0], TV[ff[2]-1][1], TV[ff[2]-1][2]
        dx1,dy1,dz1 = x2-x1,y2-y1,z2-z1
        dx2,dy2,dz2 = x3-x1,y3-y1,z3-z1
        n1, n2, n3 = TVN[ff[0]-1][0], TVN[ff[0]-1][1], TVN[ff[0]-1][2]
        c1 = getcol(n1, n2, n3)
        n1, n2, n3 = TVN[ff[1]-1][0], TVN[ff[1]-1][1], TVN[ff[1]-1][2]
        c2 = getcol(n1, n2, n3)
        n1, n2, n3 = TVN[ff[2]-1][0], TVN[ff[2]-1][1], TVN[ff[2]-1][2]
        c3 = getcol(n1, n2, n3)
        cd1 = c2 - c1
        cd2 = c3 - c1

        for l in range(M+1):
            xxa = x1 + dx1*l/M
            yya = y1 + dy1*l/M
            zza = z1 + dz1*l/M
            xxb = x1 + dx2*l/M
            yyb = y1 + dy2*l/M
            zzb = z1 + dz2*l/M
            ca = c1 + cd1*l/M
            cb = c1 + cd2*l/M
            for n in range(M+1):
                f1 = n/M
                f2 = 1 - f1
                xxx = f1 * xxa + f2 * xxb
                yyy = f1 * yya + f2 * yyb
                zzz = f1 * zza + f2 * zzb
                d = (PFAC - zzz) / PFAC     # apply some linear perspective
                c = int(f1*ca + f2*cb)
                xpos, ypos = int(xoff+s*d*xxx+xs/2), int(yoff+s*d*yyy+ys/2)
                if xpos < 0 or ypos < 0 or xpos > xs - 1 or ypos > ys - 1:
            	    continue
                zind = ypos * xs + xpos
                # use Z buffer to determine if a point is visible
                if zzz < zbuf[zind]:
                    zbuf[zind] = zzz
                    im.putpixel((xpos, ypos), (c,c,c))

    pix = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    return pix


def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_timesteps', type=int, default=10,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--fname', type=str, default='data/teapot.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    args = parser.parse_args()

    np.random.seed(args.seed)

    crop = None

    replay_buffer = {'obs': [],
                     'action': [],
                     'next_obs': [],
                     'state_ids': [],
                     'next_state_ids': []}
       
    alpha = np.pi / 2.0
    beta = 0
    rad_step = 2 * np.pi / 30.0

    state = render_teapot(alpha, beta) 

    for i in range(args.num_timesteps):

        replay_buffer['obs'].append(state)
        replay_buffer['state_ids'].append((alpha,beta))

        action = np.random.randint(4)
        deltas = [(rad_step,0),(0,rad_step),(-rad_step,0),(0,-rad_step)]
        replay_buffer['action'].append(action)
        alpha += deltas[action][0]
        beta += deltas[action][1]

        state = render_teapot(alpha, beta)
        replay_buffer['next_obs'].append(state)
        replay_buffer['next_state_ids'].append((alpha,beta))

#    if i % 10 == 0:
        print("iter "+str(i))


    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname)

