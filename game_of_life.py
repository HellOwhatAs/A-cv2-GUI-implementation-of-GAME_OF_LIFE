import os
import cv2,sys,json
import numpy as np
from numba import njit
@njit
def _MouseEvent1(sight_point,sight,x,y):
    sight_point[0]=sight*(sight_point[0]+y)/(sight+50)-y
    sight_point[1]=sight*(sight_point[1]+x)/(sight+50)-x
    sight_point[0]=max(0,min(sight_point[0],sight-640))
    sight_point[1]=max(0,min(sight_point[1],sight-640))
    return sight_point
@njit
def _MouseEvent2(sight_point,sight,x,y):
    sight_point[0]=sight*(sight_point[0]+y)/(sight-50)-y
    sight_point[1]=sight*(sight_point[1]+x)/(sight-50)-x
    sight_point[0]=max(0,min(sight_point[0],sight-640))
    sight_point[1]=max(0,min(sight_point[1],sight-640))
    return sight_point
def MouseEvent(event, x, y, flags, param):
    global world,sight,sight_point,tmp_signp,pressing
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags<0:
            if sight>640:
                sight-=50
                sight_point=_MouseEvent1(sight_point,sight,x,y)
            else:return
        else:
            sight+=50
            sight_point=_MouseEvent2(sight_point,sight,x,y)
    if pressing:
        sight_point[0],sight_point[1]=min(max(0,tmp_signp[0]-y),sight-640),min(max(0,tmp_signp[1]-x),sight-640)
    if event == cv2.EVENT_RBUTTONDOWN:
        tmp_signp[0],tmp_signp[1]=y+sight_point[0],x+sight_point[1]
        pressing=1
    elif event == cv2.EVENT_RBUTTONUP:
        tmp_signp[0],tmp_signp[1]=0,0
        pressing=0
    if mode:return
    if event == cv2.EVENT_LBUTTONDOWN:
        ptmp=((y+sight_point[0])*l//(sight)),((x+sight_point[1])*l//(sight))
        world[ptmp]=not world[ptmp]

@njit
def add_grid(img,l,alpha):
    for i in range(l):
        img[(i*img.shape[0])//l]=alpha
        img[:,(i*img.shape[0])//l]=alpha
    return img
@njit
def update_world(world):
    new_world=np.copy(world)
    s0,s1=world.shape
    for i in range(s0):
        for j in range(s1):
            tmp=round(np.sum(world[i-1:i+2,j-1:j+2])-world[i,j])
            if tmp==3:
                new_world[i,j]=1
            elif tmp<2 or tmp>3:
                new_world[i,j]=0
    return new_world
if __name__=="__main__":
    if len(sys.argv)>1:
        if os.path.isfile(sys.argv[1]):
            with open(sys.argv[1],"rb") as f:
                world=np.array(json.load(f),dtype=np.bool8)
                l1,l2=world.shape
                if l1>l2:
                    world=world[:l2]
                    l=l2
                elif l1<l2:
                    world=world[:,:l1]
                    l=l1
                else:
                    l=l1
                del l1,l2
        else:
            if sys.argv[1].isnumeric():
                l=int(sys.argv[1])
                world=np.zeros((l,l),dtype=np.bool8)
    else:
        l=200
        world=np.zeros((l,l),dtype=np.bool8)

    
    cv2.namedWindow('game of life')
    cv2.setMouseCallback('game of life', MouseEvent)

    sight=640
    sight_point=np.array([0,0],dtype=np.int64)
    tmp_signp=np.array([0,0],dtype=np.int64)
    pressing=0
    fps=50
    mode=False
    while True:
        flag=1
        while True:
            tmp=cv2.resize(world.astype(np.double),(sight,sight),interpolation=cv2.INTER_NEAREST)
            if mode:
                cv2.imshow('game of life', add_grid(tmp,l,0.1)[sight_point[0]:sight_point[0]+640,sight_point[1]:sight_point[1]+640])
                k=cv2.waitKey(fps)
                if k==61 and fps>2:
                    fps-=1
                elif k==45:
                    fps+=1
            else:
                cv2.imshow('game of life', add_grid(tmp,l,0.2)[sight_point[0]:sight_point[0]+640,sight_point[1]:sight_point[1]+640])
                k=cv2.waitKey(100)
            if k==27 or cv2.getWindowProperty('game of life', cv2.WND_PROP_VISIBLE) < 1.0:
                flag=0
                break
            if k==32:break
            if k==115:
                with open("_-_--__---___tmpfile.json","w") as f:
                    json.dump(world.astype(np.int16).tolist(),f)
            if mode:world=update_world(world)
        if not flag:break
        mode=not mode
    cv2.destroyAllWindows()