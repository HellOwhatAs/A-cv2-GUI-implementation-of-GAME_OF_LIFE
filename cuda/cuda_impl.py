from numba import cuda
import numba as nb,numpy as np,cv2

wname="CUDA-gof"

l=4000
_t_=32
view_size=100
x=y=(l-view_size)//2

pattern=(np.random.random((l,l))>0.95).astype(np.int8)

world=cuda.to_device(pattern)
world_next=cuda.device_array(world.shape,world.dtype)

@cuda.jit
def update_new(world,new_world):
    i,j=cuda.grid(2)
    if 0<i<world.shape[0]-1 and 0<j<world.shape[1]-1:
        tmp=(world[i-1,j-1])+\
        (world[i-1,j])+\
        (world[i-1,j+1])+\
        (world[i,j-1])+\
        (world[i,j+1])+\
        (world[i+1,j-1])+\
        (world[i+1,j])+\
        (world[i+1,j+1])

        if tmp==3:new_world[i,j]=1
        elif tmp<2 or tmp>3:new_world[i,j]=0

@nb.jit((nb.int64, nb.int64, nb.int64, nb.int64),nopython=True)
def ud(k,x,y,view_size):
    delta_=max(round(view_size/30),1)
    if k==ord("a"):
        if y>0:
            y-=delta_
            y=min(max(y,0),l-1)
    elif k==ord("d"):
        if y+view_size<l-1:
            y+=delta_
            y=min(max(y,0),l-1)
    elif k==ord("w"):
        if x>0:
            x-=delta_
            x=min(max(x,0),l-1)
    elif k==ord("s"):
        if x+view_size<l-1:
            x+=delta_
            x=min(max(x,0),l-1)
    elif k==ord('='):
        _=min(max(round(view_size*9/10),10),l)
        delta=view_size-_
        view_size=_
        x=min(max(x+round(delta/2),0),l-view_size)
        y=min(max(y+round(delta/2),0),l-view_size)
    elif k==ord('-'):
        _=min(max(round(view_size*10/9),10),l)
        delta=_-view_size
        view_size=_
        x=min(max(x-round(delta/2),0),l-view_size)
        y=min(max(y-round(delta/2),0),l-view_size)
    return x,y,view_size


while True:
    update_new[(l//_t_+1,l//_t_+1),(_t_,_t_)](world,world_next)
    world.copy_to_device(world_next)
    cv2.imshow(wname,cv2.resize(world[x:x+view_size,y:y+view_size].copy_to_host().astype(np.double),(600,600),interpolation=cv2.INTER_NEAREST))
    if (k:=cv2.waitKey(1))!=-1:(x,y,view_size)=ud(k,x,y,view_size)
    if cv2.getWindowProperty(wname,cv2.WND_PROP_VISIBLE)<1:break