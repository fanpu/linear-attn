"""Train a neural cellular automaton to grow and repair a target organism."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).parent; OUT=ROOT/"outputs"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, SIZE, CHANNELS=211, 64, 16


def target_image(size=SIZE):
    a=torch.linspace(-1,1,size,device=DEVICE); y,x=torch.meshgrid(a,a,indexing="ij")
    r=torch.sqrt(x*x+y*y); theta=torch.atan2(y,x)
    petal=.38+.105*torch.cos(6*theta)
    alpha=torch.sigmoid((petal-r)*27)
    center=torch.exp(-(r/.18)**2)
    hue=(theta+torch.pi)/(2*torch.pi)
    red=.32+.55*torch.sin(2*torch.pi*hue+1.1).square()
    green=.22+.56*torch.sin(2*torch.pi*hue+3.2).square()
    blue=.30+.45*torch.sin(2*torch.pi*hue+5.0).square()
    rgb=torch.stack((red,green,blue))*alpha + center*torch.tensor([.75,.53,.12],device=DEVICE)[:,None,None]
    return torch.cat((rgb.clamp(0,1),alpha[None]),0)


class NeuralCA(nn.Module):
    def __init__(self):
        super().__init__()
        identity=torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
        sobel_x=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8
        sobel_y=sobel_x.T
        kernel=torch.stack((identity,sobel_x,sobel_y))[:,None]
        self.register_buffer("kernel",kernel.repeat(CHANNELS,1,1,1))
        self.update1=nn.Conv2d(CHANNELS*3,128,1)
        self.update2=nn.Conv2d(128,CHANNELS,1,bias=False)
        nn.init.zeros_(self.update2.weight)
    def forward(self,x,fire_rate=.5):
        sensed=F.conv2d(x,self.kernel,padding=1,groups=CHANNELS)
        delta=self.update2(F.relu(self.update1(sensed)))
        fire=(torch.rand_like(delta[:,:1])<=fire_rate).float()
        return x+delta*fire


def seed(batch):
    x=torch.zeros(batch,CHANNELS,SIZE,SIZE,device=DEVICE); x[:,3,SIZE//2,SIZE//2]=1.; return x


def damage(x):
    # Deletion is a real perturbation of the current automaton state, including hidden channels.
    batch=x.shape[0]
    for i in range(batch):
        cx=int(torch.randint(SIZE//4,3*SIZE//4,(1,),device=DEVICE)); cy=int(torch.randint(SIZE//4,3*SIZE//4,(1,),device=DEVICE)); radius=int(torch.randint(7,15,(1,),device=DEVICE))
        yy,xx=torch.meshgrid(torch.arange(SIZE,device=DEVICE),torch.arange(SIZE,device=DEVICE),indexing="ij")
        x[i,:,((xx-cx)**2+(yy-cy)**2)<radius*radius]=0
    return x


def train():
    torch.manual_seed(SEED); target=target_image()[None]
    model=NeuralCA().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=1e-2)
    pool=seed(48); losses=[]
    for step in range(2100):
        ids=torch.randperm(len(pool),device=DEVICE)[:8]; x=seed(8)
        if step > 1200 and step%3==0: x=damage(x)
        iterations=int(torch.randint(48,68,(1,),device=DEVICE))
        for _ in range(iterations): x=model(x,fire_rate=1.0)
        loss=(x[:,:4].clamp(0,1)-target).square().mean() + .0001*x[:,4:].square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        pool[ids]=x.detach(); losses.append(float(loss.detach().cpu()))
    return model,target[0],np.asarray(losses)


def to_image(state,size):
    rgba=state[:4].detach().clamp(0,1).permute(1,2,0).cpu().numpy()
    rgba[...,3]=np.clip(rgba[...,3],0,1)
    image=Image.fromarray((rgba*255).astype(np.uint8),"RGBA").resize(size,Image.Resampling.NEAREST)
    glow=image.copy().filter(ImageFilter.GaussianBlur(size[0]//65))
    dark=Image.new("RGB",size,"#08101d").convert("RGBA")
    dark=Image.alpha_composite(dark,glow.putalpha(glow.getchannel("A").point(lambda v:int(v*.45))) if False else glow)
    # Rebuild because PIL putalpha mutates; retain a restrained mineral-blue background.
    bg=Image.new("RGBA",size,(5,14,27,255)); aura=image.copy().filter(ImageFilter.GaussianBlur(size[0]//40)); aura.putalpha(aura.getchannel("A").point(lambda v:int(v*.34)))
    return Image.alpha_composite(Image.alpha_composite(bg,aura),image).convert("RGB")


def main():
    OUT.mkdir(exist_ok=True); model,target,losses=train(); model.eval()
    with torch.no_grad():
        x=seed(1); states=[]; damaged_at=70
        for tick in range(145):
            if tick==damaged_at: x=damage(x)
            for _ in range(2): x=model(x,fire_rate=.5)
            states.append(x[0].detach().clone())
    loop=[to_image(state,(720,720)) for state in states]
    master=to_image(states[-1],(2560,2560)); master.save(OUT/"master.png",quality=96)
    loop[0].save(OUT/"loop-production.gif",save_all=True,append_images=loop[1:],duration=55,loop=0,optimize=False)
    saved=torch.stack([states[i] for i in np.linspace(0,len(states)-1,40,dtype=int)]).cpu().numpy()
    np.savez_compressed(OUT/"production-data.npz",states=saved,target=target.cpu().numpy(),losses=losses)
    meta={"device":str(DEVICE),"seed":SEED,"training_steps":2100,"channels":CHANNELS,"grid_size":SIZE,"render_pixels":[2560,2560],"damage_tick":damaged_at,"visual_mapping":{"rgba":"first four channels of learned cell state","glow":"blurred alpha channel of learned state","erasure":"state deletion across visible and hidden channels"}}
    (OUT/"production-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")

if __name__=="__main__": main()
