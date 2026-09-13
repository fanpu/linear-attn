"""Production render: a learned reverse diffusion process as particle archaeology."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, TIMESTEPS, STEPS, FRAMES = 149, 72, 4200, 36


class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 2))
    def forward(self, x, time): return self.net(torch.cat((x, time[:,None]), 1))


def source(n=2600):
    g = torch.Generator().manual_seed(SEED)
    theta = torch.rand(n, generator=g) * torch.pi
    first = torch.stack((torch.cos(theta), torch.sin(theta)), 1)
    second = torch.stack((1-torch.cos(theta), .45-torch.sin(theta)), 1)
    result = torch.cat((first, second)) + .052 * torch.randn(2*n, 2, generator=g)
    return (result-result.mean(0))/result.std(0)


def gradient(values):
    stops = np.array([[5,13,25],[13,45,69],[28,111,120],[159,194,139],[248,209,119],[239,99,88]], dtype=float)/255
    value = np.clip(values,0,1)*(len(stops)-1); lo = value.astype(int); hi = np.clip(lo+1,0,len(stops)-1); frac = value-lo
    return stops[lo]*(1-frac)[...,None]+stops[hi]*frac[...,None]


def train():
    torch.manual_seed(SEED)
    data = source().to(DEVICE)
    beta = torch.linspace(.00035,.055,TIMESTEPS,device=DEVICE); alpha = 1-beta; alpha_bar = torch.cumprod(alpha,0)
    model, opt = ScoreNet().to(DEVICE), torch.optim.AdamW(ScoreNet().parameters(),lr=2e-3)
    # Rebind optimizer to the actual model; done explicitly to keep experiment construction readable.
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses=[]
    for step in range(STEPS):
        choose = torch.randint(len(data),(512,),device=DEVICE); t = torch.randint(0,TIMESTEPS,(512,),device=DEVICE)
        eps = torch.randn(512,2,device=DEVICE); ab = alpha_bar[t,None]
        noisy = ab.sqrt()*data[choose]+(1-ab).sqrt()*eps
        prediction = model(noisy,t.float()/(TIMESTEPS-1)); loss=(prediction-eps).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        sample=torch.randn(5200,2,device=DEVICE); states=[sample.cpu().numpy()]
        for t in range(TIMESTEPS-1,-1,-1):
            time=torch.full((len(sample),),t/(TIMESTEPS-1),device=DEVICE)
            estimate=model(sample,time); a,ab,b=alpha[t],alpha_bar[t],beta[t]
            sample=(sample-b/(1-ab).sqrt()*estimate)/a.sqrt()
            if t>0: sample=sample+b.sqrt()*torch.randn_like(sample)
            states.append(sample.cpu().numpy())
    return np.asarray(states), np.asarray(losses)


def coords(points, width, height):
    x = (points[:,0]+3.1)/6.2*width; y=(1-(points[:,1]+3.1)/6.2)*height
    return np.c_[x,y]


def compose(states, frame, size):
    width,height=size; scale=width/1280; state_index=int(round(frame/(FRAMES-1)*(len(states)-1))); points=states[state_index]
    image=Image.new("RGB",size,"#050d18")
    # Measured point density is the geological strata in the image.
    bins=700 if width>1600 else 360
    hist,_,_=np.histogram2d(points[:,0],points[:,1],bins=bins,range=[[-3.1,3.1],[-3.1,3.1]])
    log_density=np.log1p(hist); log_density/=log_density.max()+1e-8
    density=Image.fromarray((gradient(log_density.T[::-1])*255).astype(np.uint8),"RGB").resize(size,Image.Resampling.BICUBIC)
    density=density.filter(ImageFilter.GaussianBlur(1.4*scale))
    # Alpha comes from the density itself, so black empty space stays empty.
    density_alpha=Image.fromarray((np.clip(log_density.T[::-1]*1.5,0,1)*235).astype(np.uint8)).resize(size,Image.Resampling.BICUBIC)
    image=Image.composite(density,image,density_alpha)
    trails=Image.new("RGBA",size,(0,0,0,0)); td=ImageDraw.Draw(trails)
    selected=np.linspace(0, len(points)-1, 150, dtype=int)
    begin=max(0,state_index-20); history=states[begin:state_index+1:2]
    final=states[-1][selected]
    hue=(np.arctan2(final[:,1],final[:,0])+np.pi)/(2*np.pi)
    hues=(gradient(hue)*255).astype(np.uint8)
    for idx, color in zip(selected,hues):
        route=coords(history[:,idx],width,height)
        if len(route)>1: td.line([tuple(p) for p in route],fill=(int(color[0]),int(color[1]),int(color[2]),72),width=max(1,int(.75*scale)),joint="curve")
    image=Image.alpha_composite(image.convert("RGBA"),trails.filter(ImageFilter.GaussianBlur(.35*scale)))
    stars=Image.new("RGBA",size,(0,0,0,0)); sd=ImageDraw.Draw(stars)
    pixel=coords(points,width,height); rad=max(1,int(1.15*scale))
    # A fixed sample subset preserves identity across the denoising arc.
    for p, color in zip(pixel[::3], np.repeat([[247,226,168]],len(pixel[::3]),axis=0)):
        sd.ellipse((p[0]-rad,p[1]-rad,p[0]+rad,p[1]+rad),fill=(int(color[0]),int(color[1]),int(color[2]),48))
    image=Image.alpha_composite(image,stars)
    return image.convert("RGB")


def main():
    OUT.mkdir(exist_ok=True)
    states, losses=train()
    loop=[compose(states,i,(1280,720)) for i in range(FRAMES)]
    master=compose(states,FRAMES-1,(2560,1440))
    master.save(OUT/"master.png",quality=96)
    loop[0].save(OUT/"loop-production.gif",save_all=True,append_images=loop[1:],duration=85,loop=0,optimize=False)
    chosen=np.linspace(0,len(states)-1,FRAMES,dtype=int)
    np.savez_compressed(OUT/"production-data.npz",trajectory=states[chosen],final_samples=states[-1],losses=losses)
    meta={"device":str(DEVICE),"seed":SEED,"training_steps":STEPS,"diffusion_steps":TIMESTEPS,"sample_count":len(states[-1]),"master_pixels":[2560,1440],"visual_mapping":{"density_strata":"log density of reverse-diffusion samples","particle_paths":"last 20 genuine reverse diffusion states for fixed samples","path_color":"angle of final sample","particles":"current generated samples"}}
    (OUT/"production-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")

if __name__=="__main__": main()
