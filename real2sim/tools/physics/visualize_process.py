"""Plot observed and native simulated features; optionally overlay image channels."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import read,require,write,digest


def visualize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.round);plan=read(root/'plan.json');result=read(root/'result.json')
    index=args.trial if args.trial is not None else result.get('best',{}).get('index')
    require(index is not None,'No completed native candidate to visualize')
    trial=root/'trials'/f'{index:04d}';observed=dict(np.load(root/'observations.npz'))
    predicted=dict(np.load(trial/'predictions.npz'));out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    channels=plan['channels'];fig,axes=plt.subplots(len(channels),2,figsize=(12,3.4*len(channels)),squeeze=False)
    times=observed['t'];held=np.asarray(plan['comparison']['validation_indices'])
    for row,c in enumerate(channels):
        a=observed[c['observed']].reshape(len(times),-1);b=predicted[c['predicted']].reshape(len(times),-1)
        require(len(b)==len(times) and np.max(abs(times-predicted['t']))<1e-8,'Exact observation timebase required')
        for j in range(a.shape[1]):
            color=f'C{j%10}'
            axes[row,0].plot(times,a[:,j],'.',color=color,label=f'source {j}')
            axes[row,0].plot(times,b[:,j],color=color,label=f'native {j}')
        error=np.sqrt(np.mean((a-b)**2,axis=1));axes[row,1].plot(times,error,color='C0')
        axes[row,1].plot(times[held],error[held],'o',mfc='white',ms=4,label='held out')
        axes[row,1].axhline(c['tolerance'],color='C3',ls='--',label='RMSE acceptance limit')
        axes[row,0].set_title(c['id']);axes[row,0].set_ylabel(c['unit'])
        axes[row,1].set_title('Per-frame error (acceptance uses split RMSE)');axes[row,1].set_ylabel(c['unit'])
        for ax in axes[row]:ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    if args.video_input:
        video=read(root/'inputs.json')[args.video_input]['path'];cap=cv2.VideoCapture(video)
        fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer=cv2.VideoWriter(str(out/'feature_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        require(writer.isOpened(),'Cannot create diagnostic video')
        valid_channels=[c for c in channels if c['unit']=='px' and observed[c['observed']].shape==(len(times),2)]
        require(bool(valid_channels),'Image overlay requires 2D pixel observations')
        for i,t in enumerate(times):
            frame=int(round(t*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();require(ok,'Missing source image')
            for c in valid_channels:
                for values,color in [(observed[c['observed']][i],(50,220,50)),(predicted[c['predicted']][i],(255,160,20))]:
                    if not np.isfinite(values).all():continue
                    p=tuple(np.rint(values).astype(int));cv2.circle(image,p,7,color,2)
                    cv2.putText(image,c['id'],(p[0]+10,p[1]-8),0,.5,color,1)
            cv2.putText(image,'Green: source feature. Blue: native projected feature.',(30,40),0,.8,(255,255,255),2)
            cv2.putText(image,'Feature diagnostic; this is not a rendered simulation video.',(30,75),0,.65,(255,255,255),2)
            writer.write(image)
        writer.release();cap.release()
    write(out/'visualization.json',{'round':str(root.resolve()),'trial':index,'process_pass':bool(result['pass']),
        'scope':'Observed versus simulated measurement features; not a rendered simulation or approval.',
        'inputs':{str(p.resolve()):digest(p) for p in [root/'plan.json',root/'result.json',root/'observations.npz',trial/'predictions.npz',Path(__file__)]}})
    print(str(out.resolve()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--round',required=True)
    parser.add_argument('--trial',type=int);parser.add_argument('--output',required=True)
    parser.add_argument('--video-input',help='Optional named video input for 2D source/native feature overlays')
    visualize(parser.parse_args())
