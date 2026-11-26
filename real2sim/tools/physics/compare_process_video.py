"""Compare rendered native process states with source video and source-only masks."""
import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.video import VideoWriter
from calibrated_scene_reconstruction.physics.process.contract import read,write,require,digest


REAL=(45,245,100)
SIM=(255,70,200)


def fit_image(image,width,height):
    scale=min(width/image.shape[1],height/image.shape[0])
    resized=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    canvas=np.full((height,width,3),20,np.uint8)
    y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return canvas


def crop_with_padding(image,bounds):
    x1,y1,x2,y2=map(int,bounds);require(x2>x1 and y2>y1,'Positive comparison crop required')
    result=np.full((y2-y1,x2-x1,3),20,np.uint8)
    a,b,c,d=max(0,x1),max(0,y1),min(image.shape[1],x2),min(image.shape[0],y2)
    if c>a and d>b:result[b-y1:d-y1,a-x1:c-x1]=image[b:d,a:c]
    return result


def outline(image,mask,color):
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contours,-1,color,2,lineType=cv2.LINE_AA)


def reviewed_mask(source, frame, parts, size, review):
    """Nonempty source proposals still need identity/visibility review."""
    result=np.zeros((size[1],size[0]),np.uint8);uncertain=[]
    for part in parts:
        key=part['id'];rule=review['parts'].get(key)
        require(rule and rule.get('reason'),'Every source part needs an explicit mask review')
        frames=rule['trusted_frames']
        trusted=frames=='all' or (isinstance(frames,list) and frame in frames)
        if not trusted or part['pixels']==0:
            uncertain.append(key);continue
        mask=cv2.imread(str(source/key/f'{frame:05d}.png'),0)
        require(mask is not None,'Missing reviewed source mask: '+key)
        result=cv2.bitwise_or(result,cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST))
        complete=rule.get('complete_frames',frames)
        if not (complete=='all' or isinstance(complete,list) and frame in complete):uncertain.append(key)
    return result,uncertain


def validate_video_source(video, metadata, mask_metadata, fps, count):
    """Equal frame counts cannot prove that two clips show the same process."""
    source_hash=digest(video)
    require(source_hash in metadata['sha256'].values()
            and source_hash in mask_metadata['source_sha256'].values(),
            'Comparison video must match both the native render and source-mask evidence')
    require(count==len(metadata['frames']),'Video and native frame count mismatch')
    require(np.isfinite(fps) and fps>0 and np.isclose(fps,metadata['fps'],rtol=1e-4,atol=1e-6)
            and np.isclose(fps,mask_metadata['fps'],rtol=1e-4,atol=1e-6),
            'Comparison requires the same video frame rate and timebase')


def compare(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    render=Path(args.render);source=Path(args.source_masks);metadata=read(render/'render.json');mask_metadata=read(source/'masks.json')
    features=np.load(args.regions,allow_pickle=False)['features'];count=len(metadata['frames']);fps=metadata['fps']
    require(len(mask_metadata['frames'])==count and all(row['frame']==i for i,row in enumerate(mask_metadata['frames'])),'Complete, source-ordered real masks required')
    require(len(features)==count,'Source region/timebase mismatch')
    for name,expected in metadata['sha256'].items():require(digest(name)==expected,'Changed native render source: '+name)
    for name,expected in mask_metadata['source_sha256'].items():require(digest(name)==expected,'Changed segmentation source: '+name)
    mask_review=read(args.mask_review) if args.mask_review else None
    if mask_review:
        require(mask_review.get('reason') and mask_review.get('mask_manifest_sha256')==digest(source/'masks.json'),
                'Mask review must bind the current source proposals')
    line_records={}
    if args.line_review:
        line_review=read(args.line_review)
        require(line_review.get('pass') is True and line_review.get('reason'),'Source segment review required')
        for path,expected in line_review['source_sha256'].items():require(digest(path)==expected,'Changed source line evidence')
        line_records={r['frame']:r for r in line_review['frames']}
    cap=cv2.VideoCapture(args.video)
    validate_video_source(args.video,metadata,mask_metadata,cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    require(features.ndim==2 and features.shape[1]>=4 and np.isfinite(features[:,[0,1,3]]).all()
            and (features[:,3]>0).all(),'Comparison needs finite source ROI centers and positive width scales')
    W,H=metadata['width'],metadata['height'];source_w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    force_data=None
    if getattr(args,'force_trace',None):
        from calibrated_scene_reconstruction.physics.process import force_visualization
        from calibrated_scene_reconstruction.physics.process.force_visualization import load_force,draw_force,draw_force_hud
        force_data=load_force(metadata,args.force_trace,args.camera_input,args.force_scale)
    ratios=W/source_w
    source_unit=features[:,3]*ratios
    # Crop smoothing is presentation only; it never changes images, poses or masks.
    units=np.array([np.median(source_unit[max(0,i-3):min(count,i+4)]) for i in range(count)])
    main=VideoWriter(out/'real_vs_simulation.mp4',1920,1080,float(fps),crf=17)
    over=VideoWriter(out/'silhouette_overlay.mp4',W,H,float(fps),crf=17)
    force_video=VideoWriter(out/'simulation_force.mp4',W,H,float(fps),crf=17) if force_data else None
    selected={0,count//2,round(count*.65),count-1};metrics=[]
    image_hashes={}
    try:
        for pass_index,(speed,repeats) in enumerate([(1.,1),(.5,2)]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            for i in range(count):
                ok,bgr=cap.read();require(ok,'Source decode incomplete');real=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(W,H),interpolation=cv2.INTER_AREA)
                sim=cv2.cvtColor(cv2.imread(str(render/'rgb'/f'{i:05d}.png')),cv2.COLOR_BGR2RGB)
                if mask_review:
                    observed,uncertain=reviewed_mask(source,i,mask_metadata['frames'][i]['parts'],(W,H),mask_review)
                else:
                    observed=cv2.resize(cv2.imread(str(source/'union'/f'{i:05d}.png'),0),(W,H),interpolation=cv2.INTER_NEAREST)
                    uncertain=[p['id'] for p in mask_metadata['frames'][i]['parts'] if p['pixels']==0]
                predicted=cv2.imread(str(render/'masks'/f'{i:05d}.png'),0)
                overlay=real.copy();outline(overlay,observed,REAL);outline(overlay,predicted,SIM)
                for group in line_records.get(i,{}).get('groups',[]):
                    for endpoints in group['source_segments']:
                        a,b=np.rint(np.asarray(endpoints)*ratios).astype(int)
                        cv2.line(overlay,tuple(a),tuple(b),REAL,2,cv2.LINE_AA)
                sim_force=None
                if force_data:
                    # Force data is a presentation layer over the unchanged recorded RGB.
                    sim_force=sim.copy();draw_force(sim_force,force_data,i)
                    draw_force_hud(sim_force,force_data,i)
                    for im in [real,sim,overlay]:draw_force(im,force_data,i)
                cx,cy=features[i,:2]*ratios;unit=units[i]
                bounds=np.rint(np.array([cx,cy,cx,cy])+unit*np.asarray(args.crop_offsets)).astype(int)
                panel=np.full((1080,1920,3),20,np.uint8)
                titles=['Real','Isaac replay - FAILED checks' if metadata.get('diagnostic_failed_result') else 'Isaac replay','Contours: green = real, pink = simulation']
                for col,(image,title) in enumerate(zip([real,sim,overlay],titles)):
                    x=col*640
                    panel[46:406,x:x+640]=fit_image(image,640,360)
                    panel[450:1080,x:x+640]=fit_image(crop_with_padding(image,bounds),640,630)
                    cv2.putText(panel,title,(x+14,29),0,.58,(240,240,240),1,cv2.LINE_AA)
                    subtitle=('Partial source contours: '+', '.join(uncertain)) if col==2 and uncertain else 'Task area - same crop'
                    cv2.putText(panel,subtitle,(x+14,434),0,.52,(210,210,210),1,cv2.LINE_AA)
                if force_data:
                    # Place the timeline in the empty simulation background, away from the object crop.
                    draw_force_hud(panel,force_data,i,origin=(650,455),size=(590,190))
                cv2.rectangle(panel,(5,376),(195,404),(16,16,16),-1)
                cv2.putText(panel,f't = {metadata["frames"][i]["source_t"]:.2f}s | {speed:g}x',(12,397),0,.54,(255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(overlay,(0,0),(W,58),(16,16,16),-1)
                cv2.putText(overlay,'Green: source masks / visible segments   Pink: rendered simulation',(16,25),0,.62,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(overlay,f't = {metadata["frames"][i]["source_t"]:.2f}s | {speed:g}x | Same recorded camera; native physics replay',(16,49),0,.52,(235,235,235),1,cv2.LINE_AA)
                if uncertain:
                    cv2.rectangle(overlay,(0,58),(W,85),(16,16,16),-1)
                    cv2.putText(overlay,'Partial source contours; untrusted or occluded masks omitted: '+', '.join(uncertain),(16,78),0,.52,(255,210,100),1,cv2.LINE_AA)
                if force_data:draw_force_hud(overlay,force_data,i,origin=(14,H-205))
                for _ in range(repeats):
                    main.write(panel);over.write(overlay)
                    if force_video:force_video.write(sim_force)
                if pass_index==0:
                    paths=[render/'rgb'/f'{i:05d}.png',render/'masks'/f'{i:05d}.png']
                    if mask_review:
                        paths += [source/part['id']/f'{i:05d}.png' for part in mask_metadata['frames'][i]['parts']
                                  if part['id'] not in uncertain]
                    else:paths += [source/'union'/f'{i:05d}.png']
                    image_hashes.update({str(path.resolve()):digest(path) for path in paths})
                    intersection=((observed>0)&(predicted>0)).sum();union=((observed>0)|(predicted>0)).sum()
                    metrics.append({'frame':i,'silhouette_iou':None if uncertain else float(intersection/max(union,1)),
                                    'source_mask_uncertain':uncertain,'reviewed_visible_line_groups':len(line_records.get(i,{}).get('groups',[]))})
                    if i in selected:
                        cv2.imwrite(str(out/f'comparison_{i:05d}.jpg'),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
                if i%30==0:print(f'[compare] pass={pass_index+1} frame={i+1}/{count}',flush=True)
    finally:
        main.close();over.close();cap.release()
        if force_video:force_video.close()
    shutil.copy2(render/'simulation_replay.mp4',out/'simulation_replay.mp4')
    shutil.copy2(args.video,out/'real_input.mp4')
    write(out/'comparison.json',{'native_render':str(render.resolve()),'source_masks':str(source.resolve()),
        'fps':fps,'native_frames':count,'comparison_frames':count*3,'playback':['original speed','half speed with duplicated recorded frames'],
        'new_physics_run':False,'geometry_changed':False,'pose_or_shape_interpolation':False,
        'diagnostic_failed_result':metadata.get('diagnostic_failed_result',False),
        'source_silhouette_scope':'Source-only SAM3 segmentation proposals; transparency, occlusion, blur and thin rods can cause segmentation uncertainty. These are not measured ground-truth shapes.',
        'native_scope':metadata['scope'],'mask_metrics':metrics,'image_sha256':image_hashes,
        'external_force':force_data['receipt'] if force_data else None,
        'scope':'Real video, actual Isaac-rendered native states, and source/renderer contour overlay. No real background is pasted into the simulation panel.',
        'inputs':{str(Path(p).resolve()):digest(p) for p in [args.video,args.regions,render/'render.json',source/'masks.json',Path(__file__)]
                  +[p for p in [args.mask_review,args.line_review,getattr(args,'force_trace',None)] if p]
                  +([str(Path(force_visualization.__file__).resolve())] if force_data else [])}})
    print(str(out.resolve()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['render','source-masks','regions','video','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--mask-review',help='Source part/frame identity and visibility review; excludes untrusted nonempty masks')
    p.add_argument('--line-review',help='Reviewed source-visible segments to overlay; never draws inferred hidden tips')
    p.add_argument('--force-trace',help='Sealed best-trial NPZ commands [t, Fx, Fy, Fz, px, py, pz] in world SI units')
    p.add_argument('--camera-input',default='process_camera',help='Named recorded render camera for force projection')
    p.add_argument('--force-scale',type=float,default=.06,help='World meters per newton for arrow display only; does not alter physics')
    p.add_argument('--crop-offsets',nargs=4,type=float,default=[-2,-2,2,2],metavar=('LEFT','TOP','RIGHT','BOTTOM'),help='Presentation crop in observed feature-width units; same crop for real and simulation')
    compare(p.parse_args())
