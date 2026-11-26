"""Source-image centerlines of reviewed thin stripes, independent of simulation.

Line detectors propose intensity boundaries. Unequal fragment lengths or a
single visible boundary cannot establish the center of a cylindrical part.
Recover both contrast transitions in the original pixels before fitting its
centerline; ambiguous/occluded evidence fails explicitly.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from .contract import require
from .line_features import line_probes


def refine(image, groups, probe_y, config):
    """Return refined copies of source-only edge groups and their probe values.

    Thresholds describe image evidence in pixels/intensity units. No native
    geometry, rendered image, fitted parameter or response state is accepted.
    The original reviewed fragments and their source identities are retained.
    """
    gray=np.asarray(image,float)
    require(gray.ndim==2 and np.isfinite(gray).all(), 'Finite source grayscale image required')
    probe_y=np.asarray(probe_y,float)
    require(probe_y.shape==(2,) and np.isfinite(probe_y).all() and abs(probe_y[1]-probe_y[0])>1e-6,
            'Two finite distinct source measurement rows required')
    polarity=config['polarity'];require(polarity in {'dark','light','auto'},'Declare stripe contrast polarity')
    if polarity=='light':gray=-gray
    window=float(config['half_window_px']);step=float(config['sample_step_px'])
    sigma=float(config['blur_sigma_px']);gradient=float(config['min_gradient'])
    low,high=map(float,config['width_bounds_px']);edge_distance=float(config['edge_distance_px'])
    shift=float(config['max_center_shift_px']);residual_limit=float(config['max_residual_px'])
    minimum=config['min_rows']
    prior_weight=float(config.get('center_prior_weight',.25 if polarity=='auto' else 0.))
    trim_fraction=config.get('end_trim_fraction')
    require(np.isfinite(prior_weight) and 0<=prior_weight<=2 and
            (trim_fraction is None or np.isfinite(trim_fraction) and 0<=trim_fraction<.5),
            'Bounded source proposal disambiguation and fragment trimming required')
    require(0<step<=1 and 0<sigma<=window/2 and 0<low<high<2*window and
            gradient>0 and 0<edge_distance<window and 0<shift<window and
            residual_limit>0 and type(minimum) is int and minimum>=8,
            'Bounded source image refinement and explicit uncertainty limits required')
    offsets=np.arange(-window,window+step/2,step);values=[];reports=[]
    for group in groups:
        centers=[];contrast_types=[]
        for segment in group['source_segments']:
            points=np.asarray(segment,float);points=points[np.argsort(points[:,1])]
            span=points[1,1]-points[0,1]
            require(span>6,'Reviewed fragment too short for source profile refinement')
            trim=3 if trim_fraction is None else min(3,float(trim_fraction)*span)
            ys=np.linspace(points[0,1]+trim,points[1,1]-trim,max(12,round(span)))
            xs=np.interp(ys,points[:,1],points[:,0])
            require(ys.min()>=0 and ys.max()<gray.shape[0] and
                    xs.min()-window>=0 and xs.max()+window<gray.shape[1],
                    'Source profiles must lie inside the original image')
            profiles=map_coordinates(gray,[np.broadcast_to(ys[:,None],(len(ys),len(offsets))).ravel(),
                (xs[:,None]+offsets).ravel()],order=1).reshape(len(ys),len(offsets))
            profiles=gaussian_filter1d(profiles,sigma/step,axis=1)
            gradients=np.gradient(profiles,step,axis=1)
            for y,x,derivative in zip(ys,xs,gradients):
                options=[]
                rough=group['probe_x'][0]+(group['probe_x'][1]-group['probe_x'][0])*(y-probe_y[0])/(probe_y[1]-probe_y[0])
                for contrast in ((1,-1) if polarity=='auto' else (1,)):
                    left=find_peaks(-derivative*contrast,prominence=gradient)[0]
                    right=find_peaks(derivative*contrast,prominence=gradient)[0]
                    for a in left:
                        for b in right:
                            width=offsets[b]-offsets[a]
                            distance=min(abs(offsets[a]),abs(offsets[b]))
                            center=(offsets[a]+offsets[b])/2
                            if low<=width<=high and distance<=edge_distance and (polarity!='auto' or abs(center)<=shift):
                                options.append((distance+.05*width+prior_weight*abs(x+center-rough),a,b,contrast))
                if not options:continue
                _,a,b,contrast=min(options);center=(offsets[a]+offsets[b])/2
                if abs(center)>shift:continue
                centers.append([float(y),float(x+center),float(offsets[b]-offsets[a])])
                contrast_types.append('light' if polarity=='light' or contrast==-1 else 'dark')
        require(len(centers)>=minimum,'Insufficient paired source contrast transitions; review visibility or supply more source evidence')
        samples=np.asarray(centers);mid=float(samples[:,0].mean())
        design=np.column_stack([samples[:,0]-mid,np.ones(len(samples))])
        initial=np.linalg.lstsq(design,samples[:,1],rcond=None)[0]
        fit=least_squares(lambda p:design@p-samples[:,1],initial,loss='soft_l1',f_scale=.5)
        residual=design@fit.x-samples[:,1]
        rms=float(np.sqrt(np.mean(residual**2)))
        require(rms<=residual_limit,'Source stripe centers disagree beyond the declared pixel uncertainty')
        endpoints=np.array([[fit.x[1]+fit.x[0]*(samples[:,0].min()-mid),samples[:,0].min()],
                            [fit.x[1]+fit.x[0]*(samples[:,0].max()-mid),samples[:,0].max()]])
        probes=line_probes(endpoints,probe_y)
        require(np.max(np.abs(probes-np.asarray(group['probe_x'])))<=shift,
                'Refined source centerline left the reviewed fragment neighborhood')
        report={**group,'edge_group_probe_x':group['probe_x'],'probe_x':probes.tolist(),
                'measurement_kind':'source_pixel_stripe_centerline',
                'centerline_segment':endpoints.tolist(),
                'profile_samples_y_x_width':samples.tolist(),
                'median_width_px':float(np.median(samples[:,2])),
                'centerline_residual_px':rms,
                'contrast_samples':{kind:contrast_types.count(kind) for kind in ['dark','light']},
                'source_y_span_px':float(np.ptp(samples[:,0])),
                'uncertainty_scope':'Observed contrast profiles, finite width and line-fit residual; a lone detected edge has no zero-uncertainty center.'}
        values.append(probes);reports.append(report)
    return np.asarray(values).reshape(-1,2),reports
