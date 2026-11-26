"""Retry an intact native process only while shared-GPU admission is deferred."""
import argparse
import json
import time
import threading

from calibrated_scene_reconstruction.physics.process import workflow


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('recording');parser.add_argument('--wait-seconds',type=int,default=1800)
    parser.add_argument('--progress-seconds',type=int,default=30)
    args=parser.parse_args();deadline=time.monotonic()+args.wait_seconds
    if args.progress_seconds<0:parser.error('--progress-seconds must be nonnegative')
    while True:
        state=workflow.status(args.recording)
        if state['state'] not in {'planned','deferred'}:break
        stopped=threading.Event();started=time.monotonic()
        def report():
            while not stopped.wait(args.progress_seconds):
                print(json.dumps({'waiting_for':'native_worker','round':state['round'],
                                  'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
        reporter=threading.Thread(target=report,daemon=True) if args.progress_seconds else None
        if reporter:reporter.start()
        try:state=workflow.run(args.recording)
        finally:
            stopped.set()
            if reporter:reporter.join(timeout=1.)
        print(json.dumps(state),flush=True)
        if state['state']!='deferred' or time.monotonic()>=deadline:break
        time.sleep(min(15,max(0,deadline-time.monotonic())))
