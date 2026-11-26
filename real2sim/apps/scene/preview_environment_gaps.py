"""Prepare an unapproved same-field preview of layered native GS cutouts."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session'); p.add_argument('--owners', nargs='+', required=True)
    p.add_argument('--coverage', required=True); p.add_argument('--output', required=True)
    p.add_argument('--keep-ray-mode', action='store_true')
    p.add_argument('--without-centre-depth', action='store_true', help='Diagnostic only: test whether the existing optional Gaussian-centre depth prior overpaints reliable GS')
    args = p.parse_args()
    from calibrated_scene_reconstruction.scene.background.gap_preview import prepare
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.sim.context_stage import apply_gap_mask

    result = prepare(Session(args.session), args.owners, args.coverage, args.output, apply_gap_mask,
                     layered=not args.keep_ray_mode, without_centre_depth=args.without_centre_depth)
    print(result['scene'])


if __name__ == '__main__':
    main()
