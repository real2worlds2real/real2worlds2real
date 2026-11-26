"""Versioned support/occluder observations, preserving previous evidence."""
import argparse
import re

from calibrated_scene_reconstruction.scene.session import Session


def settings(session):
    configured = session.config('support_masks.json', {}) or {}
    if not isinstance(configured, dict):
        raise ValueError('Support observation revisions must be an object')
    value = {'support_suffix': '_carve', 'occluder_suffix': '_occ', **configured}
    suffix_keys={'support_suffix','occluder_suffix'}
    if set(value)-suffix_keys-{'support_prompts','occluder_prompts','extra_occluder_suffixes'} or any(
            not isinstance(value[k],str) or not re.fullmatch(r'_[A-Za-z0-9_]+',value[k]) for k in suffix_keys):
        raise ValueError('Support-mask revisions require local support_suffix/occluder_suffix')
    for key in ('support_prompts','occluder_prompts'):
        if key in value and (not isinstance(value[key],list) or not value[key] or any(
                not isinstance(p,str) or not p.strip() or any(c in p for c in ',\r\n') for p in value[key])):
            raise ValueError('Support-mask queries require separate nonempty semantic phrases')
    if value['support_suffix'] == value['occluder_suffix']:
        raise ValueError('Positive support and occluders require separate observation revisions')
    if 'extra_occluder_suffixes' in value:
        extra=value['extra_occluder_suffixes']
        if (not isinstance(extra,list) or any(not isinstance(v,str) or not re.fullmatch(r'_[A-Za-z0-9_]+',v) for v in extra)
                or len(set(extra))!=len(extra) or set(extra)&{value[k] for k in suffix_keys}):
            raise ValueError('Extra occluders require distinct local independent mask revisions')
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('session')
    a = p.parse_args()
    cfg = settings(Session(a.session))
    print(cfg['support_suffix'], cfg['occluder_suffix'])


if __name__ == '__main__':
    main()
