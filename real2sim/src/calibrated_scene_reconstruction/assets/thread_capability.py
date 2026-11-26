"""Declaration rules for native mating thread contact."""


def validate(mechanism):
    if mechanism['type'] != 'contact':
        raise ValueError('Threads require real mating surface contact, never a revolute/screw joint')
    needed = {'screw_on', 'retention', 'screw_off', 'release', 'pull_without_rotation'}
    if not needed.issubset(mechanism['checks']):
        raise ValueError(f"{mechanism['id']}: thread validation must include {sorted(needed)}")
