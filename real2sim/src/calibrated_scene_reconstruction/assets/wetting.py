"""Finite contact exposure and flux-limited absorption, independent of Isaac."""
import numpy as np


class ContactAbsorption:
    def __init__(self,count,particle_mass,seed=17,capacity_kg=.018,mass_rate_kg_s=.006,
                 minimum_dwell_s=.10,mean_extra_dwell_s=.12):
        if minimum_dwell_s<=0 or mean_extra_dwell_s<0:raise ValueError('Uptake requires a finite positive contact dwell')
        self.ids=np.arange(count)
        self.exposure=np.zeros(count)
        self.threshold=minimum_dwell_s+np.random.default_rng(seed).exponential(mean_extra_dwell_s,count)
        self.particle_mass=particle_mass;self.capacity=capacity_kg;self.rate=mass_rate_kg_s
        self.absorbed_mass=0.;self.budget=0.

    def advance(self,contact_weight,dt):
        weight=np.asarray(contact_weight,float)
        if weight.shape!=self.exposure.shape or dt<=0 or not np.isfinite(weight).all():
            raise ValueError('Invalid contact exposure')
        self.exposure+=np.clip(weight,0,1)*dt
        saturation=self.absorbed_mass/self.capacity
        flux=self.rate*max(0.,1-saturation)
        self.budget=min(self.budget+flux*dt,flux*.05) if (weight>0).any() else 0.
        eligible=np.flatnonzero((weight>0)&(self.exposure>=self.threshold))
        maximum=max(0,int((min(self.budget,self.capacity-self.absorbed_mass)+1e-14)/self.particle_mass))
        chosen=eligible[np.argsort(-(self.exposure[eligible]-self.threshold[eligible]))[:maximum]]
        remove=np.zeros(len(weight),bool);remove[chosen]=True
        mass=len(chosen)*self.particle_mass
        self.budget-=mass;self.absorbed_mass+=mass
        self.ids=self.ids[~remove];self.exposure=self.exposure[~remove];self.threshold=self.threshold[~remove]
        return remove


def transfer_water_mass(masses,velocities,water_mass,water_momentum):
    masses=np.asarray(masses,float);velocities=np.asarray(velocities,float)
    added=masses/masses.sum()*water_mass
    new_mass=masses+added
    momentum=masses[:,None]*velocities+added[:,None]*(np.asarray(water_momentum)/water_mass)
    return new_mass,momentum/new_mass[:,None]
