"""Within-group measurements and natural-region recombination diagnostics.

No inference of perceptual acceptability from a scalar descriptor distance.
No noise injection. Regions use one stereo-preserving crossfade mask.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy import signal

VERSION = 'natural_event_regions_v1'


def audio_values(audio):
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] < 512 or a.shape[1] not in (1, 2):
        raise ValueError('Expected at least 512 mono/stereo samples')
    if not np.isfinite(a).all():
        raise ValueError('Non-finite audio')
    return a


def shape_descriptor(audio, sr):
    """Gain-invariant, bounded diagnostic shape vector, including the onset.

    Normalize the whole event before measuring; compute nonnegative normalized
    band power, relative temporal energy and channel covariance. No absolute
    PSD floor, logarithmic floor or inverse MAD weights are used.
    This intentionally measures coarse event shape, not a perceptual score.
    """
    a = audio_values(audio)
    if sr < 8000:
        raise ValueError('Sample rate must be at least 8000 Hz')
    peak = float(np.max(np.abs(a)))
    if peak == 0:
        raise ValueError('Silent audio has no event shape')
    a = a / peak
    energy = float(np.sum(a * a))
    if len(a) < round(1.6 * sr):
        raise ValueError('Prepared event must contain at least 1.6 seconds')
    temporal_edges = (0, .020, .027, .040, .055, .100, .180, .350, .650, 1.0, 1.6, len(a)/sr)
    temporal = [float(np.sum(a[round(lo*sr):round(hi*sr)]**2))/energy
                for lo, hi in zip(temporal_edges[:-1], temporal_edges[1:])]
    bands = np.geomspace(40, min(19500, .49*sr), 9)
    parts = [np.asarray(temporal)]
    for lo, hi in ((.020, .055), (.055, .180), (.180, .650), (.650, 1.6)):
        region = a[round(lo*sr):round(hi*sr)]
        # All zones have enough samples at supported rates; avoid implicit pad.
        size = min(1024, len(region))
        f, psd = signal.welch(region, fs=sr, nperseg=size,
                              noverlap=size//2, axis=0, detrend=False, scaling='spectrum')
        p = np.sum(psd, axis=1)
        band = np.array([np.sum(p[(f>=left)&(f<right)]) for left,right in zip(bands[:-1],bands[1:])])
        # Each region contributes in proportion to its share of event energy.
        weight = float(np.sum(region**2))/energy
        parts.append(weight*band/band.sum() if band.sum()>0 else np.zeros(8))
        if a.shape[1]==2:
            cov = region.T @ region
            denom = float(np.trace(cov))
            parts.append(weight*np.array([cov[0,0],cov[1,1],cov[0,1]])/denom
                         if denom>0 else np.zeros(3))
        else:
            parts.append(weight*np.array([.5,.5,.5]))
    return np.concatenate(parts)


@dataclass
class SmallGroupProfile:
    group: str
    names: tuple
    sample_rate: int
    descriptors: np.ndarray
    pairwise: np.ndarray

    def summary(self):
        return {'version': VERSION, 'group': self.group, 'names': list(self.names),
                'sample_rate': self.sample_rate, 'descriptor_dimensions': self.descriptors.shape[1],
                'distance': 'unweighted Euclidean shape diagnostic; not perceptual acceptability',
                'within_group_quantiles': np.quantile(self.pairwise,[0,.25,.5,.75,1]).tolist()}


def fit_small_groups(clips):
    """Fit each filename group independently; unrelated groups cannot set scale."""
    groups = {}
    for c in clips:
        groups.setdefault(c.group, []).append(c)
    result = {}
    for group, bank in groups.items():
        if not 3 <= len(bank) <= 5:
            raise ValueError(f'Group {group}: this pilot requires 3–5 takes')
        if len({(c.sample_rate,c.prepared.shape) for c in bank}) != 1:
            raise ValueError('Mixed formats/shapes within a group')
        d = np.stack([shape_descriptor(c.prepared,c.sample_rate) for c in bank])
        distances=np.array([np.linalg.norm(d[i]-d[j]) for i,j in combinations(range(len(bank)),2)])
        result[group]=SmallGroupProfile(group,tuple(c.metrics.name for c in bank),bank[0].sample_rate,d,distances)
    return result


def select_seam(reference, donor, sr, *, minimum_s=.045, maximum_s=.065, fade_s=.006):
    """Locate low joint energy near the old protected-region boundary."""
    a,b=audio_values(reference),audio_values(donor)
    if a.shape!=b.shape:
        raise ValueError('Mismatched prepared events')
    radius=max(1,round(fade_s*sr/2))
    lo=max(radius,round(minimum_s*sr));hi=min(len(a)-radius,round(maximum_s*sr))
    if hi<=lo:
        raise ValueError('Invalid seam search interval')
    energy=np.mean(a*a+b*b,axis=1)
    sums=np.convolve(energy,np.ones(2*radius+1),mode='same')
    frame=lo+int(np.argmin(sums[lo:hi]))
    return frame/sr


def exchange_regions(reference, donor, sr, *, seam_s, fade_s=.006):
    """Return donor-early/ref-late and ref-early/donor-late diagnostics.

    Convex smoothstep crossfade, identical in both channels; no gain matching,
    stochastic carrier, spectral transform, added fade or pitch/time change.
    These are recombinations of natural takes, not independent new recordings.
    """
    a,b=audio_values(reference),audio_values(donor)
    if a.shape!=b.shape or not np.isfinite([seam_s,fade_s]).all() or fade_s<=0:
        raise ValueError('Invalid region exchange inputs')
    start=round((seam_s-fade_s/2)*sr);end=round((seam_s+fade_s/2)*sr)
    if not 0<=start<end<=len(a):
        raise ValueError('Crossfade falls outside event')
    w=np.zeros(len(a));w[end:]=1
    u=np.linspace(0,1,end-start)
    w[start:end]=u*u*(3-2*u)
    early=a+(b-a)*(1-w[:,None])
    late=a+(b-a)*w[:,None]
    early[:start]=b[:start];early[end:]=a[end:]
    late[:start]=a[:start];late[end:]=b[end:]
    return early,late


def regional_evidence(reference, candidate, sr, *, seam_s):
    a,b=audio_values(reference),audio_values(candidate)
    if a.shape!=b.shape:
        raise ValueError('Shape mismatch')
    energy=float(np.sum(a*a))
    if energy==0:
        raise ValueError('Silent reference')
    delta=b-a;de=float(np.sum(delta*delta));cut=round(seam_s*sr)
    return {'shape_distance':float(np.linalg.norm(shape_descriptor(a,sr)-shape_descriptor(b,sr))),
            'reference_energy_before_seam_percent':100*float(np.sum(a[:cut]**2))/energy,
            'difference_energy_before_seam_percent':100*float(np.sum(delta[:cut]**2))/de if de else None,
            'difference_energy_relative_percent':100*de/energy,
            'rms_ratio':float(np.sqrt(np.sum(b*b)/energy)),
            'peak':float(np.max(np.abs(b)))}
