import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import soundfile as sf

from natural_event_regions import shape_descriptor,fit_small_groups,exchange_regions,select_seam
from run_natural_region_gate import build_package,verify_package,decode_ratings

SR=16000


def take(seed):
    rng=np.random.default_rng(seed);a=np.zeros((40000,2))
    # Energy overwhelmingly in first few ms, like the audited real recordings.
    t=np.arange(500)/SR
    attack=np.sin(2*np.pi*(1000+seed*30)*t)*np.exp(-t*1200)
    a[320:820,0]=attack;a[320:820,1]=attack*.9
    a[1000:10000]+=rng.normal(0,.0001,(9000,2))
    return a


class RegionTests(unittest.TestCase):
    def test_shape_independent_of_gain_and_no_absolute_psd_floor(self):
        a=take(1);d=shape_descriptor(a,SR)
        for gain in (1e-8,.01,.5,50):
            np.testing.assert_allclose(shape_descriptor(a*gain,SR),d,atol=1e-12,rtol=1e-10)
        with self.assertRaises(ValueError):shape_descriptor(np.zeros_like(a),SR)

    def test_stereo_energy_not_cancelled_by_antiphase(self):
        mono=take(1)[:,0]
        same=np.column_stack([mono,mono]);anti=np.column_stack([mono,-mono])
        a,b=shape_descriptor(same,SR),shape_descriptor(anti,SR)
        np.testing.assert_allclose(a[:11],b[:11])
        for off in (11,22,33,44):np.testing.assert_allclose(a[off:off+8],b[off:off+8])
        self.assertGreater(float(np.linalg.norm(a-b)),.01)

    def test_unrelated_group_cannot_change_profile(self):
        def clip(group,i,gain=1):
            return SimpleNamespace(group=group,prepared=take(i)*gain,sample_rate=SR,metrics=SimpleNamespace(name=f'{group}_{i}'))
        first=[clip('1',i) for i in range(3)]
        alone=fit_small_groups(first)['1']
        together=fit_small_groups(first+[clip('2',i+7,.0001) for i in range(5)])['1']
        np.testing.assert_array_equal(alone.descriptors,together.descriptors)
        np.testing.assert_array_equal(alone.pairwise,together.pairwise)

    def test_exact_sham_complementary_regions_and_no_noise(self):
        a,b=take(1),take(2);seam=select_seam(a,b,SR)
        early,late=exchange_regions(a,b,SR,seam_s=seam)
        np.testing.assert_allclose(early+late,a+b,atol=1e-16)
        left=round((seam-.003)*SR);right=round((seam+.003)*SR)
        np.testing.assert_array_equal(early[:left],b[:left])
        np.testing.assert_array_equal(late[:left],a[:left])
        np.testing.assert_array_equal(early[right:],a[right:])
        np.testing.assert_allclose(late[right:],b[right:],atol=1e-16)
        sham,_=exchange_regions(a,a,SR,seam_s=seam)
        np.testing.assert_array_equal(sham,a)
        zeros=np.zeros_like(a);zero,_=exchange_regions(zeros,zeros,SR,seam_s=seam)
        np.testing.assert_array_equal(zero,zeros)
        self.assertLessEqual(np.max(abs(early)),max(np.max(abs(a)),np.max(abs(b))))

    def test_invalid_seam_and_mixed_group_size_rejected(self):
        with self.assertRaises(ValueError):exchange_regions(take(1),take(2),SR,seam_s=0)
        with self.assertRaises(ValueError):fit_small_groups([SimpleNamespace(group='1')])


class PackageTests(unittest.TestCase):
    def test_roundtrip_binding_tampering_and_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'takes';source.mkdir()
            for i in range(1,4):sf.write(source/f'SHOT 1.{i}.wav',take(i),SR,subtype='PCM_24')
            target=root/'package'
            build_package(source,target,reference_name='SHOT 1.1.wav',donor_name='SHOT 1.2.wav')
            self.assertTrue(verify_package(target)['passed'])
            public=json.loads((target/'experiment/manifest_public.json').read_text(encoding='utf-8'))
            key=json.loads((target/'private/blind_key.json').read_text(encoding='utf-8'))
            rows=[]
            for cid,method in key['conditions'].items():
                good=method in ('natural_donor','donor_early')
                rows.append({'id':cid,'assets':public['comparisons'][cid],'difference':'slight' if good else 'none',
                    'identity':'yes','useful':'slight' if good else 'none','artifacts':'none','sequence':'alternate' if good else 'tie'})
            answer={'protocol':public['protocol'],'package_id':public['package_id'],
                    'stimulus_sha256':public['audio_sha256'],'ratings':rows}
            self.assertEqual(decode_ratings(target,answer)['decision'],'early_supported')
            bad=copy.deepcopy(answer);bad['package_id']='wrong'
            with self.assertRaises(ValueError):decode_ratings(target,bad)
            bad=copy.deepcopy(answer);bad['ratings'][0]['assets']['reference']='wrong.wav'
            with self.assertRaises(ValueError):decode_ratings(target,bad)
            bad=copy.deepcopy(answer);bad['ratings'].pop()
            with self.assertRaises(ValueError):decode_ratings(target,bad)
            bad=copy.deepcopy(answer)
            for row in bad['ratings']:row['useful']='none'
            self.assertEqual(decode_ratings(target,bad)['decision'],'natural_pair_not_sufficient_change_pair_before_synthesis')
            with self.assertRaises(FileExistsError):build_package(source,target)
            wav=target/'experiment/C01.wav'
            a,sr=sf.read(wav,always_2d=True);sf.write(wav,a*.9,sr,subtype='PCM_24')
            self.assertFalse(verify_package(target)['passed'])
            with self.assertRaises(ValueError):decode_ratings(target,answer)


if __name__=='__main__':unittest.main()
