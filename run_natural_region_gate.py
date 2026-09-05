"""Reproducible natural-region diagnostic; no stochastic generation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import uuid
from html.parser import HTMLParser

import numpy as np
import scipy
import soundfile as sf
from natural_event_regions import (VERSION, fit_small_groups, select_seam,
                                  exchange_regions, regional_evidence)
from sfx_pool_optimizer import analyze_directory, assemble_sequence

PROTOCOL='natural_region_gate_v1'
BASE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def page_html(public):
    cards=[]
    for cid,pair in public['comparisons'].items():
        cards.append(f'''<section data-id="{cid}"><h2>{cid}</h2>
<div class="players"><div><b>Референс · один выстрел</b><audio controls preload="metadata" src="{pair['reference']}"></audio></div>
<div><b>Вариант · один выстрел</b><audio controls preload="metadata" src="{pair['candidate']}"></audio></div>
<div><b>Повтор референса · 6 выстрелов</b><audio controls preload="metadata" src="{pair['repeat']}"></audio></div>
<div><b>Чередование референса и варианта · 6 выстрелов</b><audio controls preload="metadata" src="{pair['alternate']}"></audio></div></div>
<div class="questions"><label>Различие между одиночными звуками<select data-field="difference"><option value="">Выберите</option><option value="none">Не слышу</option><option value="slight">Слабое</option><option value="clear">Явное</option></select></label>
<label>Вариант сохраняет тот же выстрел?<select data-field="identity"><option value="">Выберите</option><option value="yes">Да</option><option value="uncertain">Не уверен</option><option value="no">Нет</option></select></label>
<label>Полезен как дополнительный дубль?<select data-field="useful"><option value="">Выберите</option><option value="none">Нет</option><option value="slight">Слегка</option><option value="clear">Да</option></select></label>
<label>Новые артефакты в варианте<select data-field="artifacts"><option value="">Выберите</option><option value="none">Нет</option><option value="yes">Есть</option><option value="uncertain">Не уверен</option></select></label>
<label>Какую серию выбрали бы для игры?<select data-field="sequence"><option value="">Выберите</option><option value="repeat">Повтор референса</option><option value="tie">Без предпочтения</option><option value="alternate">Чередование</option></select></label></div>
<label>Комментарий (необязательно)<textarea data-field="comment" rows="2"></textarea></label></section>''')
    data=json.dumps(public,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Из чего складывается новый дубль</title><style>
*{box-sizing:border-box}body{margin:0;background:#111923;color:#edf2f7;font:16px/1.5 system-ui,sans-serif}main{max-width:1040px;margin:auto;padding:26px 20px 80px}h1{font-size:28px;line-height:1.25}h2{margin:0 0 12px;font-size:22px}p{color:#c3ceda}section{padding:22px;margin:22px 0;background:#1b2735;border:1px solid #405269;border-radius:12px}.players,.questions{display:grid;grid-template-columns:1fr 1fr;gap:16px}.questions{margin:20px 0}audio{display:block;width:100%;margin-top:8px}label{display:block}select,textarea{display:block;width:100%;padding:10px;margin-top:5px;color:#edf2f7;background:#111923;border:1px solid #63788e;border-radius:6px;font:inherit}button{background:#92d2be;color:#10271f;border:0;border-radius:8px;padding:14px 18px;font-weight:bold;font-size:16px;cursor:pointer}.notice{padding:12px;border-left:3px solid #92d2be}#status{display:block;margin:12px 0}a{color:#92d2be}@media(max-width:650px){.players,.questions{grid-template-columns:1fr}}
</style></head><body><main><h1>Из чего складывается новый дубль</h1>
<p>Пять сравнений на двух записях из одной группы. Сначала сравните одиночные звуки, затем при необходимости короткие серии. Коды скрывают способ подготовки варианта.</p>
<p class="notice">Это диагностический опыт с натуральными записями. Здесь могут встречаться исходные дубли, точная копия и обработанные варианты. Отметьте отдельно полезное отличие и возможный дефект. Подробную аналитику смотрите после сохранения ответов.</p>
''' + ''.join(cards) + '''<button id="save" type="button">Сохранить ответы</button><span id="status" role="status" aria-live="polite"></span>
</main><script id="experiment-data" type="application/json">'''+data+'''</script><script>
const experiment=JSON.parse(document.getElementById('experiment-data').textContent);
const statusNode=document.getElementById('status');
const storageKey='natural-region:'+experiment.package_id;
let sessionId=crypto.randomUUID?crypto.randomUUID():String(Date.now());
function documentValue(){return {protocol:experiment.protocol,package_id:experiment.package_id,session_id:sessionId,saved_at:new Date().toISOString(),stimulus_sha256:experiment.audio_sha256,ratings:[...document.querySelectorAll('section[data-id]')].map(card=>{const row={id:card.dataset.id,assets:experiment.comparisons[card.dataset.id]};card.querySelectorAll('[data-field]').forEach(el=>row[el.dataset.field]=el.value);return row})}}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(documentValue()));}catch(e){statusNode.textContent='Автосохранение недоступно. Скачайте ответы перед закрытием страницы.'}}
try{const saved=JSON.parse(localStorage.getItem(storageKey)||'null');if(saved&&saved.package_id===experiment.package_id){sessionId=saved.session_id;for(const row of saved.ratings){const card=document.querySelector('section[data-id="'+row.id+'"]');if(card)card.querySelectorAll('[data-field]').forEach(el=>{if(typeof row[el.dataset.field]==='string')el.value=row[el.dataset.field]});}}}catch(e){}
document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('change',persist));
document.querySelectorAll('audio').forEach(el=>el.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(other=>{if(other!==el){other.pause();other.currentTime=0;}})}));
document.getElementById('save').addEventListener('click',()=>{const fields=[...document.querySelectorAll('select[data-field]')];const empty=fields.find(el=>!el.value);if(empty){statusNode.textContent='Заполните все пять ответов в каждом сравнении. Комментарии необязательны.';empty.focus();return;}persist();const blob=new Blob([JSON.stringify(documentValue(),null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='natural_region_ratings_'+experiment.package_id.slice(0,8)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);statusNode.textContent='Ответы сохранены. Прикрепите скачанный JSON к разговору.'});
</script></body></html>'''


class PageInventory(HTMLParser):
    def __init__(self):
        super().__init__();self.audio=[];self.cards=[];self.json_text='';self.capture=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='audio':self.audio.append(a.get('src'))
        if tag=='section':self.cards.append(a.get('data-id'))
        if tag=='script' and a.get('id')=='experiment-data':self.capture=True
    def handle_endtag(self,tag):
        if tag=='script':self.capture=False
    def handle_data(self,data):
        if self.capture:self.json_text+=data


def verify_package(target):
    target=Path(target)
    manifest=json.loads((target/'run_manifest.json').read_text(encoding='utf-8'))
    public=json.loads((target/'experiment/manifest_public.json').read_text(encoding='utf-8'))
    key=json.loads((target/'private/blind_key.json').read_text(encoding='utf-8'))
    failures=[]
    for relative,expected in manifest['output_sha256'].items():
        path=target/relative
        if not path.is_file() or sha(path)!=expected:failures.append('hash:'+relative)
    page=PageInventory();page.feed((target/'experiment/region_gate.html').read_text(encoding='utf-8'))
    if json.loads(page.json_text)!=public:failures.append('page manifest mismatch')
    if sorted(page.cards)!=sorted(public['comparisons']):failures.append('cards mismatch')
    if set(page.audio)!=set(public['audio_sha256']):failures.append('audio inventory mismatch')
    if public['package_id']!=key['package_id']:failures.append('key package mismatch')
    for name,h in public['audio_sha256'].items():
        path=target/'experiment'/name
        if sha(path)!=h:failures.append('audio:'+name)
        data,sr=sf.read(path,always_2d=True);info=sf.info(path)
        if sr!=public['sample_rate'] or info.subtype!='PCM_24' or info.channels!=public['channels']:failures.append('format:'+name)
        if not np.isfinite(data).all() or np.max(np.abs(data))>public['peak_limit']+1e-6:failures.append('samples:'+name)
    for cid,method in key['conditions'].items():
        pair=public['comparisons'][cid]
        if method=='exact_copy':
            if sha(target/'experiment'/pair['reference'])!=sha(target/'experiment'/pair['candidate']):failures.append('sham differs')
    return {'passed':not failures,'failures':failures,'audio_files':len(public['audio_sha256'])}


def decode_ratings(target,document):
    target=Path(target)
    verified=verify_package(target)
    if not verified['passed']:raise ValueError('Package verification failed')
    public=json.loads((target/'experiment/manifest_public.json').read_text(encoding='utf-8'))
    key=json.loads((target/'private/blind_key.json').read_text(encoding='utf-8'))
    if document.get('protocol')!=PROTOCOL or document.get('package_id')!=public['package_id']:
        raise ValueError('Ratings belong to another package')
    if document.get('stimulus_sha256')!=public['audio_sha256']:raise ValueError('Stimulus hashes differ')
    rows=document.get('ratings',[])
    if len(rows)!=5 or {r['id'] for r in rows}!=set(public['comparisons']):raise ValueError('Incomplete or duplicate ratings')
    allowed={'difference':{'none','slight','clear'},'identity':{'yes','no','uncertain'},
             'useful':{'none','slight','clear'},'artifacts':{'none','yes','uncertain'},'sequence':{'repeat','tie','alternate'}}
    decoded={}
    for row in rows:
        if row.get('assets')!=public['comparisons'][row['id']]:raise ValueError('Asset mapping differs')
        for field,values in allowed.items():
            if row.get(field) not in values:raise ValueError('Invalid '+field)
        decoded[key['conditions'][row['id']]]=row
    def useful(method):
        row=decoded[method]
        return row['useful'] in ('slight','clear') and row['identity']=='yes' and row['artifacts']=='none'
    sham=decoded['exact_copy']
    if sham['difference']!='none' or sham['artifacts']!='none':decision='control_inconclusive'
    elif not useful('natural_donor'):decision='natural_pair_not_sufficient_change_pair_before_synthesis'
    elif useful('level_only'):decision='level_is_a_confound_need_level_controlled_confirmation'
    else:
        early,late=useful('donor_early'),useful('donor_late')
        decision=('early_supported' if early and not late else 'late_supported' if late and not early else
                  'both_supported' if early and late else 'regional_exchange_not_supported')
    return {'package_id':public['package_id'],'decoded':decoded,'decision':decision,
            'scope':'one pair, one listener; diagnostic only, not validation of generated takes'}


def build_package(input_dir,target,*,group='1',reference_name='SHOT 1.4.wav',donor_name='SHOT 1.1.wav',seed=5092026):
    target=Path(target).resolve()
    if target.exists():raise FileExistsError(target)
    clips=analyze_directory(Path(input_dir))
    profiles=fit_small_groups(clips)
    bank=[c for c in clips if c.group==group]
    if group not in profiles:raise ValueError('Group not found')
    def find(name):
        matches=[c for c in bank if c.metrics.name==name]
        if len(matches)!=1:raise ValueError('Source not found: '+name)
        return matches[0]
    rc,dc=find(reference_name),find(donor_name)
    if reference_name==donor_name:raise ValueError('Natural pair must contain two different files')
    ref,donor=rc.prepared.astype(float),dc.prepared.astype(float);sr=rc.sample_rate
    seam=select_seam(ref,donor,sr)
    early,late=exchange_regions(ref,donor,sr,seam_s=seam)
    sham,_=exchange_regions(ref,ref,sr,seam_s=seam)
    # Match ONLY total gain of reference to donor as an explicit nuisance control.
    gain=float(np.sqrt(np.sum(donor**2)/np.sum(ref**2)))
    assets={'exact_copy':sham,'natural_donor':donor,'donor_early':early,'donor_late':late,'level_only':ref*gain}
    peak_limit=10**(-1/20)
    reference_loop=assemble_sequence({0:ref},[0]*6,sr,interval_ms=1200)
    loops={m:assemble_sequence({0:ref,1:a},[0,1]*3,sr,interval_ms=1200) for m,a in assets.items()}
    peak=max(np.max(abs(ref)),np.max(abs(reference_loop)),*(np.max(abs(a)) for a in [*assets.values(),*loops.values()]))
    scale=float(min(1.,peak_limit/peak))
    rng=np.random.default_rng(seed);methods=list(assets);rng.shuffle(methods)
    package_id=str(uuid.uuid4())
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.natural-region-',dir=target.parent) as tmp:
        staging=Path(tmp)/'package';staging.mkdir()
        exp=staging/'experiment';exp.mkdir();private=staging/'private';private.mkdir();analysis=staging/'analysis';analysis.mkdir()
        def write_audio(name,a):sf.write(exp/name,a*scale,sr,subtype='PCM_24')
        write_audio('reference.wav',ref);write_audio('repeat.wav',reference_loop)
        comparisons={};conditions={}
        for i,method in enumerate(methods,1):
            cid=f'C{i:02d}';write_audio(cid+'.wav',assets[method]);write_audio(cid+'_sequence.wav',loops[method])
            comparisons[cid]={'reference':'reference.wav','candidate':cid+'.wav','repeat':'repeat.wav','alternate':cid+'_sequence.wav'}
            conditions[cid]=method
        public={'protocol':PROTOCOL,'package_id':package_id,'sample_rate':sr,'channels':ref.shape[1],
                'peak_limit':peak_limit,'comparisons':comparisons,
                'audio_sha256':{p.name:sha(p) for p in sorted(exp.glob('*.wav'))}}
        write_json(exp/'manifest_public.json',public)
        (exp/'region_gate.html').write_text(page_html(public),encoding='utf-8')
        write_json(private/'blind_key.json',{'package_id':package_id,'conditions':conditions,'reference':reference_name,'donor':donor_name})
        # Analyze exported samples, not a different pre-normalization signal.
        actual_ref,_=sf.read(exp/'reference.wav',always_2d=True)
        actual_metrics={}
        for cid,method in conditions.items():
            actual,_=sf.read(exp/(cid+'.wav'),always_2d=True)
            actual_metrics[method]=regional_evidence(actual_ref,actual,sr,seam_s=seam)
        write_json(analysis/'measurements.json',{'groups':{g:p.summary() for g,p in profiles.items()},
                    'selected_group':group,'seam_s':seam,'fade_s':.006,'global_common_gain':scale,
                    'level_control_gain':gain,'export_metrics':actual_metrics,
                    'purpose':'localize useful variation; no automatic perceptual acceptance threshold'})
        impl={f:sha(BASE/f) for f in ('natural_event_regions.py','run_natural_region_gate.py','sfx_pool_optimizer.py')}
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=BASE,capture_output=True,text=True)
        manifest={'protocol':PROTOCOL,'package_id':package_id,'version':VERSION,'implementation_sha256':impl,
                  'git_head':git.stdout.strip(),'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':sf.__version__},
                  'settings':{'group':group,'reference':reference_name,'donor':donor_name,'seed':seed,'seam_s':seam,'fade_s':.006,'events':6,'interval_ms':1200},
                  'sources':[{'path':str(c.path.resolve()),'sha256':sha(c.path),'group':c.group} for c in clips],
                  'output_sha256':{str(p.relative_to(staging)).replace('\\','/'):sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}}
        write_json(staging/'run_manifest.json',manifest)
        result=verify_package(staging)
        if not result['passed']:raise RuntimeError(result)
        write_json(staging/'verification_report.json',result)
        staging.rename(target)
    return {'target':str(target),'verification':result,'seam_ms':1000*seam,'measurements':actual_metrics}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path('references/group_1'))
    parser.add_argument('--group',default='1')
    parser.add_argument('--reference',default='SHOT 1.4.wav')
    parser.add_argument('--donor',default='SHOT 1.1.wav')
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--ratings',type=Path)
    args=parser.parse_args()
    if args.ratings:
        document=json.loads(args.ratings.read_text(encoding='utf-8'))
        result=decode_ratings(args.results_dir,document)
        output=args.results_dir/('ratings_decoded_'+sha(args.ratings)[:12]+'.json')
        if output.exists():raise FileExistsError(output)
        write_json(output,result);print(result['decision']);print(output)
    else:
        result=build_package(args.input_dir,args.results_dir,group=args.group,reference_name=args.reference,donor_name=args.donor)
        print('Verified:',result['verification']);print('Seam ms:',result['seam_ms'])
        print(Path(result['target'])/'experiment/region_gate.html')


if __name__=='__main__':main()
