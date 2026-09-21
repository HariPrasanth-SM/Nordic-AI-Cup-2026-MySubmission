"""Run with pytest. Uses generated pixels; no challenge images or score claims."""
import json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
import build_balanced as b


def test_geometry_and_scale():
    im=np.zeros((540,960,3),np.uint8); im[100:160,100:220]=255
    rows=[{'class_id':0,'origin':'original','bbox':[100,100,220,160]}]
    transformed,rr,flags=b.scene_transform(im,rows,random.Random(4),True)
    assert flags['rotation_degrees']==180
    assert rr[0]['bbox']==[740,380,860,440]
    im,rr=b.downsample(transformed,rr,2)
    assert rr[0]['bbox']==[370,190,430,220]
    assert (im[190:220,370:430]==255).all()


def test_build_quotas_and_no_leakage(tmp_path):
    root=tmp_path/'source'
    for d in ('images','annotations','target_objects'): (root/d).mkdir(parents=True)
    for i in range(3):
        # Nonuniform backgrounds make duplicate checks meaningful.
        im=np.random.default_rng(i).integers(0,255,(2160,3840,3),dtype=np.uint8)
        Image.fromarray(im).save(root/'images'/f'f{i}.jpg')
        (root/'annotations'/f'f{i}.json').write_text(json.dumps({'frame':i,'annotations':[{'object_id':'hangar','bbox':[100,100,220,160]}]}))
    asset=np.zeros((30,50,4),np.uint8); asset[2:-2,2:-2]=[180,80,50,255]
    for c in b.CLASSES: Image.fromarray(asset).save(root/'target_objects'/f'{c}.png')
    from argparse import Namespace
    a=Namespace(root=root,output=tmp_path/'out',val_frames=1,gap_frames=1,mode='build',annotations_complete=True,seed=42,
                train_count=40,val_count=40,negative_fraction=.125,min_train_instances=3,min_val_instances=3,
                preview_positives=10,preview_negatives=3,min_objects=2,max_objects=5,gap=8,negative_margin=8,real_fraction=.15)
    b.build(a)
    summary=json.loads((a.output/'summary.json').read_text())
    for s in ('train','val'):
        for l in (1,2):
            r=summary[f'{s}_L{l}']; assert r['negative']==5 and min(r['instances'].values())>=3
    assert (a.output/'COMPLETE').exists()


def test_final_tile_conditioning_matches_camera_renderer():
    base=np.random.default_rng(7).integers(0,256,(2160,3840,3),dtype=np.uint8)
    f={'id':'fixture','objects':[{'class_id':0,'origin':'original','bbox':[1200,900,1320,960]}]}
    for lev in (1,2):
        patch,rows,meta=b.crop_source(f,base,lev,random.Random(11))
        actual,labels=b.downsample(patch,rows,meta['source_pixels_per_output_pixel'])
        expected=next(x for x in b.render(base,f['objects'],lev,meta['camera_window']) if x[0]==meta['tile_index'])
        np.testing.assert_array_equal(actual,expected[2])
        assert labels==expected[3]
