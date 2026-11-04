"""Repackage only upstream wheel dependency metadata for the validated torch pair.

All retinaface Python sources are preserved byte-for-byte. This local version
avoids a permanently inconsistent environment caused by the training-only pins.
"""
import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile
from urllib.request import urlopen

SOURCE_URL='https://files.pythonhosted.org/packages/16/a7/0d59e76c03d7446ea4d47f17e05bdd1153ce3259c84b883d903d5d2f4da8/retinaface_pytorch-0.0.8-py2.py3-none-any.whl'
SOURCE_SHA256='bf64dd6562e7dd0342c7408e086c57b675bd15cf759f51def9737b31cd581de7'
VERSION='0.0.8+torch27'


def build(source: bytes,output: Path):
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:raise ValueError('Upstream wheel hash mismatch')
    old='retinaface_pytorch-0.0.8.dist-info';new=f'retinaface_pytorch-{VERSION}.dist-info'
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        files={name.replace(old,new):archive.read(name) for name in archive.namelist() if not name.endswith('/RECORD')}
    key=new+'/METADATA';metadata=files[key].decode('utf-8')
    for before,after in [('Version: 0.0.8','Version: '+VERSION),
                         ('torch (==1.9.0)','torch (==2.7.1)'),('torchvision (==0.10.0)','torchvision (==0.22.1)')]:
        if before not in metadata:raise ValueError('Unexpected upstream dependency metadata')
        metadata=metadata.replace(before,after)
    files[key]=metadata.encode('utf-8')
    records=[]
    for name,payload in sorted(files.items()):
        digest=base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')
        records.append((name,'sha256='+digest,str(len(payload))))
    records.append((new+'/RECORD','',''))
    text=io.StringIO(newline='');csv.writer(text,lineterminator='\n').writerows(records)
    files[new+'/RECORD']=text.getvalue().encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,payload in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,11,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,payload)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.source:source=args.source.read_bytes()
    elif args.download:
        with urlopen(SOURCE_URL,timeout=30) as response:source=response.read(1024*1024)
    else:parser.error('Provide --source or explicitly request --download')
    output=args.output_dir/f'retinaface_pytorch-{VERSION}-py2.py3-none-any.whl'
    if output.exists():
        # Reproducibility is checked against a fresh deterministic build.
        temp=output.with_name('check-'+output.name)
        if temp.exists():raise ValueError('Compatibility check temporary file already exists')
        try:
            build(source,temp)
            if output.read_bytes()!=temp.read_bytes():raise ValueError('Existing compatibility wheel mismatch')
        finally:
            if temp.exists():temp.unlink()
    else:build(source,output)
    print(output.resolve())


if __name__=='__main__':main()
