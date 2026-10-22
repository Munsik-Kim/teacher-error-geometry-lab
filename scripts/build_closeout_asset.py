#!/usr/bin/env python3
"""Refresh current appendix inventories and build/restore its CPU release asset.

This stdlib tool never imports scientific modules, fits a head, reads a model or
rewrites historical experiment/evidence freezes. The archive is written outside
the repository; only publication/closeout_asset.json records its identity here.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

REPOSITORY=Path(__file__).resolve().parents[1]
APPENDIX=Path('followups/linear_feature_closeout')
TOP='linear_feature_closeout'
CAP=100_000_000
GENERATED={'source_manifest.json','config_manifest.json','input_manifest.json',
           'result_manifest.json','package_manifest.json'}
EXCLUDED={'.git','.venv','.gpu-venv','.cache','.mplcache','__pycache__'}
FORBIDDEN={'models','model_staging','cache','publication_snapshot'}


def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(data): return json.dumps(data,sort_keys=True,separators=(',',':')).encode()
def encoded(data): return (json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode()
def record(root,path):
    data=path.read_bytes()
    return dict(path=path.relative_to(root).as_posix(),bytes=len(data),sha256=digest(data))


def members(root):
    files=[]
    for path in root.rglob('*'):
        rel=path.relative_to(root)
        if EXCLUDED.intersection(rel.parts): continue
        if FORBIDDEN.intersection(rel.parts): raise ValueError('FORBIDDEN_DIRECTORY:'+str(rel))
        if path.is_symlink(): raise ValueError('SYMLINK_NOT_ALLOWED:'+str(rel))
        if path.is_file(): files.append(path)
    return sorted(files,key=lambda p:p.relative_to(root).as_posix())


def scope_objects(root):
    paths=[p for p in members(root) if p.relative_to(root).as_posix() not in GENERATED]
    definitions={
        'source_manifest.json':lambda n:n.startswith(('src/','scripts/','tests/')),
        'config_manifest.json':lambda n:n in {'CLOSURE_PROTOCOL.json','requirements.txt','requirements-lock.txt'},
        'input_manifest.json':lambda n:n.startswith('inputs/'),
        'result_manifest.json':lambda n:n.startswith(('results/','audit/','figures/')),
    }
    result={}
    for name,condition in definitions.items():
        rows=[record(root,p) for p in paths if condition(p.relative_to(root).as_posix())]
        result[name]=dict(scope='v0.2.0 current public appendix files; not a historical execution freeze',
            files=rows,bundle_sha256=digest(canonical(rows)))
    return result


def refresh(repository,check=False):
    root=repository/APPENDIX
    for required in ['CLOSURE_PROTOCOL.json','facts.json','inputs/data.npz',
                     'requirements-lock.txt','results/closure_posthoc_01.source_freeze.json']:
        if not (root/required).is_file(): raise ValueError('MISSING_APPENDIX_INPUT:'+required)
    scopes=scope_objects(root)
    for name,value in scopes.items():
        path=root/name
        if check:
            if json.loads(path.read_text())!=value: raise ValueError('CURRENT_SCOPE_MISMATCH:'+name)
        else: path.write_bytes(encoded(value))
    rows=[record(root,p) for p in members(root) if p.name!='package_manifest.json']
    package=dict(schema='TEG-linear-feature-closeout-package-v1',top_directory=TOP,
        manifest_self_hash_excluded=True,content_bundle_sha256=digest(canonical(rows)),files=rows,
        scientific_scope='stored PCA64 inputs; four posthoc CPU OLS fits; no GPU reproduction',
        identity='v0.2.0 current public documentation and portable inventory; historical execution snapshot unchanged')
    if check:
        if json.loads((root/'package_manifest.json').read_text())!=package:
            raise ValueError('CURRENT_PACKAGE_MANIFEST_MISMATCH')
    else: (root/'package_manifest.json').write_bytes(encoded(package))
    return dict(status='PASS',mode='check' if check else 'refresh',appendix=str(APPENDIX),
                files=len(rows)+1,content_bundle_sha256=package['content_bundle_sha256'],
                scope_files={name:len(value['files']) for name,value in scopes.items()})


def add_file(archive,name,data):
    info=tarfile.TarInfo(f'{TOP}/{name}')
    info.mode=0o644;info.uid=info.gid=0;info.mtime=0;info.size=len(data)
    archive.addfile(info,io.BytesIO(data))


def build(repository,output):
    status=refresh(repository,check=True)
    root=repository/APPENDIX;output=Path(output).absolute()
    if output.is_relative_to(repository): raise ValueError('RELEASE_ASSET_MUST_BE_OUTSIDE_REPOSITORY')
    if output.exists() or output.with_suffix(output.suffix+'.sha256').exists():
        raise FileExistsError('RELEASE_ASSET_EXISTS')
    executed=json.loads((root/'results/closure_posthoc_01.source_freeze.json').read_text())
    execution_bundle=executed['source_bundle_sha256']
    files=members(root);total=sum(p.stat().st_size for p in files)
    if total>CAP: raise ValueError('UNCOMPRESSED_ASSET_BUDGET_EXCEEDED')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial')
    if temporary.exists(): raise FileExistsError('PARTIAL_ASSET_EXISTS')
    try:
        with temporary.open('xb') as file:
            with gzip.GzipFile(filename='',mode='wb',fileobj=file,mtime=0) as compressed:
                with tarfile.open(fileobj=compressed,mode='w',format=tarfile.USTAR_FORMAT) as archive:
                    for path in files: add_file(archive,path.relative_to(root).as_posix(),path.read_bytes())
        if temporary.stat().st_size>CAP: raise ValueError('COMPRESSED_ASSET_BUDGET_EXCEEDED')
        temporary.rename(output)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise
    identity=dict(identity='v0.2.0 CPU standalone release asset; distinct from executed source/evidence hashes',
        filename=output.name,bytes=output.stat().st_size,sha256=digest(output.read_bytes()),
        member_count=len(files),uncompressed_bytes=total,archive_top_directory=TOP,
        appendix_content_bundle_sha256=status['content_bundle_sha256'],
        execution_source_bundle_sha256=execution_bundle,
        raw_or_scientific_fits_modified=False)
    sha_path=output.with_suffix(output.suffix+'.sha256')
    with sha_path.open('x') as file: file.write(f"{identity['sha256']}  {output.name}\n")
    (repository/'publication/closeout_asset.json').write_bytes(encoded(identity))
    return dict(status='PASS',archive=str(output),sha256_file=str(sha_path),**identity)


def restore(archive_path,output):
    archive_path=Path(archive_path).resolve();output=Path(output).absolute()
    if output.exists(): raise FileExistsError('RESTORE_OUTPUT_EXISTS')
    with tarfile.open(archive_path,'r:gz') as archive:
        files=archive.getmembers();seen=set();total=0
        for file in files:
            rel=PurePosixPath(file.name)
            if (not file.isfile() or rel.is_absolute() or file.name!=rel.as_posix()
                or len(rel.parts)<2 or rel.parts[0]!=TOP or '..' in rel.parts
                or '\\' in file.name or file.name in seen or EXCLUDED.intersection(rel.parts)
                or FORBIDDEN.intersection(rel.parts)):
                raise ValueError('UNSAFE_ASSET_MEMBER:'+file.name)
            seen.add(file.name);total+=file.size
            if total>CAP: raise ValueError('RESTORE_BUDGET_EXCEEDED')
        manifest_name=f'{TOP}/package_manifest.json'
        if manifest_name not in seen: raise ValueError('PACKAGE_MANIFEST_MISSING')
        manifest=json.loads(archive.extractfile(manifest_name).read())
        if (manifest.get('schema')!='TEG-linear-feature-closeout-package-v1'
            or manifest.get('top_directory')!=TOP
            or manifest.get('manifest_self_hash_excluded') is not True):
            raise ValueError('ASSET_MANIFEST_SCHEMA_MISMATCH')
        records=manifest['files'];expected={f"{TOP}/{r['path']}":r for r in records}
        if len(expected)!=len(records) or set(expected)|{manifest_name}!=seen:
            raise ValueError('ASSET_MEMBER_INVENTORY_MISMATCH')
        if digest(canonical(records))!=manifest['content_bundle_sha256']:
            raise ValueError('ASSET_BUNDLE_MISMATCH')
        for file in files:
            if file.name==manifest_name: continue
            data=archive.extractfile(file).read();recorded=expected[file.name]
            if len(data)!=recorded['bytes'] or digest(data)!=recorded['sha256']:
                raise ValueError('ASSET_MEMBER_HASH_MISMATCH:'+file.name)
        output.mkdir(parents=True)
        for file in files:
            destination=output.joinpath(*PurePosixPath(file.name).parts)
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as target: target.write(archive.extractfile(file).read())
    return dict(status='PASS',restored_root=str(output/TOP),files=len(files),
                archive_sha256=digest(archive_path.read_bytes()),
                manifest_bundle_sha256=manifest['content_bundle_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    ref=sub.add_parser('refresh');ref.add_argument('--check',action='store_true')
    b=sub.add_parser('build');b.add_argument('--output',required=True)
    e=sub.add_parser('extract');e.add_argument('--archive',required=True);e.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='refresh': result=refresh(REPOSITORY,args.check)
    elif args.command=='build': result=build(REPOSITORY,args.output)
    else: result=restore(args.archive,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
