"""Verified, synchronous off-runtime checkpoints; no training framework imports.

Snapshots are per epoch/task, not repeatedly compressed copies of the entire run.
The remote commit is published only after the ZIP has been read back and hashed.
Latest and previous snapshots per key are retained. Storage availability is not
a promise of permanent retention by a cloud provider; failures stop training.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('w',encoding='utf-8') as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,path)
    finally: temp.unlink(missing_ok=True)


def safe_relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe checkpoint path')
    return p.as_posix()


class DirectoryBackend:
    def __init__(self,root): self.root=Path(root).resolve()
    def put(self,source,name):
        dest=self.root/safe_relative(name); dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp')
        try: shutil.copyfile(source,temp); os.replace(temp,dest)
        finally: temp.unlink(missing_ok=True)
    def get(self,name,destination): shutil.copyfile(self.root/safe_relative(name),destination)
    def names(self):
        return [] if not self.root.exists() else [p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()]
    def remove(self,name): (self.root/safe_relative(name)).unlink(missing_ok=True)


class RcloneBackend:
    def __init__(self,remote):
        if not re.fullmatch(r'[\w-]+:.+',remote) or remote.startswith(':'):
            raise ValueError('Use a configured rclone remote and folder, e.g. researchdrive:gait/run-id')
        if shutil.which('rclone') is None: raise RuntimeError('rclone is not installed; configure it outside the notebook first')
        self.remote=remote.rstrip('/')
    def command(self,args):
        proc=subprocess.run(['rclone',*args,'--retries','3','--low-level-retries','3'],capture_output=True,text=True)
        if proc.returncode:
            # Avoid exposing remote credentials/config in model or notebook logs.
            raise RuntimeError('Remote checkpoint operation failed. Check rclone access in your terminal; training stopped.')
        return proc.stdout
    def path(self,name): return self.remote+'/'+safe_relative(name)
    def put(self,source,name): self.command(['copyto',str(source),self.path(name)])
    def get(self,name,destination): self.command(['copyto',self.path(name),str(destination)])
    def names(self):
        self.command(['mkdir',self.remote])
        return [s for s in self.command(['lsf',self.remote,'--recursive','--files-only']).splitlines() if s]
    def remove(self,name): self.command(['deletefile',self.path(name)])


class DurableMirror:
    def __init__(self,local_root,backend):
        self.local_root=Path(local_root).resolve(); self.backend=backend
        self.verified=False; self.commits={}

    def verify(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'probe'; source.write_bytes(os.urandom(64))
            name='probes/'+uuid.uuid4().hex
            self.backend.put(source,name)
            restored=Path(d)/'restored'; self.backend.get(name,restored)
            if file_hash(source)!=file_hash(restored): raise RuntimeError('Durable storage round-trip check failed')
            self.backend.remove(name)
        self.verified=True

    def publish(self,key,relative_files):
        if not self.verified: raise RuntimeError('Durable storage has not passed verification')
        files=sorted({safe_relative(str(f)) for f in relative_files})
        key_hash=hashlib.sha256(str(key).encode()).hexdigest()
        previous=self.commits.get(key_hash,[])
        serial=max([int(c['serial']) for c in previous]+[0])+1
        token=f'{serial:012d}-{uuid.uuid4().hex}'
        object_name=f'objects/{key_hash}/{token}.zip'
        marker_name=f'commits/{key_hash}/{token}.json'
        start=time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'snapshot.zip'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                for name in files:
                    source=(self.local_root/name).resolve()
                    if not source.is_relative_to(self.local_root) or not source.is_file(): raise ValueError('Missing checkpoint member')
                    z.write(source,name)
            checksum=file_hash(archive)
            self.backend.put(archive,object_name)
            readback=Path(d)/'readback.zip'; self.backend.get(object_name,readback)
            if file_hash(readback)!=checksum: raise RuntimeError('Remote checkpoint verification failed; no commit published')
            commit=dict(schema=1,key=str(key),key_hash=key_hash,serial=serial,object=object_name,
                        sha256=checksum,files=files,marker=marker_name)
            marker=Path(d)/'commit.json'; atomic_json(marker,commit)
            self.backend.put(marker,marker_name)
            received=Path(d)/'marker-readback'; self.backend.get(marker_name,received)
            if received.read_bytes()!=marker.read_bytes(): raise RuntimeError('Remote commit read-back failed')
        self.commits[key_hash]=previous+[commit]
        # Commit is durable before cleanup. Retain previous generation for recovery.
        for old in self.commits[key_hash][:-2]:
            self.backend.remove(old['marker']); self.backend.remove(old['object'])
        self.commits[key_hash]=self.commits[key_hash][-2:]
        print(f'DURABLE: {key} revision {serial} verified ({time.perf_counter()-start:.1f}s)',flush=True)

    def restore(self):
        """Restore all latest committed units into an EMPTY local workspace."""
        if not self.verified: raise RuntimeError('Verify storage before restoring')
        if self.local_root.exists() and any(self.local_root.iterdir()):
            raise ValueError('Restore target must be empty; existing local state will not be overwritten')
        names=self.backend.names(); grouped={}
        with tempfile.TemporaryDirectory() as d:
            for name in sorted(n for n in names if n.startswith('commits/') and n.endswith('.json')):
                target=Path(d)/'marker'; self.backend.get(name,target)
                c=json.loads(target.read_text())
                if c.get('schema')!=1 or c.get('marker')!=name or hashlib.sha256(c['key'].encode()).hexdigest()!=c['key_hash']:
                    raise ValueError('Invalid durable commit')
                grouped.setdefault(c['key_hash'],[]).append(c)
            self.commits={k:sorted(v,key=lambda c:(int(c['serial']),c['marker'])) for k,v in grouped.items()}
            if any(len({c['serial'] for c in group})!=len(group) for group in self.commits.values()):
                raise ValueError('Concurrent writers detected; use one runtime per run-id')
            staging=Path(d)/'staging'; staging.mkdir()
            owned={}
            for group in self.commits.values():
                c=group[-1]; archive=Path(d)/'restore.zip'; self.backend.get(c['object'],archive)
                if file_hash(archive)!=c['sha256']: raise ValueError('Durable archive corrupted; refusing silent rollback')
                with zipfile.ZipFile(archive) as z:
                    if sorted(z.namelist())!=c['files']: raise ValueError('Archive inventory mismatch')
                    for info in z.infolist():
                        name=safe_relative(info.filename)
                        if (info.external_attr>>16)&0o170000==0o120000: raise ValueError('Archive symlink not permitted')
                        if name in owned and owned[name]!=c['key_hash']: raise ValueError('Checkpoint snapshots have conflicting ownership')
                        owned[name]=c['key_hash']
                    z.extractall(staging)
            self.local_root.mkdir(parents=True,exist_ok=True)
            shutil.copytree(staging,self.local_root,dirs_exist_ok=True)
        print(f'Restored {len(self.commits)} verified checkpoint units from durable storage.',flush=True)
        return len(self.commits)


def connect_storage(local_root,run_id,backend='colab_drive',drive_root='/content/drive/MyDrive/gait-checkpoints',
                    rclone_remote='',mount_drive=True):
    """Public notebook entry: only a mounted Colab Drive or verified rclone remote.

    No secret is read/printed by this code. Configure rclone independently using
    notebook secrets/environment or its interactive authentication, never paste
    tokens in notebook cells. A folder merely named 'drive' is not accepted.
    """
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id): raise ValueError('Run ID must contain only letters, digits, - and _')
    local=Path(local_root).resolve()
    if backend=='colab_drive':
        try: from google.colab import drive
        except ImportError as exc:
            raise RuntimeError('Not a Colab runtime. On Kaggle configure an authenticated rclone remote; temporary /kaggle/working is not durable.') from exc
        if mount_drive: drive.mount('/content/drive')
        mounts=Path('/proc/mounts').read_text()
        if not any(line.split()[1].rstrip('/')=='/content/drive' for line in mounts.splitlines() if len(line.split())>=2):
            raise RuntimeError('Google Drive is not mounted; refusing a long ephemeral run')
        remote=(Path(drive_root)/run_id).resolve()
        if not remote.is_relative_to(Path('/content/drive/MyDrive').resolve()) or local.is_relative_to(remote) or remote.is_relative_to(local):
            raise ValueError('Use separate runtime scratch and mounted MyDrive checkpoint roots')
        impl=DirectoryBackend(remote)
    elif backend=='rclone':
        impl=RcloneBackend(rclone_remote.rstrip('/')+'/'+run_id)
    else:
        raise ValueError('Choose colab_drive or rclone; ephemeral-only research runs are disabled')
    mirror=DurableMirror(local,impl); mirror.verify()
    if local.exists() and any(local.iterdir()):
        raise ValueError('Use a new empty local scratch folder when reconnecting; committed progress will be restored from cloud storage')
    mirror.restore()
    return mirror