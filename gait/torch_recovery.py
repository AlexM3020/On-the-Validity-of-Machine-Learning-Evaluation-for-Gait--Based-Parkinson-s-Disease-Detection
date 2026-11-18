"""Small, restricted-load PyTorch epoch snapshots, mirrored before continuing."""
import json
import os
from pathlib import Path
import time
import uuid

import torch
from .durable_storage import atomic_json, file_hash


class TorchRecovery:
    def __init__(self,folder,mirror=None):
        self.folder=Path(folder); self.mirror=mirror

    def load(self):
        pointer=self.folder/'state.json'
        if not pointer.exists(): return None
        meta=json.loads(pointer.read_text())
        path=self.folder/meta['file']
        if file_hash(path)!=meta['sha256']: raise ValueError('Corrupt PyTorch recovery checkpoint')
        return torch.load(path,map_location='cpu',weights_only=True)

    def save(self,payload):
        self.folder.mkdir(parents=True,exist_ok=True)
        generation='epoch-'+uuid.uuid4().hex+'.pt'
        path=self.folder/generation
        with path.open('wb') as stream:
            torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        pointer=self.folder/'state.json'
        atomic_json(pointer,dict(file=generation,sha256=file_hash(path),phase=payload['phase'],epoch=payload.get('epoch')))
        if self.mirror is not None:
            root=self.mirror.local_root
            self.mirror.publish('torch/'+self.folder.relative_to(root).as_posix(),
                                [path.relative_to(root).as_posix(),pointer.relative_to(root).as_posix()])
        for old in self.folder.glob('epoch-*.pt'):
            if old!=path: old.unlink()


def capture_rng(generator):
    return dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader_rng=generator.get_state())


def restore_rng(state,generator):
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    generator.set_state(state['loader_rng'])