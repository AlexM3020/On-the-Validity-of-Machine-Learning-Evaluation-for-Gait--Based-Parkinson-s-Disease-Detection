"""Simulate destroyed runtime storage and failed cloud transfers, without credentials."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from gait.durable_storage import DirectoryBackend, DurableMirror, connect_storage


class DurableStorageTests(unittest.TestCase):
    def test_recover_after_runtime_deleted_and_keep_previous_generations(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); local=root/'runtime'; remote=root/'durable'; local.mkdir()
            mirror=DurableMirror(local,DirectoryBackend(remote)); mirror.verify()
            for epoch in range(4):
                (local/'state.json').write_text(json.dumps(dict(epoch=epoch)))
                (local/'model.bin').write_bytes(bytes([epoch])*100)
                mirror.publish('epoch', ['state.json','model.bin'])
            self.assertEqual(len(list(remote.glob('commits/*/*.json'))),2)
            # Models/results in runtime are gone, as with a full Colab VM reset.
            shutil.rmtree(local)
            fresh=root/'new-runtime'; recovery=DurableMirror(fresh,DirectoryBackend(remote)); recovery.verify()
            self.assertEqual(recovery.restore(),1)
            self.assertEqual(json.loads((fresh/'state.json').read_text()),dict(epoch=3))
            self.assertEqual((fresh/'model.bin').read_bytes(),bytes([3])*100)

    def test_failed_object_or_marker_does_not_replace_committed_progress(self):
        for fail_marker in (False,True):
            with self.subTest(marker=fail_marker),tempfile.TemporaryDirectory() as d:
                root=Path(d); local=root/'runtime'; local.mkdir(); backend=DirectoryBackend(root/'cloud')
                mirror=DurableMirror(local,backend); mirror.verify()
                (local/'state.json').write_text('old'); mirror.publish('test',['state.json'])
                original=backend.put
                def fail(source,name):
                    if name.startswith('commits/' if fail_marker else 'objects/'): raise RuntimeError('cloud disconnected')
                    return original(source,name)
                (local/'state.json').write_text('new')
                with patch.object(backend,'put',side_effect=fail),self.assertRaisesRegex(RuntimeError,'disconnected'):
                    mirror.publish('test',['state.json'])
                new=DurableMirror(root/'fresh',backend); new.verify(); new.restore()
                self.assertEqual((root/'fresh/state.json').read_text(),'old')

    def test_unverified_storage_and_corrupt_remote_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); local=root/'runtime'; local.mkdir(); (local/'a').write_text('data')
            mirror=DurableMirror(local,DirectoryBackend(root/'cloud'))
            with self.assertRaisesRegex(RuntimeError,'verification'): mirror.publish('k',['a'])
            mirror.verify(); mirror.publish('k',['a'])
            next((root/'cloud/objects').rglob('*.zip')).write_bytes(b'corrupt')
            restored=DurableMirror(root/'new',mirror.backend); restored.verify()
            with self.assertRaisesRegex(ValueError,'corrupted'): restored.restore()

    def test_public_api_rejects_ephemeral_backend(self):
        with self.assertRaisesRegex(ValueError,'ephemeral'):
            connect_storage('/tmp/test','run',backend='local')
        with self.assertRaises(ValueError): connect_storage('/tmp/test','../escape',backend='rclone')


if __name__=='__main__': unittest.main()