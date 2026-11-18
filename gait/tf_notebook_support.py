"""Keras 3-compatible LPGNet-family models and atomic epoch/task checkpoints.

No GPU training, downloads or output writes on import. Scientific settings are
supplied by each notebook; leaky factorial conditions remain explicitly labelled.
"""
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import uuid

import numpy as np
import pandas as pd
import scipy
from scipy.signal import lfilter
import sklearn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import official_data
from .durable_storage import connect_storage


@keras.utils.register_keras_serializable(package="GaitSafety")
class WeightedSum(layers.Layer):
    def call(self, inputs):
        values, weights = inputs
        if weights.shape.rank == 2:
            weights = weights[..., None]
        return tf.reduce_sum(values * tf.cast(weights, values.dtype), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


@keras.utils.register_keras_serializable(package="GaitSafety")
class MaskedSoftmax(layers.Layer):
    """Serializable masked gated-pooling weights, not Q/K/V self-attention."""
    def call(self, inputs):
        logits, mask = inputs
        logits = tf.cast(tf.squeeze(logits, axis=-1), tf.float32)
        mask = tf.cast(mask, tf.bool)
        tf.debugging.assert_positive(tf.reduce_sum(tf.cast(mask, tf.int32), axis=1),
                                     message="Every recording needs at least one valid window")
        masked = tf.where(mask, logits, tf.constant(-1e9, tf.float32))
        weights = tf.nn.softmax(masked, axis=1)
        weights = tf.where(mask, weights, tf.zeros_like(weights))
        return tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        return input_shape[1]


@keras.utils.register_keras_serializable(package="GaitSafety")
class ValidWindowEncoder(layers.Layer):
    """Encode ONLY real windows: padded windows never enter BatchNorm/dropout."""
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.embedding_dim = int(encoder.output_shape[-1])

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        windows, mask = inputs
        mask = tf.cast(mask, tf.bool)
        tf.debugging.assert_positive(tf.reduce_sum(tf.cast(mask, tf.int32), axis=1),
                                     message="Every recording needs a valid window")
        shape = tf.shape(windows)
        flattened = tf.reshape(windows, (-1, shape[2], shape[3]))
        valid_mask = tf.reshape(mask, (-1,))
        values = tf.boolean_mask(flattened, valid_mask)
        values = tf.ensure_shape(values, (None, self.encoder.input_shape[1], self.encoder.input_shape[2]))
        embedded = self.encoder(values, training=training if self.encoder.trainable else False)
        indices = tf.cast(tf.where(valid_mask), tf.int32)
        scattered = tf.scatter_nd(indices, embedded, tf.stack([shape[0]*shape[1], self.embedding_dim]))
        return tf.reshape(scattered, (shape[0], shape[1], self.embedding_dim))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.embedding_dim)

    def get_config(self):
        return {**super().get_config(), "encoder":keras.utils.serialize_keras_object(self.encoder)}

    @classmethod
    def from_config(cls, config):
        config["encoder"] = keras.utils.deserialize_keras_object(config["encoder"])
        return cls(**config)


def sepconv(filters, kernel_size, reg, padding="same", dilation_rate=1, name=None):
    # Explicit supported API. Do not mislabel this as legacy kernel_regularizer bit parity.
    return layers.SeparableConv1D(filters, kernel_size, padding=padding, dilation_rate=dilation_rate,
                activation="linear", depthwise_regularizer=reg, pointwise_regularizer=reg, name=name)


def se_block(x, ratio, prefix):
    ch = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D(name=prefix+"_squeeze")(x)
    s = layers.Dense(max(ch//ratio,4), activation="relu", name=prefix+"_reduce")(s)
    s = layers.Dense(ch, activation="sigmoid", name=prefix+"_expand")(s)
    return layers.Multiply(name=prefix+"_scale")([x, layers.Reshape((1,ch))(s)])


def build_lpgnet(T=None, n_ch=18, channels=32, kernels=(7,5,3), spat_drop=.2,
                 drop=.25, l2=1e-3, noise=.01):
    reg = keras.regularizers.l2(l2)
    widths = [channels]*(len(kernels)-1)+[2*channels] if isinstance(channels,int) else channels
    inp = layers.Input((None,n_ch))
    x = layers.GaussianNoise(noise)(layers.SpatialDropout1D(spat_drop)(inp))
    for i,(k,c) in enumerate(zip(kernels,widths)):
        x = sepconv(c,k,reg,padding="valid")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("elu",name="embedding" if i==len(kernels)-1 else None)(x)
        if i < len(kernels)-1:
            x = layers.MaxPool1D(2,2)(x)
    x = layers.GlobalAveragePooling1D(name="pooled")(x)
    out = layers.Dense(1,activation="sigmoid",kernel_regularizer=reg,name="det")(layers.Dropout(drop)(x))
    return keras.Model(inp,out,name="LPGNet_compatible")


def build_variantA(n_ch=18, channels=32, kernels=(7,5,3), se_ratio=8,
                   spat_drop=.2, drop=.25, l2=1e-3, noise=.01):
    reg = keras.regularizers.l2(l2)
    widths = [channels]*(len(kernels)-1)+[2*channels] if isinstance(channels,int) else channels
    inp = layers.Input((None,n_ch))
    x = layers.GaussianNoise(noise)(layers.SpatialDropout1D(spat_drop)(inp))
    for i,(k,c) in enumerate(zip(kernels,widths)):
        x = layers.Activation("elu")(layers.BatchNormalization()(sepconv(c,k,reg,"valid")(x)))
        x = se_block(x,se_ratio,f"se{i}")
        if i < len(kernels)-1:
            x = layers.MaxPool1D(2,2)(x)
    weights = layers.Softmax(axis=1)(layers.Dense(1)(x))
    x = WeightedSum(name="pooled")([x,weights])
    out = layers.Dense(1,activation="sigmoid",kernel_regularizer=reg)(layers.Dropout(drop)(x))
    return keras.Model(inp,out,name="VariantA")


def build_recording_model(window_model):
    backbone = keras.Model(window_model.input,window_model.get_layer("pooled").output)
    backbone.trainable = False
    inp = layers.Input((None,18))
    return keras.Model(inp,layers.Dense(1,activation="sigmoid",name='recording_pd_probability')(
        layers.Dropout(.25,name='recording_dropout')(backbone(inp))))


def build_window_encoder(cfg, win=100, n_ch=18):
    reg, ch = keras.regularizers.l2(cfg["l2"]), cfg["channels"]
    inp = layers.Input((win,n_ch),name="window_input")
    x = layers.GaussianNoise(.01,name="input_gaussian_noise")(layers.SpatialDropout1D(cfg["spat_drop"],name="input_spatial_dropout")(inp))
    branches = []
    for j,k in enumerate(cfg["kernels"]):
        b = sepconv(max(ch//2,16),k,reg,name=f"multiscale_{j}_sep")(x)
        b = layers.Activation("elu",name=f"multiscale_{j}_elu")(layers.BatchNormalization(name=f"multiscale_{j}_bn")(b))
        branches.append(b)
    x = layers.Concatenate(name="multiscale_concat")(branches)
    x = layers.Conv1D(ch,1,padding="same",kernel_regularizer=reg,name="multiscale_projection")(x)
    x = layers.Activation("elu",name="multiscale_elu")(layers.BatchNormalization(name="multiscale_bn")(x))
    x = se_block(x,cfg["se_ratio"],"multiscale_se")
    for block,dilation in enumerate((1,2,4)):
        prefix=f"block{block}"
        y=sepconv(ch,5,reg,dilation_rate=dilation,name=prefix+"_sep")(x)
        y=layers.Activation("elu",name=prefix+"_pre_elu")(layers.BatchNormalization(name=prefix+"_bn")(y))
        y=se_block(y,cfg["se_ratio"],prefix+"_se")
        x=layers.Activation("elu",name=prefix+"_elu")(layers.Add(name=prefix+"_add")([x,y]))
    scores=layers.Dense(1,name="window_temporal_scores")(x)
    weights=layers.Softmax(axis=1,name="window_temporal_softmax")(scores)
    x=WeightedSum(name="window_temporal_sum")([x,weights])
    x=layers.Dense(cfg["embedding_dim"],activation="elu",kernel_regularizer=reg,name="window_embedding")(x)
    return keras.Model(inp,x,name="window_encoder")


def build_window_classifier(cfg, win=100):
    inp=layers.Input((win,18),name="window_classifier_input")
    x=build_window_encoder(cfg,win)(inp)
    out=layers.Dense(1,activation="sigmoid",kernel_regularizer=keras.regularizers.l2(cfg["l2"]),
                     name="window_pd_probability")(layers.Dropout(cfg["drop"])(x))
    return keras.Model(inp,out,name="window_classifier")


def build_mil_model(window_model,cfg,win=100):
    encoder=window_model.get_layer("window_encoder")
    if cfg["fine_tune"] == "frozen":
        encoder.trainable=False
    elif cfg["fine_tune"] == "last":
        encoder.trainable=True
        for layer in encoder.layers:
            layer.trainable=layer.name.startswith("block2_") or layer.name in {"window_temporal_scores","window_embedding"}
    else:
        raise ValueError("Unknown fine_tune mode")
    reg=keras.regularizers.l2(cfg["l2"])
    windows=layers.Input((None,win,18),name="recording_windows")
    mask=layers.Input((None,),dtype="bool",name="recording_window_mask")
    h=ValidWindowEncoder(encoder,name="shared_window_encoder")([windows,mask])
    ah=max(cfg["embedding_dim"]//2,16)
    a=layers.Dense(ah,activation="tanh",kernel_regularizer=reg,name="recording_attention_tanh")(h)
    g=layers.Dense(ah,activation="sigmoid",kernel_regularizer=reg,name="recording_attention_gate")(h)
    a=layers.Multiply(name="recording_attention_gated")([a,g])
    logits=layers.Dense(1,kernel_regularizer=reg,name="recording_attention_logits")(a)
    weights=MaskedSoftmax(name="window_attention_remask")([logits,mask])
    z=WeightedSum(name="recording_embedding")([h,weights])
    z=layers.Dropout(cfg["drop"],name="recording_dropout")(z)
    out=layers.Dense(1,activation="sigmoid",kernel_regularizer=reg,name="recording_pd_probability")(z)
    return keras.Model([windows,mask],out,name="LP_ResMIL")


def json_ready(value):
    if isinstance(value,dict): return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [json_ready(v) for v in value]
    if isinstance(value,np.ndarray): return json_ready(value.tolist())
    if isinstance(value,np.generic): return json_ready(value.item())
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_ready(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("w",encoding="utf-8") as stream:
            json.dump(json_ready(value),stream,sort_keys=True,allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


class RunStore:
    """Immutable run plan; one atomic JSON transaction per completed task."""
    def __init__(self,root,plan,mirror=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.mirror=mirror
        self.plan=json_ready(plan); self.signature=digest(self.plan)
        path=self.root/"run_plan.json"
        if path.exists():
            if json.loads(path.read_text()) != self.plan:
                raise ValueError("Checkpoint plan mismatch: use a new output directory; no old results were changed")
        else:
            if any(self.root.iterdir()):
                raise ValueError("Output directory contains legacy/unverified checkpoints; choose a new directory")
            atomic_json(path,self.plan)
        if self.mirror is not None:
            self.mirror.publish('run-plan',['run_plan.json'])

    def folder(self,key):
        return self.root/"tasks"/digest(key)

    def load(self,key):
        path=self.folder(key)/"complete.json"
        if not path.exists(): return None
        saved=json.loads(path.read_text())
        if saved["plan_signature"]!=self.signature or saved["key"]!=json_ready(key) or saved["payload_hash"]!=digest(saved["payload"]):
            raise ValueError("Corrupt or mismatched task checkpoint")
        if self.mirror is not None:
            # A previous transfer may have failed after the local task committed.
            self.mirror.publish('task/'+digest(key),[path.relative_to(self.root).as_posix()])
        return saved["payload"]

    def save(self,key,payload):
        payload=json_ready(payload)
        atomic_json(self.folder(key)/"complete.json",dict(plan_signature=self.signature,key=key,
                    payload=payload,payload_hash=digest(payload)))
        if self.mirror is not None:
            self.mirror.publish('task/'+digest(key),[(self.folder(key)/'complete.json').relative_to(self.root).as_posix()])


class Batches:
    """Bounded-memory deterministic batches; avoids Keras 2/3 Sequence workers API."""
    def __init__(self,records,indices,features,kind,win=100,stride=50,batch=8,seed=1,shuffle=False,balanced=False):
        self.records,self.indices,self.features=records,list(indices),features
        self.kind,self.win,self.stride,self.batch,self.seed,self.shuffle=kind,win,stride,batch,seed,shuffle
        self.pairs=[(i,st) for i in self.indices for st in range(0,len(features[i])-win,stride)] if kind=="window" else None
        self.items=self.pairs if kind=="window" else self.indices
        if not self.items: raise ValueError("Empty training/evaluation partition")
        self.length=max(len(features[i]) for i in self.indices)
        counts=Counter(records[i]["subject"] for i in self.indices)
        window_counts={i:len(range(0,len(features[i])-win,stride)) for i in self.indices}
        if any(v<1 for v in window_counts.values()): raise ValueError("Recording shorter than one complete window")
        weights=[]
        for item in self.items:
            i=item[0] if kind=="window" else item
            weights.append(1/counts[records[i]["subject"]]/(window_counts[i] if kind=="window" else 1))
        self.weights=np.array(weights,dtype=np.float32) if balanced else None
        if balanced: self.weights/=self.weights.mean()
        self.order=np.arange(len(self.items))

    def set_epoch(self,epoch):
        self.order=np.arange(len(self.items))
        if self.shuffle: self.order=np.random.default_rng((self.seed+epoch)%2**32).permutation(self.order)

    def __len__(self): return int(np.ceil(len(self.items)/self.batch))

    def __getitem__(self,j):
        ids=self.order[j*self.batch:(j+1)*self.batch]
        items=[self.items[int(k)] for k in ids]
        if self.kind=="window":
            X=np.stack([self.features[i][st:st+self.win] for i,st in items]).astype(np.float32)
            indices=[i for i,_ in items]
        elif self.kind=="recording":
            indices=items
            X=np.stack([np.pad(self.features[i],((0,self.length-len(self.features[i])),(0,0)),mode="wrap") for i in indices]).astype(np.float32)
        elif self.kind=="mil":
            indices=items
            arrays=[np.stack([self.features[i][st:st+self.win] for st in range(0,len(self.features[i])-self.win,self.stride)]) for i in indices]
            max_n=max(len(a) for a in arrays)
            values=np.zeros((len(indices),max_n,self.win,18),np.float32); mask=np.zeros((len(indices),max_n),bool)
            for k,a in enumerate(arrays): values[k,:len(a)]=a; mask[k,:len(a)]=True
            X=[values,mask]
        else: raise ValueError("Unknown batch kind")
        y=np.array([self.records[i]["label"] for i in indices],np.float32)
        return X,y,None if self.weights is None else self.weights[ids]


def compile_model(model,lr,label_smoothing,clipvalue=None,clipnorm=None):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr,clipvalue=clipvalue,clipnorm=clipnorm),
                  loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
                  metrics=["accuracy"],weighted_metrics=[])


@keras.utils.register_keras_serializable(package='GaitSafety')
class ConstantPenalty(layers.Layer):
    def __init__(self,value=0.,**kwargs):
        super().__init__(**kwargs); self.value=float(value)
    def call(self,x):
        self.add_loss(tf.cast(self.value,x.dtype)); return x
    def get_config(self): return {**super().get_config(),'value':self.value}


class CachedEmbeddings:
    """Exact fixed features for a declared deterministic frozen encoder."""
    def __init__(self,source,extractor,mil=False):
        self.source=source; self.mil=mil
        self.values=[]; self.labels=[]; self.weights=[]
        source.order=np.arange(len(source.items))
        for j in range(len(source)):
            x,y,w=source[j]
            h=np.asarray(extractor(x,training=False),dtype=np.float32)
            for k in range(len(y)):
                self.values.append(h[k,x[1][k]] if mil else h[k])
                self.labels.append(y[k]); self.weights.append(w[k] if w is not None else 1.)
        self.order=np.arange(len(self.values))
    def __len__(self): return int(np.ceil(len(self.values)/self.source.batch))
    def set_epoch(self,epoch):
        self.order=np.arange(len(self.values))
        if self.source.shuffle: self.order=np.random.default_rng((self.source.seed+epoch)%2**32).permutation(self.order)
    def __getitem__(self,j):
        ids=self.order[j*self.source.batch:(j+1)*self.source.batch]
        values=[self.values[i] for i in ids]
        if self.mil:
            size=max(len(v) for v in values)
            h=np.zeros((len(ids),size,values[0].shape[-1]),np.float32); mask=np.zeros((len(ids),size),bool)
            for i,v in enumerate(values): h[i,:len(v)]=v; mask[i,:len(v)]=True
            x=[h,mask]
        else: x=np.stack(values)
        return x,np.array(self.labels,dtype=np.float32)[ids],np.array(self.weights,dtype=np.float32)[ids] if self.source.weights is not None else None


def cached_head(full,training,validation,mil=False):
    """Cache deterministic inference-mode backbone, including its constant L2 loss.

    Original LPGNet frozen backbones still contain noise/dropout whose training
    behavior is stochastic. This is an explicit accelerated-protocol choice,
    not a claimed exact stochastic reproduction. Uncached mode remains available.
    """
    layer=full.get_layer('shared_window_encoder') if mil else next(l for l in full.layers if isinstance(l,keras.Model))
    if (layer.encoder.trainable if mil else layer.trainable):
        raise ValueError('Cannot cache a trainable encoder')
    encoded=layer(full.inputs if mil else full.inputs[0],training=False)
    extractor=keras.Model(full.inputs if mil else full.inputs[0],encoded)
    tr=CachedEmbeddings(training,extractor,mil)
    va=CachedEmbeddings(validation,extractor,mil) if validation is not None else None
    # extractor was executed; preserve regularizers of its fixed weights exactly.
    penalty=float(sum(float(v.numpy()) for v in extractor.losses))
    if mil:
        h=layers.Input((None,int(encoded.shape[-1])),name='cached_window_embeddings')
        mask=layers.Input((None,),dtype='bool',name='cached_mask')
        a=full.get_layer('recording_attention_tanh')(h)
        g=full.get_layer('recording_attention_gate')(h)
        logits=full.get_layer('recording_attention_logits')(full.get_layer('recording_attention_gated')([a,g]))
        weights=full.get_layer('window_attention_remask')([logits,mask])
        z=full.get_layer('recording_embedding')([h,weights])
        out=full.get_layer('recording_pd_probability')(full.get_layer('recording_dropout')(z))
        head=keras.Model([h,mask],ConstantPenalty(penalty)(out))
        names=['recording_attention_tanh','recording_attention_gate','recording_attention_logits','recording_pd_probability']
    else:
        h=layers.Input((int(encoded.shape[-1]),),name='cached_recording_features')
        drop=next(l for l in full.layers if isinstance(l,layers.Dropout))
        dense=next(l for l in full.layers if isinstance(l,layers.Dense))
        head=keras.Model(h,ConstantPenalty(penalty)(dense(drop(h))))
        names=[dense.name]
    return head,tr,va,names


def profile_settings(family,profile='efficient'):
    """Declare the scientific budget before training; never silently skip folds."""
    if profile not in ('efficient','full','factorial_fast','fixed'):
        raise ValueError('Unknown run profile')
    if profile=='factorial_fast' and family!='factorial': raise ValueError('factorial_fast is only for the factorial study')
    result=dict(RUN_PROFILE=profile,N_FOLDS=10,N_INNER=3,
                CACHE_FROZEN_ENCODER=profile!='full',REQUIRE_DURABLE_STORAGE=True)
    if profile=='full':
        result.update(MAX_EPOCHS_WIN=1000,MAX_EPOCHS_REC=200,MAX_EPOCHS_MIL=200,PATIENCE=25,RLR_PATIENCE=14,
                      N_TRIALS=12,TRAIN_SEEDS=[42,123,2026] if family=='LP_ResMIL' else [42],
                      CONDITIONS=[['global','test'],['local','test'],['global','inner'],['local','inner']])
    else:
        result.update(MAX_EPOCHS_WIN=100,MAX_EPOCHS_REC=40,MAX_EPOCHS_MIL=40,PATIENCE=12,RLR_PATIENCE=5,
                      N_TRIALS=1 if profile=='fixed' else 3,TRAIN_SEEDS=[42],
                      CONDITIONS=[['global','test'],['local','test'],['global','inner'],['local','inner']]
                                 if profile=='factorial_fast' else [['local','inner']])
    return result


def budget_report(ns,family):
    if family=='factorial':
        tasks=len(ns.get('CONDITIONS',[1,2,3,4]))*ns['N_FOLDS']
        pairs=tasks
    else:
        tasks=len(ns['TRAIN_SEEDS'])*ns['N_FOLDS']
        # A fixed configuration needs epoch selection and refit, not selection-CV.
        pairs=tasks*(2 if ns['N_TRIALS']==1 else ns['N_TRIALS']*ns['N_INNER']+2)
    return dict(outer_tasks=tasks,two_stage_fits=pairs,training_stages=2*pairs,
                maximum_total_epochs=pairs*(ns['MAX_EPOCHS_WIN']+ns['MAX_EPOCHS_MIL' if family=='LP_ResMIL' else 'MAX_EPOCHS_REC']))


def safe_fit(model,train,validation,folder,epochs,patience,lr_patience,min_lr,verbose=0,mirror=None):
    """Save optimizer/model, history, LR and stopping counters at every epoch.

    Write a generation before atomically publishing its state pointer. Partial
    epochs restart; data order depends on absolute epoch. Stochastic GPU bitwise
    equivalence across sessions is not claimed. Old best/latest generations only
    are retained, and no loaded checkpoint uses unsafe Lambda deserialization.
    """
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    state_path=folder/"state.json"
    state=dict(next_epoch=0,best=None,best_epoch=0,wait=0,lr_wait=0,lr_best=None,history={},complete=False)
    if state_path.exists():
        state=json.loads(state_path.read_text())
        for key in ("last","best_file"):
            file=folder/state[key]
            if hashlib.sha256(file.read_bytes()).hexdigest()!=state[key+"_sha256"]:
                raise ValueError("Corrupt epoch checkpoint")
        model=keras.models.load_model(folder/state["last"],safe_mode=True)
    def publish_epoch():
        if mirror is not None:
            files=[state_path]+[folder/state[k] for k in ('last','best_file')]
            mirror.publish('epoch/'+folder.relative_to(mirror.local_root).as_posix(),
                           [p.relative_to(mirror.local_root).as_posix() for p in files])
    if state_path.exists(): publish_epoch()  # Recommit a local epoch after a failed remote write.
    for epoch in range(state["next_epoch"],epochs):
        if state["complete"]: break
        epoch_start=time.perf_counter()
        train.set_epoch(epoch); model.reset_metrics()
        train_logs={}
        for j in range(len(train)):
            x,y,w=train[j]
            train_logs=model.train_on_batch(x,y,sample_weight=w,return_dict=True)
        logs={k:float(v) for k,v in train_logs.items()}
        logs['train_seconds']=time.perf_counter()-epoch_start
        validation_start=time.perf_counter()
        if validation is not None:
            validation.set_epoch(0); model.reset_metrics()
            for j in range(len(validation)):
                x,y,w=validation[j]
                val_logs=model.test_on_batch(x,y,sample_weight=w,return_dict=True)
            logs.update({"val_"+k:float(v) for k,v in val_logs.items()})
        logs['validation_seconds']=time.perf_counter()-validation_start
        if not all(np.isfinite(v) for v in logs.values()): raise FloatingPointError("Nonfinite training/validation metrics")
        score=logs.get("val_loss",logs["loss"])
        improved=state["best"] is None or score<state["best"]
        if improved: state.update(best=score,best_epoch=epoch+1,wait=0)
        else: state["wait"]+=1
        if validation is not None:
            if state["lr_best"] is None or score<state["lr_best"]-1e-4:
                state.update(lr_best=score,lr_wait=0)
            else:
                state["lr_wait"]+=1
                if state["lr_wait"]>=lr_patience:
                    lr=float(keras.backend.get_value(model.optimizer.learning_rate))
                    model.optimizer.learning_rate.assign(max(min_lr,lr*.25)); state["lr_wait"]=0
        logs["learning_rate"]=float(keras.backend.get_value(model.optimizer.learning_rate))
        for k,v in logs.items(): state["history"].setdefault(k,[]).append(v)
        generation="epoch-"+str(epoch+1)+"-"+uuid.uuid4().hex
        target=folder/generation; target.mkdir()
        last=target/"model.keras"; model.save(last)
        state["last"]=last.relative_to(folder).as_posix()
        state["last_sha256"]=hashlib.sha256(last.read_bytes()).hexdigest()
        if improved or validation is None:
            state["best_file"]=state["last"]; state["best_file_sha256"]=state["last_sha256"]
        state["next_epoch"]=epoch+1
        state["complete"]=(epoch+1>=epochs or (validation is not None and state["wait"]>=patience))
        atomic_json(state_path,state)
        checkpoint_start=time.perf_counter()
        publish_epoch()  # Training never advances until the off-runtime commit is verified.
        keep={Path(state[k]).parts[0] for k in ("last","best_file")}
        for old in folder.glob("epoch-*"):
            if old.is_dir() and old.name not in keep: shutil.rmtree(old)
        if verbose: print(f"  epoch {epoch+1}: loss={logs['loss']:.4f}, monitored={score:.4f}, "
                  f"train={logs['train_seconds']:.1f}s val={logs['validation_seconds']:.1f}s "
                  f"remote={time.perf_counter()-checkpoint_start:.1f}s",flush=True)
    if "last" not in state: raise ValueError("No completed epoch")
    model=keras.models.load_model(folder/state["best_file"] if validation is not None else folder/state["last"],safe_mode=True)
    return model,state["history"],state["best_epoch"] if validation is not None else state["next_epoch"]


def load_inputs(cache,policy="curated",download=True):
    """Original-all for intentional replication; curated for defensible nested studies."""
    if policy not in ("curated","official_all"): raise ValueError("Invalid cohort policy")
    entries=official_data.manifest(cache,download)
    if policy=="curated":
        retained,audit,provenance=official_data.load_verified_cohort(cache,2000,download)
        records=[{**r,"file":r["path"]} for r in retained]
    else:
        records=[]
        for name in sorted(n for n in entries if official_data.RECORDING.fullmatch(n)):
            blob=official_data.verified_file(cache,name,entries[name],download)
            import io
            x=np.loadtxt(io.BytesIO(blob),ndmin=2)
            if x.shape[1]!=19 or not np.isfinite(x).all(): raise ValueError("Invalid original recording")
            records.append(dict(file=name,subject=name[:6],study=name[:2],label=int(name[2:4]=="Pt"),
                                signal=x[:,1:19].astype(np.float32),sha256=entries[name]))
        audit=[{k:v for k,v in r.items() if k!='signal'} for r in records]
        provenance=dict(policy=policy,manifest_sha256=official_data.MANIFEST_SHA256,
            warning="Original-all retains known cross-ID duplicates, including conflicting labels. Not a clean cohort.")
    metadata,meta_provenance=official_data.load_verified_clinical(cache,download)
    clinical={}
    for r in records:
        subject=r["subject"]; row=metadata.get(subject,{})
        hy=row.get("hy") if r["label"] and row.get("hy_status")=="verified" else None
        clinical[subject]=dict(study=r["study"],diagnosis="PD" if r["label"] else "Control",
            hy=np.nan if hy is None else hy,hy_label=str(hy) if hy is not None else "Unknown" if r["label"] else "Not applicable",
            hy_status=row.get("hy_status","unmatched") if r["label"] else "not_applicable",
            age=row.get("age",np.nan),gender=row.get("gender_code",np.nan))
    return records,clinical,audit,dict(cohort=provenance,clinical=meta_provenance)


def fit_lp(signals,order=11):
    if not signals: raise ValueError("No training controls for LP")
    C=np.zeros((18,order)); stds=[]
    for c in range(18):
        conc=np.concatenate([np.zeros(order)]+[np.r_[s[:,c],np.zeros(order)] for s in signals]).astype(np.float64)
        std=float(np.std(conc,ddof=1)); stds.append(std+1e-8)
        if std<1e-10: raise ValueError("Degenerate control channel; do not silently change the estimator")
        n=len(conc); nfft=1<<(2*n-1).bit_length()
        f=np.fft.rfft(conc,nfft); r=np.fft.irfft(f*np.conj(f),nfft)[:order+1]/n
        from scipy.linalg import solve_toeplitz
        C[c]=solve_toeplitz(r[:order],r[1:order+1])
    if not np.isfinite(C).all(): raise ValueError("Invalid LP coefficients")
    return C,np.array(stds)


def apply_lp(x,C,stds):
    values=np.stack([lfilter(np.r_[1.,-C[c]],1,x[:,c]) for c in range(18)],axis=1)/stds
    if not np.isfinite(values).all(): raise ValueError("Nonfinite LP residual")
    return values.astype(np.float32)


def subject_split(records,indices,fraction,seed):
    subjects=sorted({records[i]["subject"] for i in indices})
    labels=[next(records[i]["label"] for i in indices if records[i]["subject"]==s) for s in subjects]
    if len(set(labels))!=2: raise ValueError("Both diagnoses are required in every split")
    tr,va=next(StratifiedShuffleSplit(1,test_size=fraction,random_state=seed).split(subjects,labels))
    a,b={subjects[i] for i in tr},{subjects[i] for i in va}
    if a&b: raise AssertionError("Participant overlap")
    return [i for i in indices if records[i]["subject"] in a],[i for i in indices if records[i]["subject"] in b]


def outer_folds(records,n=10,seed=0):
    subjects=sorted({r["subject"] for r in records})
    label={r["subject"]:r["label"] for r in records}
    yy=[label[s] for s in subjects]
    if min(Counter(yy).values())<n: raise ValueError("Not enough participants per diagnosis")
    folds=[]
    for tr,te in StratifiedKFold(n,shuffle=True,random_state=seed).split(subjects,yy):
        a,b={subjects[i] for i in tr},{subjects[i] for i in te}
        folds.append(dict(train=[i for i,r in enumerate(records) if r["subject"] in a],
                          test=[i for i,r in enumerate(records) if r["subject"] in b]))
    tested=[i for f in folds for i in f["test"]]
    if sorted(tested)!=list(range(len(records))): raise AssertionError("Incomplete or repeated outer coverage")
    return folds


def checked_predictions(records,indices,rows):
    expected={records[i]["file"]:records[i] for i in indices}
    if len(rows)!=len(expected) or {r["file"] for r in rows}!=set(expected):
        raise ValueError("Checkpoint recording coverage mismatch")
    for row in rows:
        original=expected[row["file"]]
        if row["subject"]!=original["subject"] or row["label"]!=original["label"]:
            raise ValueError("Checkpoint participant/label mismatch")
        if not np.isfinite(row["probability"]) or not 0<=row["probability"]<=1:
            raise ValueError("Invalid checkpoint probability")


class NotebookExperiment:
    """Shared tested runner; namespace is read only except returned results."""
    def __init__(self,ns,family):
        self.ns,self.family=ns,family
        self.mirror=ns.get('DURABLE_MIRROR')
        if ns.get('REQUIRE_DURABLE_STORAGE',False) and (self.mirror is None or not self.mirror.verified):
            raise RuntimeError('Long training requires verified off-runtime storage; run the storage cell first')
        if self.mirror is not None and self.mirror.local_root!=Path(ns['CHECKPOINT_DIR']).resolve():
            raise ValueError('Checkpoint directory does not match durable mirror')
        if ns.get('REQUIRE_GPU',False) and not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('No TensorFlow GPU detected; refusing accidental multi-hour CPU research run')
        self.records=ns["records"]
        self.folds=outer_folds(self.records,ns["N_FOLDS"],ns["FOLD_SEED"])
        self.processed={i:lfilter([.5,.5],1,r["signal"],axis=0)[::2] for i,r in enumerate(self.records)}
        keys=("SEED","FOLD_SEED","SPLIT_SEED","SEARCH_SEED","N_FOLDS","N_INNER","VAL_FRAC","LP_ORDER",
              "WIN","STRIDE","BATCH_WIN","BATCH_REC","PRED_BATCH","LR_WIN","LR_REC","LABEL_SMOOTH","L2_REG",
              "MAX_EPOCHS_WIN","MAX_EPOCHS_REC","MAX_EPOCHS_MIL","PATIENCE","RLR_PATIENCE","MIN_LR",
              "TRAIN_SEEDS","N_TRIALS","THRESHOLD","COHORT_POLICY","SPACE_A","SPACE_RESMIL","EXPORT_ATTENTION",
              "RUN_PROFILE","CACHE_FROZEN_ENCODER","CONDITIONS")
        self.config={k:json_ready(ns[k]) for k in keys if k in ns}
        self.plan=dict(schema=3,family=family,configuration=self.config,
            runtime=dict(tensorflow=tf.__version__,keras=keras.__version__,numpy=np.__version__,
                         scipy=scipy.__version__,sklearn=sklearn.__version__,python=platform.python_version()),
            source_hash=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest(),
            storage_source_hash=hashlib.sha256(Path(__file__).with_name('durable_storage.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            audit=ns.get("data_audit",[]),provenance=ns.get("data_provenance",{}),
            inventory=[dict(file=r["file"],subject=r["subject"],label=r["label"],
                       signal_sha256=hashlib.sha256(r["signal"].tobytes()).hexdigest()) for r in self.records],
            folds=[{part:[self.records[i]["file"] for i in indices] for part,indices in f.items()} for f in self.folds])
        self.store=RunStore(ns["CHECKPOINT_DIR"],self.plan,self.mirror)
        self._feature_cache={}

    def features(self,training,evaluation):
        cache_key=digest([training,evaluation])
        if cache_key in self._feature_cache: return self._feature_cache[cache_key]
        controls=[self.processed[i] for i in training if self.records[i]["label"]==0]
        C,stds=fit_lp(controls,self.ns["LP_ORDER"])
        feats={i:apply_lp(self.processed[i],C,stds) for i in dict.fromkeys(training+evaluation)}
        reference=dict(coefficients=C.tolist(),stds=stds.tolist(),
                       controls=[self.records[i]["file"] for i in training if self.records[i]["label"]==0])
        # Small LRU: inner folds reuse the same core reference across candidates.
        if len(self._feature_cache)>=self.ns.get('N_INNER',3)+1:
            self._feature_cache.pop(next(iter(self._feature_cache)))
        self._feature_cache[cache_key]=(feats,reference)
        return feats,reference

    def batches(self,indices,feats,kind,seed,shuffle=False):
        ns=self.ns
        return Batches(self.records,indices,feats,kind,win=ns["WIN"],stride=ns["STRIDE"],
                       batch=ns["BATCH_WIN"] if kind=="window" else ns["BATCH_REC"],
                       seed=seed,shuffle=shuffle,balanced=self.family=="LP_ResMIL")

    def train(self,cfg,tr,va,feats,key,seed,epochs=None):
        ns=self.ns; keras.backend.clear_session(); gc.collect(); keras.utils.set_random_seed(seed)
        win=build_window_classifier(cfg,ns["WIN"]) if self.family=="LP_ResMIL" else build_variantA(**cfg) if self.family=="A" else build_lpgnet(**cfg)
        compile_model(win,ns["LR_WIN"],ns["LABEL_SMOOTH"],clipvalue=.5)
        val_kind="window" if self.family=="LP_ResMIL" else "recording"
        validation=self.batches(va,feats,val_kind,seed) if va is not None else None
        folder=self.store.folder(key)
        window_epochs=epochs["window"] if epochs else ns["MAX_EPOCHS_WIN"]
        second="mil" if self.family=="LP_ResMIL" else "recording"
        win,h1,e1=safe_fit(win,self.batches(tr,feats,"window",seed,True),validation,folder/"window",
            max(1,window_epochs),ns["PATIENCE"],ns["RLR_PATIENCE"],ns["MIN_LR"],ns.get("VERBOSE_EPOCHS",0),self.mirror)
        full=build_mil_model(win,cfg,ns["WIN"]) if second=="mil" else build_recording_model(win)
        compile_model(full,cfg["lr_mil"] if second=="mil" else ns["LR_REC"],ns["LABEL_SMOOTH"],
                      clipnorm=1. if second=="mil" else None,clipvalue=None if second=="mil" else 1.)
        limit=epochs[second] if epochs else ns["MAX_EPOCHS_MIL"] if second=="mil" else ns["MAX_EPOCHS_REC"]
        training=self.batches(tr,feats,second,seed+1,True)
        validation=self.batches(va,feats,second,seed+1) if va is not None else None
        cache_enabled=ns.get('CACHE_FROZEN_ENCODER',False) and (second!='mil' or cfg['fine_tune']=='frozen')
        if cache_enabled:
            start=time.perf_counter()
            head,training,validation,names=cached_head(full,training,validation,mil=second=='mil')
            compile_model(head,cfg['lr_mil'] if second=='mil' else ns['LR_REC'],ns['LABEL_SMOOTH'],
                          clipnorm=1. if second=='mil' else None,clipvalue=None if second=='mil' else 1.)
            print(f'Frozen encoder cached once in {time.perf_counter()-start:.1f}s; stage 2 trains the small head only.',flush=True)
            head,h2,e2=safe_fit(head,training,validation,folder/'recording_cached',max(1,limit),
                ns['PATIENCE'],ns['RLR_PATIENCE'],ns['MIN_LR'],ns.get('VERBOSE_EPOCHS',0),self.mirror)
            for name in names: full.get_layer(name).set_weights(head.get_layer(name).get_weights())
        else:
            full,h2,e2=safe_fit(full,training,validation,folder/"recording",
                max(1,limit),ns["PATIENCE"],ns["RLR_PATIENCE"],ns["MIN_LR"],ns.get("VERBOSE_EPOCHS",0),self.mirror)
        return full,dict(window=e1,**{second:e2}),dict(window=h1,**{second:h2})

    def predict(self,model,indices,feats,base,attention=False):
        rows,attention_rows=[],[]
        if self.family=="LP_ResMIL":
            batches=self.batches(indices,feats,"mil",0)
            inspector=keras.Model(model.inputs,model.get_layer("window_attention_remask").output) if attention else None
            pos=0
            for j in range(len(batches)):
                x,y,_=batches[j]
                p=np.asarray(model(x,training=False)).reshape(-1)
                att=np.asarray(inspector(x,training=False)) if inspector is not None else None
                for local,prob in enumerate(p):
                    i=indices[pos]; pos+=1
                    r=self.records[i]
                    rows.append(dict(**base,file=r["file"],subject=r["subject"],label=r["label"],probability=float(prob)))
                    if att is not None:
                        starts=list(range(0,len(feats[i])-self.ns["WIN"],self.ns["STRIDE"]))
                        for k,st in enumerate(starts):
                            attention_rows.append(dict(**base,file=r["file"],subject=r["subject"],label=r["label"],window=k,
                                start_sample=st,end_sample=st+self.ns["WIN"],start_seconds=st/50,end_seconds=(st+self.ns["WIN"])/50,
                                attention=float(att[local,k])))
        else:
            # Do not stack variable lengths or wrap-pad inference: preserve full-record scoring.
            for i in indices:
                r=self.records[i]; p=float(np.asarray(model(feats[i][None],training=False)).reshape(-1)[0])
                rows.append(dict(**base,file=r["file"],subject=r["subject"],label=r["label"],probability=p))
        checked_predictions(self.records,indices,rows)
        return rows,attention_rows

    def participant_metrics(self,rows):
        groups={}
        for r in rows: groups.setdefault(r["subject"],[]).append(r)
        yy=[g[0]["label"] for g in groups.values()]
        pp=[np.mean([r["probability"] for r in g]) for g in groups.values()]
        return dict(log_loss=float(log_loss(yy,pp,labels=[0,1])),roc_auc=float(roc_auc_score(yy,pp)),
                    accuracy=float(np.mean((np.array(pp)>=self.ns["THRESHOLD"])==yy)),
                    brier=float(np.mean((np.array(pp)-yy)**2)))

    def nested(self,max_tasks=None):
        ns=self.ns; selections=[]; oof=[]; trials_all=[]; selected_inner=[]; attention_all=[]
        new_tasks=0; complete=True
        space=ns["SPACE_RESMIL"] if self.family=="LP_ResMIL" else ns["SPACE_A"]
        for seed in ns["TRAIN_SEEDS"]:
            for k,fold in enumerate(self.folds):
                outer_key=("outer",self.family,seed,k)
                saved=self.store.load(outer_key)
                if saved is None:
                    if max_tasks is not None and new_tasks>=max_tasks:
                        complete=False; continue
                    configs=unique_candidates(space,ns["N_TRIALS"],ns["SEARCH_SEED"]+k)
                    outer_train=fold["train"]
                    inner=outer_folds([self.records[i] for i in outer_train],ns["N_INNER"],ns["SPLIT_SEED"]+777*k)
                    # Local fold indices map back to the immutable global inventory.
                    inner=[dict(train=[outer_train[i] for i in f["train"]],test=[outer_train[i] for i in f["test"]]) for f in inner]
                    trial_rows=[]; all_inner=[]
                    second='mil' if self.family=='LP_ResMIL' else 'recording'
                    for t,cfg in enumerate(configs if ns['N_TRIALS']>1 else []):
                        prediction_rows=[]; best_epochs=[]
                        for j,f in enumerate(inner):
                            key=("inner",self.family,seed,k,t,j)
                            result=self.store.load(key)
                            if result is None:
                                core,early=subject_split(self.records,f["train"],ns["VAL_FRAC"],ns["SPLIT_SEED"]+1000+j)
                                feats,reference=self.features(core,early+f["test"])
                                start=time.time()
                                model,best,history=self.train(cfg,core,early,feats,key,seed+10000+j)
                                rows,_=self.predict(model,f["test"],feats,dict(family=self.family,seed=seed,outer_fold=k,trial=t,inner_fold=j,fold=j))
                                result=dict(rows=rows,best=best,history=history,reference=reference,seconds=time.time()-start)
                                self.store.save(key,result)
                                del model,feats; keras.backend.clear_session(); gc.collect()
                            checked_predictions(self.records,f["test"],result["rows"])
                            prediction_rows+=result["rows"]; best_epochs.append(result["best"])
                        score=self.participant_metrics(prediction_rows)
                        params=model_parameter_report(self.family,cfg,ns["WIN"],ns["LP_ORDER"])
                        second="mil" if self.family=="LP_ResMIL" else "recording"
                        trial=dict(family=self.family,seed=seed,outer_fold=k,trial=t,candidate=cfg,
                            inner_participant_log_loss=score["log_loss"],inner_participant_auc=score["roc_auc"],
                            inner_participant_accuracy=score["accuracy"],inner_participant_brier=score["brier"],
                            mean_best_window_epoch=float(np.mean([e["window"] for e in best_epochs])),
                            **{"mean_best_"+second+"_epoch":float(np.mean([e[second] for e in best_epochs]))},**params)
                        trial_rows.append(trial); all_inner.append(prediction_rows)
                        print(f"{self.family}: outer {k}, seed {seed}, trial {t}: inner NLL={score['log_loss']:.4f}",flush=True)
                    if ns['N_TRIALS']==1:
                        # No hyperparameter contest: avoid three redundant assessment fits.
                        trial_rows=[dict(family=self.family,seed=seed,outer_fold=k,trial=0,candidate=configs[0],
                            selection_mode='prespecified_single_configuration',
                            inner_participant_log_loss=None,inner_participant_auc=None,
                            **model_parameter_report(self.family,configs[0],ns['WIN'],ns['LP_ORDER']))]
                        all_inner=[[]]
                    selected=0 if ns['N_TRIALS']==1 else min(range(len(trial_rows)),key=lambda i:(trial_rows[i]["inner_participant_log_loss"],-trial_rows[i]["inner_participant_auc"]))
                    cfg=configs[selected]
                    # Additional outer-development-only early stopping; early controls excluded from LP.
                    core,early=subject_split(self.records,outer_train,ns["VAL_FRAC"],ns["SPLIT_SEED"]+300000+k)
                    feats,early_reference=self.features(core,early)
                    selection_key=("epochs",self.family,seed,k)
                    model,best,history=self.train(cfg,core,early,feats,selection_key,seed+400000)
                    del model,feats; keras.backend.clear_session(); gc.collect()
                    # Fixed-epoch final refit is allowed to use ALL outer-training controls.
                    feats,reference=self.features(outer_train,fold["test"])
                    model,_,refit_history=self.train(cfg,outer_train,None,feats,("refit",self.family,seed,k),seed+400000,epochs=best)
                    rows,attention=self.predict(model,fold["test"],feats,dict(family=self.family,seed=seed,fold=k),
                                                ns.get("EXPORT_ATTENTION",False))
                    chosen=trial_rows[selected]
                    selection=dict(chosen,best_window_epoch=best["window"],
                                   **{"best_"+second+"_epoch":best[second]})
                    saved=dict(oof=rows,trials=trial_rows,selection=selection,inner=all_inner[selected],attention=attention,
                               reference=reference,early_reference=early_reference,epoch_history=history,refit_history=refit_history)
                    self.store.save(outer_key,saved); new_tasks+=1
                    del model,feats; keras.backend.clear_session(); gc.collect()
                checked_predictions(self.records,fold["test"],saved["oof"])
                if len(saved["trials"])!=ns["N_TRIALS"]: raise ValueError("Incomplete candidate inventory")
                oof+=saved["oof"]; trials_all+=saved["trials"]; selections.append(saved["selection"])
                selected_inner+=saved["inner"]; attention_all+=saved["attention"]
        return dict(complete=complete,all_oof_rows=oof,all_trials=trials_all,selections=selections,
                    selected_inner_rows=selected_inner,all_attention_rows=attention_all,run_signature=self.store.signature)

    def factorial(self,max_tasks=None):
        ns=self.ns; oof=[]; folds=[]; new_tasks=0; complete=True
        all_indices=list(range(len(self.records)))
        global_feats=None
        conditions=ns.get('CONDITIONS',[("global","test"),("local","test"),("global","inner"),("local","inner")])
        if not conditions or len({tuple(c) for c in conditions})!=len(conditions) or any(tuple(c) not in {('global','test'),('local','test'),('global','inner'),('local','inner')} for c in conditions):
            raise ValueError('Invalid factorial conditions')
        for scope,validation in conditions:
            condition=scope+"__"+validation
            for k,fold in enumerate(self.folds):
                key=("factorial",condition,k)
                saved=self.store.load(key)
                if saved is None:
                    if max_tasks is not None and new_tasks>=max_tasks:
                        complete=False; continue
                    tr,va=(fold["train"],fold["test"]) if validation=="test" else subject_split(self.records,fold["train"],ns["VAL_FRAC"],ns["SEED"]+k)
                    if scope=="global":
                        if global_feats is None: global_feats,global_reference=self.features(all_indices,[])
                        feats,reference=global_feats,global_reference
                    else:
                        # local__inner excludes early-stop controls; no hidden preprocessing leakage.
                        feats,reference=self.features(tr,list(dict.fromkeys(va+fold["test"])))
                    model,best,history=self.train({},tr,va,feats,key,ns["SEED"]+k)
                    rows,_=self.predict(model,fold["test"],feats,dict(condition=condition,lp_scope=scope,validation=validation,fold=k))
                    metrics=self.ns["metric_dict"]([r["label"] for r in rows],[r["probability"] for r in rows])
                    saved=dict(oof=rows,fold=dict(condition=condition,fold=k,best_window_epoch=best["window"],
                             best_recording_epoch=best["recording"],**metrics),history=history,reference=reference)
                    self.store.save(key,saved); new_tasks+=1
                    del model; keras.backend.clear_session(); gc.collect()
                    print(f"Completed {condition} fold {k} (intentional test exposure: {validation=='test' or scope=='global'})",flush=True)
                checked_predictions(self.records,fold["test"],saved["oof"])
                oof+=saved["oof"]; folds.append(saved["fold"])
        return dict(complete=complete,oof_rows=oof,fold_rows=folds,run_signature=self.store.signature)


def unique_candidates(space,n,seed):
    if n<1 or n>int(np.prod([len(v) for v in space.values()])): raise ValueError("Invalid candidate budget")
    rng=np.random.default_rng(seed); configs=[]; seen=set()
    while len(configs)<n:
        c={k:v[int(rng.integers(len(v)))] for k,v in space.items()}
        token=digest(c)
        if token not in seen: configs.append(c); seen.add(token)
    return configs


def model_parameter_report(family,cfg,win=100,order=11):
    if family=="LP_ResMIL":
        model=build_window_classifier(cfg,win); full=build_mil_model(model,cfg,win)
        result=dict(window_parameters=int(model.count_params()),total_classifier_parameters=int(full.count_params()),
            trainable_parameters=int(sum(np.prod(w.shape) for w in full.trainable_weights)),
            lp_parameters=18*order,total_including_lp=int(full.count_params()+18*order))
    else:
        model=build_variantA(**cfg)
        result=dict(classifier_parameters=int(model.count_params()),predictor_parameters=18*order,
                    total_parameters=int(model.count_params()+18*order))
    keras.backend.clear_session(); gc.collect()
    return result


def smoke_check(family,cfg,win=100):
    """Real construction, backward, masked padding invariance and safe roundtrip."""
    import tempfile
    keras.utils.set_random_seed(39)
    if family=="LP_ResMIL":
        model=build_mil_model(build_window_classifier(cfg,win),cfg,win)
        x=np.random.default_rng(9).normal(size=(2,3,win,18)).astype(np.float32)
        mask=np.array([[True,False,False],[True,True,False]])
        inputs=[x,mask]
        original=np.asarray(model(inputs,training=False))
        changed=x.copy(); changed[~mask]=9999
        np.testing.assert_allclose(original,model([changed,mask],training=False),atol=1e-5)
    else:
        model=build_variantA(**cfg) if family=="A" else build_lpgnet()
        inputs=np.random.default_rng(9).normal(size=(2,win,18)).astype(np.float32)
    compile_model(model,1e-4,.1,clipnorm=1.)
    model.train_on_batch(inputs,np.array([0.,1.],np.float32))
    expected=np.asarray(model(inputs,training=False))
    if not np.isfinite(expected).all(): raise ValueError("Nonfinite smoke output")
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"smoke.keras"; model.save(path)
        restored=keras.models.load_model(path,safe_mode=True)
        np.testing.assert_allclose(expected,restored(inputs,training=False),atol=1e-5)
        if int(restored.optimizer.iterations)!=int(model.optimizer.iterations): raise AssertionError("Optimizer state lost")
    n=model.count_params(); keras.backend.clear_session(); gc.collect()
    return dict(family=family,parameters=int(n),tensorflow=tf.__version__,keras=keras.__version__,
                safe_model_optimizer_roundtrip=True,padding_invariance=family=="LP_ResMIL")


def unpack_result(ns,result,family):
    """Validate complete inventories before constructing any outer-score tables."""
    if not result["complete"]: return dict(EXPERIMENT_COMPLETE=False)
    if family=="factorial":
        rows=result["oof_rows"]
        groups=["condition"]
        expected=len(ns.get('CONDITIONS',[1,2,3,4]))
        data=pd.DataFrame(rows)
        fold_metrics=pd.DataFrame(result["fold_rows"])
        extras=dict(oof=ns["oof_frame"](rows),fold_metrics=fold_metrics)
    else:
        rows=result["all_oof_rows"]; data=pd.DataFrame(rows); groups=["family","seed"]
        expected=len(ns["TRAIN_SEEDS"])
        extras=dict(nested_oof=ns["oof_frame"](rows),trial_table=pd.DataFrame(result["all_trials"]),
            selection_table=pd.DataFrame(result["selections"]),
            inner_selected_oof=ns["oof_frame"](result["selected_inner_rows"]) if result['selected_inner_rows'] else pd.DataFrame(),
            attention_oof=pd.DataFrame(result["all_attention_rows"]))
    if data.duplicated(groups+["file"]).any(): raise ValueError("Duplicate OOF recordings")
    if data.groupby(groups).ngroups!=expected: raise ValueError("Incomplete experimental groups")
    files={r["file"] for r in ns["records"]}
    for _,group in data.groupby(groups):
        if set(group["file"])!=files or len(group)!=len(files): raise ValueError("Incomplete OOF inventory")
        if group.groupby("subject")["fold"].nunique().max()!=1: raise ValueError("Participant crosses test folds")
    return dict(EXPERIMENT_COMPLETE=True,**extras)


def summary_tables(ns,family):
    groups=["condition"] if family=="factorial" else ["family","seed"]
    oof=ns["oof"] if family=="factorial" else ns["nested_oof"]
    summary=ns["summarize_oof"](oof,groups)
    intervals=ns["pooled_bootstrap"](oof,groups)
    folds=[]
    for keys,g in oof.groupby(groups+["fold"]):
        part=ns["participant_oof"](g,groups)
        folds.append(dict(zip(groups+["fold"],keys))|ns["metric_dict"](part["label"],part["probability"]))
    return dict(summary=summary,participant_ci=intervals,participant_fold_metrics=pd.DataFrame(folds))


def error_reports(ns,family):
    groups=["condition"] if family=="factorial" else ["family","seed"]
    oof=ns["oof"] if family=="factorial" else ns["nested_oof"]
    part,errors,severity,study=ns["error_tables"](oof,groups)
    if family=="factorial":
        stability=errors.groupby("subject").agg(error_conditions=("condition","nunique"),mean_nll=("nll","mean")).reset_index()
    else:
        stability=part.groupby(["family","subject"]).agg(error_seeds=("correct",lambda x:int((x==0).sum())),
             n_seeds=("seed","nunique"),probability_mean=("probability","mean"),probability_std=("probability","std"),
             prediction_variants=("prediction","nunique")).reset_index()
    inner=pd.DataFrame()
    if family!="factorial" and not ns["inner_selected_oof"].empty:
        inner=ns["participant_oof"](ns["inner_selected_oof"],["family","seed","outer_fold","trial"]).sort_values("nll",ascending=False)
    attention=ns.get("attention_oof",pd.DataFrame()); audit=pd.DataFrame()
    if not attention.empty:
        audit=attention.groupby(["family","seed","fold","file"]).agg(valid_windows=("window","count"),attention_sum=("attention","sum")).reset_index()
        if not np.allclose(audit["attention_sum"],1,atol=1e-5): raise ValueError("Exported valid attention weights do not sum to one")
    return dict(participants=part,participant_errors=errors,pd_severity=severity,by_study=study,
                seed_stability=stability,inner_loss=inner,attention_audit=audit)


def paired_tables(ns,family,reference_path=None):
    """Explicit label/participant/fold alignment, even across different training seeds."""
    part=ns["participants"]
    contrasts=[]
    if family=="factorial":
        for a,b in [("global__test","local__test"),("global__inner","local__inner"),
                    ("global__test","global__inner"),("local__test","local__inner"),("global__test","local__inner")]:
            if {a,b}<=set(part.condition):
                contrasts.append((a+" minus "+b,[(part[part.condition==a],part[part.condition==b])]))
    elif reference_path and Path(reference_path).is_file():
        ref=pd.read_csv(reference_path)
        required={"family","seed","file","subject","label","fold","probability"}
        if not required<=set(ref): raise ValueError("Reference OOF schema is incomplete")
        if ref.duplicated(["family","seed","file"]).any(): raise ValueError("Duplicated reference predictions")
        if not np.isfinite(ref.probability).all() or not ref.probability.between(0,1).all(): raise ValueError("Invalid reference probabilities")
        for name,g in ref.groupby("family"):
            pairs=[]
            seeds=sorted(set(part.seed)&set(g.seed))
            if not seeds: continue
            for seed in seeds:
                old=ns["participant_oof"](g[g.seed==seed],["family","seed"])
                pairs.append((part[part.seed==seed],old))
            contrasts.append((family+" minus "+str(name),pairs))
    rows=[]
    for title,pairs in contrasts:
        aligned=[]; identities=None
        for a,b in pairs:
            aa=a.set_index("subject").sort_index(); bb=b.set_index("subject").sort_index()
            if not aa.index.equals(bb.index) or aa.index.has_duplicates or bb.index.has_duplicates:
                raise ValueError("Reference cohort/participant alignment differs; do not silently take the intersection")
            if not np.array_equal(aa.label,bb.label) or not np.array_equal(aa.fold,bb.fold):
                raise ValueError("Reference diagnosis/fold mismatch")
            if identities is not None and not identities.equals(aa.index): raise ValueError("Seed cohort mismatch")
            identities=aa.index
            aligned.append((aa.label.to_numpy(),aa.probability.to_numpy(),bb.probability.to_numpy()))
        groups=[np.flatnonzero(aligned[0][0]==c) for c in (0,1)]
        names=("accuracy","balanced_accuracy","macro_f1","sensitivity","specificity","roc_auc","log_loss","brier")
        def delta(indices):
            per_seed=[]
            for y,a,b in aligned:
                am,bm=ns["metric_dict"](y[indices],a[indices]),ns["metric_dict"](y[indices],b[indices])
                per_seed.append([am[k]-bm[k] for k in names])
            return np.mean(per_seed,axis=0)
        point=delta(np.arange(len(aligned[0][0])))
        rng=np.random.default_rng(2028)
        samples=np.array([delta(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])) for _ in range(ns["N_BOOT"])])
        for i,name in enumerate(names):
            rows.append(dict(comparison=title,metric=name,difference=float(point[i]),
                 bootstrap_95_low=float(np.percentile(samples[:,i],2.5)),bootstrap_95_high=float(np.percentile(samples[:,i],97.5)),
                 participants=len(identities),paired_seeds=len(aligned)))
    return pd.DataFrame(rows)


def export_tables(ns,family):
    out=Path(ns["CHECKPOINT_DIR"])/"reports"; out.mkdir(parents=True,exist_ok=True)
    names=["summary","participant_ci","participant_fold_metrics","participants","participant_errors", "pd_severity",
           "by_study","seed_stability","inner_loss","paired_comparisons","attention_audit"]
    names += ["oof","fold_metrics"] if family=="factorial" else ["nested_oof","trial_table","selection_table","inner_selected_oof","attention_oof"]
    for name in names:
        value=ns.get(name)
        if isinstance(value,pd.DataFrame):
            value.to_csv(out/("recording_oof.csv" if name in ("oof","nested_oof") else name+".csv"),index=False)
    atomic_json(out/"data_provenance.json",ns.get("data_provenance",{}))
    print("Analysis tables:",out.resolve())


def export_checkpoint_bundle(root):
    """Explicit user action; archive models, plan and atomic task/epoch states."""
    import zipfile
    root=Path(root)
    if not (root/"run_plan.json").is_file(): raise ValueError("No verified checkpoint run exists")
    target=root.parent/(root.name+"-"+time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]+".zip")
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path,path.relative_to(root))
    return target


def restore_checkpoint_bundle(bundle,root):
    """Restore a trusted own-run bundle only into a new/empty directory."""
    import zipfile
    root=Path(root).resolve()
    if root.exists() and any(root.iterdir()): raise ValueError("Restore target must be empty; existing checkpoints are not overwritten")
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            dest=(root/info.filename).resolve()
            if not dest.is_relative_to(root) or info.filename.startswith(("/","\\")):
                raise ValueError("Unsafe checkpoint archive path")
            if (info.external_attr>>16)&0o170000==0o120000: raise ValueError("Archive symlinks are forbidden")
        if "run_plan.json" not in archive.namelist(): raise ValueError("Not a verified run bundle")
        root.mkdir(parents=True,exist_ok=True)
        archive.extractall(root)
    return root