from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import time
import json
import h5py
import string
import numpy as np
np.set_printoptions(precision=4)
from tqdm import tqdm
import random
import torch.nn.functional as F

import torch
import torch.utils.data as data

from src.dataset.abstract_dataset import AbstractDataset
from src.utils import utils, io_utils

def create_loaders(split, loader_configs, num_workers):
    dsets, L = {}, {}
    for di,dt in enumerate(split):
        shuffle = True if dt == "train" else False
        drop_last = True if dt == "train" else False
        dsets[dt] = CharadesDataset(loader_configs[di])
        L[dt] = data.DataLoader(
            dsets[dt],
            batch_size = loader_configs[di]["batch_size"],
            num_workers = num_workers,
            shuffle = shuffle, # shuffle
            collate_fn = dsets[dt].collate_fn,
            drop_last= drop_last #drop_last
        )
    return dsets, L


class CharadesDataset(AbstractDataset):
    def __init__(self, config):
        super(self.__class__, self).__init__(config)
        # get options
        self.config = config
        self.S = config.get("num_segment", 128)
        self.split = config.get("split", "train")
        self.data_dir = config.get("data_dir", "data/charades")
        self.feature_type = config.get("feature_type", "I3D")
        self.in_memory = config.get("in_memory", False)
        self.anno_path = config.get("anno_path", "data/charades/pseudo/charades_train_pseudo_unique.json")
        self.sample_num = config.get("sample_num", 10)
        self.clip_video_feature_path = config.get("clip_video_feature_path", "data/charades/charades_clip_video.h5")
        self.clip_text_feature_path = config.get("clip_text_feature_path", "data/charades/charades_clip_text.h5")
        
        self.use_corpus = config.get("use_corpus", False)
        self.sen_corpus_feat_path = config.get("sentences_corpus_feat_path", "data/charades/charades_corpus_feat.npy")
        
        if self.feature_type == "I3D":
            self.feat_path = config.get(
                "video_feature_path",
                "data/charades/i3d_features.hdf5"
            )
        else:
            raise ValueError("Wrong feature_type")
        
        self.video_features = h5py.File(self.feat_path, 'r')

        self.annos = self._read_annos(self.anno_path)

        self.batch_size = config.get("batch_size", 64)
        self.num_instances = len(self.annos)

        if self.use_corpus:
            self.sen_corpus_feat = np.load(self.sen_corpus_feat_path)

    def _read_annos(self,anno_path):
        # read annotations
        with open(anno_path,'r') as f:
            annos = json.load(f)
        return annos
    
    def _sample_frame_features(self, frames_feat):
        num_clips = self.S
        keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)
        keep_idx = np.round(keep_idx).astype(np.int64)
        keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
        frames_feat1 = []
        for j in range(num_clips):
            s, e = keep_idx[j], keep_idx[j + 1]
            assert s <= e
            if s == e:
                frames_feat1.append(frames_feat[s])
            else:
                frames_feat1.append(frames_feat[s:e].mean(axis=0))
        return np.stack(frames_feat1, 0)
    
    def random_sample_frame_features(self, frames_feats, sample_num, proposal_id):
        if sample_num > len(proposal_id):
            sample_num = len(proposal_id)
        rand_id = np.sort(random.sample(proposal_id, sample_num))
        return frames_feats[rand_id]
    
    def make_attention_mask(self,start_index,end_index):
        attn_mask = np.zeros([self.S])
        attn_mask[start_index:end_index+1] = 1
        attn_mask = torch.Tensor(attn_mask)
        return attn_mask
    
    def get_clip_features(self, vid):
        with h5py.File(self.clip_video_feature_path, 'r') as fr:
            features = np.asarray(fr[vid])
        features = torch.from_numpy(features).float()
        # features = F.normalize(features,dim=1)
        return features

    def __getitem__(self, idx):
        # get query id and corresponding video id
        anno = self.annos[idx]
        vid = anno["vid"]
        duration = anno['duration']
        timestamp = [x*duration for x in anno['timestamp']]
        start_pos, end_pos = anno['timestamp']
        vid_feat = np.asarray(self.video_features[vid]).astype(np.float32)
        
        clip_feat = self.get_clip_features(vid)
        
        vid_feat = self._sample_frame_features(vid_feat)
        nfeats = len(vid_feat)

        clip_feat = self._sample_frame_features(clip_feat)
        proposal_s_id = int((clip_feat.shape[0] * timestamp[0]) / duration)
        proposal_e_id = int((clip_feat.shape[0] * timestamp[1]) / duration)
        if proposal_e_id >= clip_feat.shape[0]:
            proposal_e_id = clip_feat.shape[0] - 1
        if proposal_s_id > proposal_e_id:
            proposal_e_id = proposal_s_id

        proposal_id = range(proposal_s_id, proposal_e_id + 1)

        sampled_frames_feat = self.random_sample_frame_features(clip_feat, self.sample_num, proposal_id)
        sample_len = len(sampled_frames_feat)

        # get video masks
        vid_mask = np.zeros((self.S, 1))
        vid_mask[:nfeats] = 1
        clip_mask = np.ones((self.sample_num))
        clip_mask[:sample_len] = 0
        clip_mask = clip_mask.astype(bool)

        instance = {
            "vids": vid,
            "qids": idx,
            "timestamps": timestamp, # GT location [s, e] (second)
            "duration": duration, # video span (second)
            "grounding_start_pos": torch.FloatTensor([start_pos]), # [1]; normalized
            "grounding_end_pos": torch.FloatTensor([end_pos]),     # [1]; normalized
            "grounding_att_masks": self.make_attention_mask(proposal_s_id, proposal_e_id),  # [L_v]
            "nfeats": torch.FloatTensor([nfeats]),
            "video_feats": torch.FloatTensor(vid_feat), # [L_v,D_v]
            "video_masks": torch.ByteTensor(vid_mask), # [L_v,1]
            "clip_vid_feats": torch.FloatTensor(sampled_frames_feat),
            "clip_masks": torch.BoolTensor(clip_mask),
            "clip_len": sample_len
        }
        if self.split != "train":
            sentence = anno['sentence']
            with h5py.File(self.clip_text_feature_path, 'r') as fr:
                sentences_list = [s.decode('utf-8') for s in list(fr[vid]['sentences'])]
                clip_text_feat = np.asarray(fr[vid]['features']).astype(np.float32)[sentences_list.index(sentence),:] # 512
            instance.update({"clip_text_feat": torch.FloatTensor(clip_text_feat).unsqueeze(0)})
        if self.split == "train" and self.use_corpus:
            instance.update({"sen_corpus_feat": torch.FloatTensor(self.sen_corpus_feat)})
        return instance

    def collate_fn(self, data):
        seq_items = ["video_feats", "video_masks", "grounding_att_masks", "clip_vid_feats", "clip_masks"]
        tensor_items = [
            "nfeats", "grounding_start_pos", "grounding_end_pos", "clip_text_feat"
        ]
        batch = {k: [d[k] for d in data] for k in data[0].keys()}

        if len(data) == 1:
            for k,v in batch.items():
                if k in tensor_items:
                    batch[k] = torch.cat(batch[k], 0)
                elif k in seq_items:
                    batch[k] = torch.nn.utils.rnn.pad_sequence(
                            batch[k], batch_first=True)
                else:
                    batch[k] = batch[k][0]

        else:
            for k in tensor_items:
                if k in batch:
                    batch[k] = torch.cat(batch[k], 0)
            for k in seq_items:
                if k in batch:
                    batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)
        if "sen_corpus_feat" in batch:
            batch["sen_corpus_feat"] = batch["sen_corpus_feat"][0]
        return batch
    

    def __len__(self):
        return len(self.annos)
# for debugging
def get_loader():
    conf = {
        "train_loader": {
            "dataset": "charades",
            "split": "train",
            "batch_size": 1,
            "data_dir": "data/charades",
            "video_feature_path": "data/i3d_features.hdf5",
            "max_length": 10,
            "word_frequency_threshold": 1,
            "num_segment": 128,
            "feature_type": "I3D",
            "anno_path": "data/charades/pseudo/charades_train_pseudo_unique.json"
        },
        "test_loader": {
            "dataset": "charades",
            "split": "test",
            "batch_size": 1,
            "data_dir": "data/charades",
            "video_feature_path": "data/i3d_features.hdf5",
            "max_length": 25,
            "word_frequency_threshold": 1,
            "num_segment": 128,
            "feature_type": "I3D",
            "anno_path": "data/charades/pseudo/charades_test_original.json"
        }
    }
    print(json.dumps(conf, indent=4))
    dsets, L = create_loaders(["train","test"],
                              [conf["train_loader"], conf["test_loader"]],
                              num_workers=5)
    return dsets, L

if __name__ == "__main__":
    i = 1
    dset, l = get_loader()
    bt = time.time()
    st = time.time()
    num_ol = 0
    for batch in l["train"]:
        i += 1
        # print(batch["grounding_start_pos"], batch["grounding_end_pos"])
        if batch["grounding_start_pos"] < 0.0 or batch["grounding_end_pos"] > 1.0:
            num_ol += 1
        st = time.time()
    print("# of outlier in training data: {}/{}".format(num_ol, len(l["train"])))
    i = 1
    num_ol = 0
    for batch in l["test"]:
        i += 1
        if batch["grounding_start_pos"] < 0.0 or batch["grounding_end_pos"] > 1.0:
            num_ol += 1
        st = time.time()
    print("# of outlier in test data: {}/{}".format(num_ol, len(l["test"])))
    print("Total elapsed time ({:.5f}s)".format(time.time() - bt))
