import os
from collections import OrderedDict
import torch
import torch.nn as nn
import time
import numpy as np
from .apt import ATPSelectorModel, ATPConfig

from src.model import building_blocks as bb
from src.model.abstract_network import AbstractNetwork
from src.utils import io_utils, net_utils, vis_utils

class LGI(AbstractNetwork):
    def __init__(self, config, logger=None, verbose=True):
        """ Initialize baseline network for Temporal Language Grounding
        """
        super(LGI, self).__init__(config=config, logger=logger)

        self._build_network()
        self._build_evaluator()

        # create counters and initialize status
        self._create_counters()
        self.reset_status(init_reset=True)

    def _build_network(self):
        """ build network that consists of following four components
        1. encoders - query_enc & video enc
        2. sequential query attentio network (sqan)
        3. local-global video-text interactions layer (vti_fn)
        4. temporal attentive localization by regression (talr)
        """
        self.max_epoch = self.config["optimize"].get("num_step", 500)
        self.dataset_name = self.config["misc"].get("dataset", "charades")
        mconfig = self.config["model"]
        
        mconfig["use_corpus"] = self.config["train_loader"].get("use_corpus", False)
        mconfig["max_epoch"] = self.max_epoch
        self.noise_level = mconfig.get("noise_level", 0.1)
        self.sigma = mconfig.get("sigma", 9)
        self.gamma = mconfig.get("gamma", 0)
        self.vote_thresh = mconfig.get("vote_thresh", 0.146)
        # build video & query encoder
        self.query_enc = LanguageFree(mconfig)
        self.video_enc = bb.VideoEmbeddingWithPosition(mconfig, "video_enc")

        # build sequential query attention network (sqan)
        self.nse = mconfig.get("num_semantic_entity", -1) # number of semantic phrases
        if self.nse > 1:
            self.sqan = bb.SequentialQueryAttention(mconfig)
        
        self.num_proposals = mconfig.get("num_proposals", 1)
        if self.num_proposals > 1:
            self.l1_smooth = nn.SmoothL1Loss(reduction="none")

        # build local-global video-text interactions network (vti_fn)
        self.vti_fn = bb.LocalGlobalVideoTextInteractions(mconfig)

        # build grounding fn
        self.ta_reg_fn = bb.AttentionLocRegressor(mconfig)
        self.sdim = mconfig.get("sentence_emb_dim", 512)
        if self.nse > 1:
            self.txt_concept_enc = TextEncoder(self.sdim, self.nse)

        self.sen_fc = nn.Sequential(*[
            nn.Linear(self.sdim, 2 * self.sdim),
            nn.ReLU(),
            nn.Linear(2 * self.sdim, self.sdim),
            nn.ReLU(),
        ])

        # build criterion
        self.use_tag_loss = mconfig.get("use_temporal_attention_guidance_loss", True)
        self.use_l1_loss = mconfig.get("use_l1_loss", True)
        self.use_nce_loss = mconfig.get("use_nce_loss", True)
        self.use_negative_proposal = mconfig.get("use_negative_proposal", False)
        self.use_proposal_nce = mconfig.get("use_proposal_nce", False)
        self.use_multi_tag = mconfig.get("use_multi_tag", False)
        self.use_attention_div = mconfig.get("use_attention_div", True)
        self.criterion = bb.MultipleCriterions(
            ["grounding"],
            [bb.TGRegressionCriterion(mconfig, prefix="grounding")]
        )
        if self.use_tag_loss:
            self.criterion.add("tag", bb.TAGLoss(mconfig))
        if self.use_l1_loss:
            self.criterion.add("l1", bb.L1Loss(mconfig))
        if self.use_nce_loss:
            self.criterion.add("nce", bb.NCELoss(mconfig))
        if self.use_negative_proposal:
            self.criterion.add("intra nce", bb.IntraNCELoss(mconfig))

        # set model list
        self.model_list = ["video_enc", "query_enc",
                           "vti_fn", "ta_reg_fn", "criterion"]
        self.models_to_update = ["video_enc", "query_enc",
                                 "vti_fn", "ta_reg_fn", "criterion"]
        if self.nse > 1:
            self.model_list.append("sqan")
            self.models_to_update.append("sqan")
            self.models_to_update.append("txt_concept_enc")
            self.model_list.append("text_concept_enc")

        self.log("===> We train [{}]".format("|".join(self.models_to_update)))
        self.iter = 0
    
    def get_index1(self,lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]

    def forward(self, net_inps, gts=None):
        return self._infer(net_inps, "forward", gts)

    def visualize(self, vis_inps, vis_gt, prefix):
        vis_data = self._infer(vis_inps, "visualize", vis_gt)
        vis_utils.visualize_LGI(self.config, vis_data, self.itow, prefix)

    def extract_output(self, vis_inps, vis_gt, save_dir):
        vis_data = self._infer(vis_inps, "save_output", vis_gt)

        qids = vis_data["qids"]
        preds = net_utils.loc2mask(loc, seg_masks)
        for i,qid in enumerate(qids):
            out = dict()
            for k in vis_data.keys():
                out[k] = vis_data[k][i]
            # save output
            save_path = os.path.join(save_dir, "{}.pkl".format(qid))
            io_utils.check_and_create_dir(save_dir)
            io_utils.write_pkl(save_path, out)
    
    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1).clamp(0,1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        if torch.any(torch.isnan(weight)):
                print("weight", weight)
                exit(0)

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight
    
    def caculate_iou_loss(self, loc, gts, t_attw):
        # loc: [B, num_proposal, 2]
        s_gt = gts["grounding_start_pos"]     # [B]
        e_gt = gts["grounding_end_pos"]       # [B]

        multi_t_attw = t_attw
        multi_loc = loc
        # iou_loss: [B, num_proposal]
        iou_loss = self.l1_smooth(loc[:,:,0], s_gt.unsqueeze(1)) + self.l1_smooth(loc[:,:,1], e_gt.unsqueeze(1))
        min_iou_loss, idx = iou_loss.min(dim=1)
        sorted_idx = iou_loss.argsort(dim=-1, descending=False)
        idx = sorted_idx[:,0]
        max_idx = sorted_idx[:,-1]
        min_max_idx = torch.stack((idx, max_idx), dim=-1)
        min_loc = torch.gather(loc, index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,loc.shape[-1]), dim=1).squeeze(1)
        min_tattw = torch.gather(t_attw, index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,t_attw.shape[-1]), dim=1).squeeze(1)
        return min_loc, min_tattw, multi_t_attw, multi_loc, min_max_idx, sorted_idx
    
    def select_best_proposal(self, summarized_vfeat, sen_feat, t_attw, loc):
        # [B,num_proposal,D], [B,D]
        multi_t_attw = t_attw
        multi_loc = loc
        summarized_vfeat = summarized_vfeat / summarized_vfeat.norm(dim=-1, keepdim=True)
        sen_feat = sen_feat / sen_feat.norm(dim=-1, keepdim=True)
        cos_sim = torch.bmm(summarized_vfeat, sen_feat.unsqueeze(-1)).squeeze(-1)
        # loss = torch.sigmoid(iou_score)
        idx = cos_sim.argsort(dim=-1, descending=False)
        sorted_loc = loc.gather(index=idx.unsqueeze(-1).expand(-1,-1,loc.shape[-1]), dim=1)
        sorted_tattw = t_attw.gather(index=idx.unsqueeze(-1).expand(-1,-1,t_attw.shape[-1]), dim=1)

        return sorted_loc, sorted_tattw, multi_t_attw, multi_loc
    
    def calculate_IoU_batch(self, i0, i1):
        union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
        inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
        iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
        iou[union[1] - union[0] < -1e-5] = 0
        iou[iou < 0] = 0.0
        return iou
    
    def random_sample_frame_features(self, frames_feats, sample_num, proposal_id):
        # frames_feats bsz, L, D
        import random
        tmp_sample_num = sample_num
        if sample_num > len(proposal_id):
            sample_num = len(proposal_id)
        rand_id = np.sort(random.sample(proposal_id, sample_num))
        if rand_id.shape[0] < tmp_sample_num:
            rand_id = np.append(rand_id, np.array([proposal_id[-1]]*(tmp_sample_num-rand_id.shape[0])))
        return torch.Tensor(frames_feats[rand_id])

    def _infer(self, net_inps, mode="forward", gts=None):
        # fetch inputs
        c3d_feats = net_inps["video_feats"]  # [B,T,d_v]
        seg_masks = net_inps["video_masks"].squeeze(2) # [B,T]
        B, nseg, _ = c3d_feats.size() # nseg == T

        # forward encoders
        # get word-level, sentence-level and segment-level features
        # word_feats, sen_feats = self.query_enc(word_labels, word_masks, "both") # [B,L,*]
        sen_feats, word_masks, cos_similarity = self.query_enc(net_inps)

        seg_feats = self.video_enc(c3d_feats, seg_masks) # [B,nseg,*]

        # get semantic phrase features:
        # se_feats: semantic phrase features [B,nse,*];
        #           ([e^1,...,e^n]) in Eq. (7)
        # se_attw: attention weights for semantic phrase [B,nse,nword];
        #           ([a^1,...,a^n]) in Eq. (6)
        # if self.nse > 1:
        #     se_feats, se_attw = self.sqan(sen_feats, word_feats, word_masks)
        if self.nse > 1:
            se_feats = self.txt_concept_enc(sen_feats, word_masks)
            se_attw = None
        else: se_attw = None

        # Local-global video-text interactions
        # sa_feats: semantics-aware segment features [B,nseg,d]; R in Eq. (12)
        # s_attw: aggregating weights [B,nse]
        if self.nse > 1:
            q_feats = se_feats
        else:
            q_feats = sen_feats
        sa_feats, s_attw = self.vti_fn(seg_feats, seg_masks, q_feats)
        # Temporal attentive localization by regression
        # loc: prediction of time span (t^s, t^e)
        # t_attw: temporal attention weights (o)
        loc, t_attw, summarized_vfeat, attw_norm = self.ta_reg_fn(sa_feats, seg_masks)
        if self.use_multi_tag:
            bsz, props_len = sa_feats.shape[:2]
            if self.num_proposals > 1:
                loc_tmp = loc.reshape(bsz*self.num_proposals, -1)
            else:
                loc_tmp = loc
            center = torch.clamp((loc_tmp[:, 0] + loc_tmp[:, 1]) / 2., min=0, max=1)
            width = torch.abs(loc_tmp[:, 0] - loc_tmp[:, 1])
            gauss_weight = self.generate_gauss_weight(props_len, center, width)
            multi_gauss_weight = gauss_weight.reshape(bsz, self.num_proposals, -1)
            attw_norm = attw_norm.reshape(bsz, self.num_proposals, -1)

        self.iter+=1
                    
        if self.use_nce_loss and self.training:
            video_feats = (t_attw.unsqueeze(-1) * sa_feats).sum(dim=1) # [B,d]
        if self.use_negative_proposal and self.training:
            pos_center = torch.clamp((loc[:, 0] + loc[:, 1]) / 2., min=0, max=1)
            pos_width = torch.abs(loc[:, 0] - loc[:, 1])
            bsz, props_len = sa_feats.shape[:2]
            epoch = net_inps["epoch"]
            gauss_weight = self.generate_gauss_weight(props_len, pos_center, pos_width)
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, pos_center, pos_width, epoch)
            pos_feat = (gauss_weight.unsqueeze(-1) * sa_feats).sum(dim=1)
            neg_1_feat = (neg_1_weight.unsqueeze(-1) * sa_feats).sum(dim=1)
            neg_2_feat = (neg_2_weight.unsqueeze(-1) * sa_feats).sum(dim=1)
            ref_feat = sa_feats.sum(dim=1)

        if mode == "forward":
            outs = OrderedDict()
            outs["grounding_loc"] = loc
            if self.training:
                if self.use_tag_loss:
                    outs["epoch"] = net_inps["epoch"]
                    outs["tag_attw"] = t_attw
                if self.use_l1_loss:
                    outs["cos_similarity"] = cos_similarity
                if self.use_nce_loss:
                    outs["video_feats"] = video_feats
                    outs["sen_feats"] = sen_feats
                if self.use_negative_proposal:
                    outs["pos_feat"] = pos_feat
                    outs["neg_1_feat"] = neg_1_feat
                    outs["neg_2_feat"] = neg_2_feat
                    outs["ref_feat"] = ref_feat
                    if "sen_feats" not in outs:
                        outs["sen_feats"] = sen_feats
                
        else:
            outs = dict()
            outs["vids"] = gts["vids"]
            outs["qids"] = gts["qids"]
            outs["grounding_gt"] = net_utils.to_data(gts["grounding_att_masks"])
            outs["grounding_pred"] = net_utils.loc2mask(loc, seg_masks)
            outs["nfeats"] = gts["nfeats"]
            outs["t_attw"] = net_utils.to_data(t_attw.unsqueeze(1))
            if s_attw is None:
                outs["s_attw"] = net_utils.to_data(t_attw.new_zeros(t_attw.size(0),2,4))
            else:
                outs["s_attw"] = net_utils.to_data(s_attw)

            if mode == "save_output":
                outs["duration"] = gts["duration"]
                outs["timestamps"] = gts["timestamps"]
                outs["grounding_pred_loc"] = net_utils.to_data(loc)

        return outs

    def prepare_batch(self, batch):
        self.gt_list = ["vids", "qids", "timestamps", "duration",
                   "grounding_start_pos", "grounding_end_pos",
                   "grounding_att_masks", "nfeats"]
        self.both_list = ["grounding_att_masks"]

        net_inps, gts = {}, {}
        for k in batch.keys():
            item = batch[k].to(self.device) \
                if net_utils.istensor(batch[k]) else batch[k]

            if k in self.gt_list: gts[k] = item
            else: net_inps[k] = item

            if k in self.both_list: net_inps[k] = item

        if self.use_tag_loss:
            gts["tag_att_masks"] = gts["grounding_att_masks"]
        return net_inps, gts

    """ methods for status & counters """
    def reset_status(self, init_reset=False):
        """ Reset (initialize) metric scores or losses (status).
        """
        super(LGI, self).reset_status(init_reset=init_reset)

        # initialize prediction maintainer for each epoch
        self.results = {"timestamps":[], "predictions": [], "gts": [],
                        "durations": [], "vids": [], "qids": []}

    def compute_status(self, net_outs, gts, mode="Train"):

        # fetch data
        loc = net_outs["grounding_loc"].detach()
        B = loc.size(0)
        gt_ts = gts["timestamps"]
        vid_d = gts["duration"]

        # prepare results for evaluation
        if B == 1:
            ii = 0
            if self.training:
                pred = [[float(loc[ii,0])*vid_d[ii], float(loc[ii,1])*vid_d[ii]]]
                timestamps = [[float(loc[ii,0]), float(loc[ii,1])]]
            else:
                if len(loc.shape) == 3:
                    pred = (loc[ii, :, :]*vid_d[ii]).tolist()
                    timestamps = (loc[ii, :, :]).tolist()
                else:
                    pred = [[float(loc[ii,0])*vid_d[ii], float(loc[ii,1])*vid_d[ii]]]
                    timestamps = [[float(loc[ii,0]), float(loc[ii,1])]]
            self.results["timestamps"].append(timestamps)
            self.results["predictions"].append(pred)
            self.results["gts"].append(gt_ts)
            self.results["durations"].append(vid_d)
            self.results["vids"].append(gts["vids"][ii])
            self.results["qids"].append(gts["qids"][ii])
            return
        for ii in range(B):
            if self.training:
                pred = [[float(loc[ii,0])*vid_d[ii], float(loc[ii,1])*vid_d[ii]]]
                timestamps = [[float(loc[ii,0]), float(loc[ii,1])]]
            else:
                if len(loc.shape) == 3:
                    pred = (loc[ii, :, :]*vid_d[ii]).tolist()
                    timestamps = (loc[ii, :, :]).tolist()
                else:
                    pred = [[float(loc[ii,0])*vid_d[ii], float(loc[ii,1])*vid_d[ii]]]
                    timestamps = [[float(loc[ii,0]), float(loc[ii,1])]]
                
            self.results["timestamps"].append(timestamps)
            self.results["predictions"].append(pred)
            self.results["gts"].append(gt_ts[ii])
            self.results["durations"].append(vid_d[ii])
            self.results["vids"].append(gts["vids"][ii])
            self.results["qids"].append(gts["qids"][ii])

    def save_results(self, prefix, mode="Train"):
        # save predictions
        # save_dir = os.path.join(self.config["misc"]["result_dir"], "predictions", mode)
        # save_to = os.path.join(save_dir, prefix+".json")
        # io_utils.check_and_create_dir(save_dir)
        # io_utils.write_json(save_to, self.results)

        # compute performances
        nb = float(len(self.results["gts"]))
        self.evaluator.set_duration(self.results["durations"])
        rank1, _, miou = self.evaluator.eval(
                self.results["predictions"], self.results["gts"])

        for k,v  in rank1.items():
            self.counters[k].add(v/nb, 1)
        self.counters["mIoU"].add(miou/nb, 1)

    def renew_best_score(self):
        cur_score = self._get_score()
        if (self.best_score is None) or (cur_score > self.best_score):
            self.best_score = cur_score
            self.log("Iteration {}: New best R1 score {:4f}".format(self.it, self.best_score))
            return True
        self.log("Iteration {}: Current score {:4f}".format(self.it, cur_score))
        self.log("Iteration {}: Current best score {:4f}".format(
                self.it, self.best_score))
        return False

    """ methods for updating configuration """
    def bring_dataset_info(self, dset):
        super(LGI, self).bring_dataset_info(dset)

    def model_specific_config_update(self, config):
        mconfig = config["model"]
        ### Video Encoder
        vdim = mconfig["video_enc_vemb_odim"]
        ### Query Encoder
        mconfig["query_enc_rnn_idim"] = mconfig["query_enc_emb_odim"]
        qdim = 2 * mconfig["query_enc_rnn_hdim"]

        ### Sequential Query Attention Network (SQAN)
        mconfig["sqan_qdim"] = qdim
        mconfig["sqan_att_cand_dim"] = qdim
        mconfig["sqan_att_key_dim"] = qdim

        ### Local-Global Video-Text Interactions
        # Segment-level Modality Fusion
        self.lgi_fusion_method = mconfig.get("lgi_fusion_method", "concat")
        mdim = vdim # dim of fused multimodal feature
        if self.lgi_fusion_method == "mul":
            mconfig["lgi_hp_idim_1"] = vdim
            mconfig["lgi_hp_idim_2"] = qdim
            mconfig["lgi_hp_hdim"] = vdim
        # Local Context Modeling
        l_type = mconfig.get("lgi_local_type", "res_block")
        if l_type == "res_block":
            mconfig["lgi_local_res_block_1d_idim"] = mdim
            mconfig["lgi_local_res_block_1d_odim"] = mdim
            l_odim = mconfig["lgi_local_res_block_1d_odim"]
        elif l_type == "masked_nl":
            mconfig["lgi_local_nl_odim"] = mdim
            l_odim = mconfig["lgi_local_nl_odim"]
        else:
            l_odim = mdim

        # Global Context Modeling
        g_type = mconfig.get("lgi_global_type", "nl")
        mconfig["lgi_global_satt_att_cand_dim"] = l_odim
        mconfig["lgi_global_nl_idim"] = l_odim

        ### Temporal Attention based Regression
        mconfig["grounding_att_key_dim"] = qdim
        mconfig["grounding_att_cand_dim"] = qdim
        if mconfig.get("lgi_global_satt_att_use_embedding", False):
            mconfig["grounding_idim"] = mconfig["lgi_global_satt_att_edim"]
        else:
            mconfig["grounding_idim"] = mconfig["lgi_global_satt_att_cand_dim"]

        return config

    @staticmethod
    def dataset_specific_config_update(config, dset):
        mconfig = config["model"]
        # Query Encoder
        # mconfig["query_enc_emb_idim"] = len(list(dset.wtoi.keys()))
        # mconfig["loc_word_emb_vocab_size"] = len(list(dset.wtoi.keys()))
        return config

class LanguageFree(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.noise_level = self.cfg.get("noise_level", 0.04)
        self.use_clip_mask = self.cfg.get("use_clip_mask", False)
        self.learned_noise = self.cfg.get("learned_noise", True)
        self.use_corpus = self.cfg.get("use_corpus", False)
        self.mlp1 = MLP(cfg)
        self.mlp2 = MLP(cfg)

        self.cos = nn.CosineSimilarity(dim = -1)

        self.selector = ATPSelectorModel(ATPConfig)

        self.sdim = self.cfg.get("sentence_emb_dim", 512)
        self.sodim = self.cfg.get("sentence_emb_odim", 512)

        if self.use_corpus:
            # original
            self.sen_fc = nn.Sequential(*[
                nn.Linear(self.sdim * 2, 2 * self.sdim),
                nn.ReLU(),
                nn.Linear(2 * self.sdim, self.sodim),
                nn.ReLU(),
            ])
        else:
            self.sen_fc = nn.Sequential(*[
                nn.Linear(self.sdim, 2 * self.sdim),
                nn.ReLU(),
                nn.Linear(2 * self.sdim, self.sodim),
                nn.ReLU(),
            ])

    def add_noise(self, clip_vis_feats):
        # normalize the visual features
        clip_vis_feats = clip_vis_feats/clip_vis_feats.norm(dim=-1, keepdim=True)

        noise = torch.normal(0., 1., size=(clip_vis_feats.shape[0], clip_vis_feats.shape[1], clip_vis_feats.shape[2])).to(self.device)
        revised_frame_fts = (1-self.noise_level)*clip_vis_feats + \
                        self.noise_level*noise/noise.norm(dim=-1, keepdim=True) 
        noise_clip_vis_feats = revised_frame_fts/revised_frame_fts.norm(dim=-1, keepdim=True)
        return noise_clip_vis_feats
    
    def noise_func(self, clip_vis_feats):
        # Reparameterization Trick
        var = self.mlp1(clip_vis_feats)
        mu = self.mlp2(clip_vis_feats)
        std = var.mul(0.5).exp_()
        eps = torch.normal(0., 1., size=(clip_vis_feats.shape[0], clip_vis_feats.shape[1], clip_vis_feats.shape[2])).to(self.device)

        noise = eps.mul(std).add_(mu)

        clip_vis_feats = clip_vis_feats/clip_vis_feats.norm(dim=-1, keepdim=True) 
        revised_frame_fts = (1-self.noise_level)*clip_vis_feats + \
                        self.noise_level*noise/noise.norm(dim=-1, keepdim=True) 
        noise_clip_vis_feats = revised_frame_fts/revised_frame_fts.norm(dim=-1, keepdim=True)

        return noise_clip_vis_feats

    def find_best_sen_feat(self, selected_clip_vis_feats, sen_corpus_feat):
        sim_matrix = selected_clip_vis_feats @ sen_corpus_feat.T
        max_idx = torch.argmax(sim_matrix, dim=-1)
        most_similar = torch.gather(sen_corpus_feat,  index=max_idx.unsqueeze(-1).expand(-1, sen_corpus_feat.shape[-1]), dim=0)
        return most_similar

    def forward(self,inputs):
        clip_vis_feats = inputs['clip_vid_feats'] # B, 9, D
        clip_masks = inputs['clip_masks']
        self.device = clip_vis_feats.device
        if not self.training:
            text_feat = inputs['clip_text_feat']
            assert text_feat.sum() != 0
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_mask = torch.ones(text_feat.shape[0],1, 1).to(self.device)
            clip_vis_feats = clip_vis_feats.mean(dim=1)
            cos_sim = self.cos(clip_vis_feats, text_mask)
            if self.use_corpus:
                text_feat_tmp = torch.cat((text_feat, text_feat),dim=1)
                text_feat = self.sen_fc(text_feat_tmp)
            else:
                text_feat = self.sen_fc(text_feat)
            return text_feat, text_mask, cos_sim

        if self.use_corpus:
            sen_corpus_feat = inputs["sen_corpus_feat"]
            sen_corpus_feat = sen_corpus_feat / sen_corpus_feat.norm(dim=-1, keepdim=True)
        if self.learned_noise:
            noise_clip_vis_feats = self.noise_func(clip_vis_feats)
        else:
            noise_clip_vis_feats = self.add_noise(clip_vis_feats)

        # selected_clip_vis_feats [bsz, 512]
        if self.use_clip_mask:
            selected_clip_vis_feats, _ = self.selector(noise_clip_vis_feats, clip_masks)[0:2]
            clip_vis_feats_sum = self.selector(clip_vis_feats, clip_masks)[0]
        else:
            selected_clip_vis_feats, _ = self.selector(noise_clip_vis_feats)[0:2]
            clip_vis_feats_sum = self.selector(clip_vis_feats)[0]

        # compute the similarity
        clip_vis_feats_sum = clip_vis_feats_sum/clip_vis_feats_sum.norm(dim=-1, keepdim=True)
        selected_clip_vis_feats = selected_clip_vis_feats/selected_clip_vis_feats.norm(dim=-1, keepdim=True)
        cos_sim = self.cos(clip_vis_feats_sum, selected_clip_vis_feats)
        
        if self.use_corpus:
            sim_feat = self.find_best_sen_feat(selected_clip_vis_feats, sen_corpus_feat)
            pseudo_sen_emb = torch.cat((sim_feat, selected_clip_vis_feats), dim=-1)
            pseudo_sen_emb = self.sen_fc(pseudo_sen_emb)
        else:
            pseudo_sen_emb = self.sen_fc(selected_clip_vis_feats)

        # re-generate mask
        text_mask = torch.ones(selected_clip_vis_feats.shape[0],1, 1).to(self.device)
        return pseudo_sen_emb, text_mask, cos_sim
    
class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super(MLP,self).__init__()    # 
        self.sdim = cfg.get("sentence_emb_dim", 512)
        self.fc1 = torch.nn.Linear(self.sdim,256)
        self.fc2 = torch.nn.Linear(256,128)
        self.fc3 = torch.nn.Linear(128,256)
        self.fc4 = torch.nn.Linear(256,self.sdim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.relu(self.fc4(x))
        return out
        
def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:int(l)] = 1
        mask = torch.stack(mask, 0)
    return mask

class TextEncoder(nn.Module):
    def __init__(self, d_model, concept_nums = 3, dropout=0.1):
        super().__init__()
        self.concept_nums = concept_nums

        self.txt_necks = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU(),
            nn.Linear(d_model, self.concept_nums*d_model, bias=True),
            nn.ReLU()
        )

    def forward(self, txt = None, txt_mask = None):

        if txt != None and txt_mask != None:
            txt_concept = self.txt_necks(txt)
            txt_concepts = torch.stack(torch.split(txt_concept, self.concept_nums, dim=-1), dim=1).permute(0, 2, 1)
        else:
            txt_concepts = None

        return txt_concepts