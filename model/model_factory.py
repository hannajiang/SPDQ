# multi-path paradigm
# from model.clip_multi_path import CLIP_Multi_Path
# from model.coop_multi_path import COOP_Multi_Path
import shutil
from model.troika import Troika
from model.troika_em import Troika_EM
from model.troika_mpt import Troika_MPT
from model.troika_qformer import Troika_QFormer
from model.troika_ipg import Troika_IPG
from model.troika_base import Troika_Base
from model.troika_dp import Troika_DyP
from model.troika_visual_only import Troika_Visual_Only
from model.troika_mutualqf import Troika_MQF
from model.troika_mmqf import Troika_MMQF
from model.troika_mqf import Troika_MQF_O
from model.troika_dpqd import Troika_DPQD
from model.troika_dpqd_dpe import Troika_DPQD_DPE
from model.troika_dp_ot import Troika_DP_OT
from model.troika_pro_ot import Troika_Pro_OT
from model.troika_dp_sa import Troika_DPSA
from model.troika_pro_ot_stageII import Troika_Pro_OT_StageII
from model.troika_tpt import Troika_TPT
from model.troika_clip import Troika_CLIP

import torch

def get_model(config, attributes, classes, offset):
    if config.model_name == 'troika':
        model = Troika(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_em':
        shutil.copy('./model/troika_em.py', config.save_path)
        model = Troika_EM(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_mpt':
        model = Troika_MPT(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_qformer':
        shutil.copy('./model/troika_qformer.py', config.save_path)
        model = Troika_QFormer(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_ipg':
        shutil.copy('./model/troika_ipg.py', config.save_path)
        model = Troika_IPG(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_base':
        shutil.copy('./model/troika_base.py', config.save_path)
        model = Troika_Base(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_dp':
        shutil.copy('./model/troika_dp.py', config.save_path)
        model = Troika_DyP(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_visual_only':
        shutil.copy('./model/troika_visual_only.py', config.save_path)
        model = Troika_Visual_Only(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_mqf':
        shutil.copy('./model/troika_mutualqf.py', config.save_path)
        model = Troika_MQF(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_mmqf':
        shutil.copy('./model/troika_mmqf.py', config.save_path)
        model = Troika_MMQF(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_mqf_o':
        shutil.copy('./model/troika_mqf.py', config.save_path)
        model = Troika_MQF_O(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_dpqd':
        shutil.copy('./model/troika_dpqd.py',config.save_path)
        model = Troika_DPQD(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_dpqd_dpe':
        shutil.copy('./model/troika_dpqd_dpe.py',config.save_path)
        model = Troika_DPQD_DPE(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_dpot':
        shutil.copy('./model/troika_dp_ot.py',config.save_path)
        model = Troika_DP_OT(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_pro':
        shutil.copy('./model/troika_pro_ot.py',config.save_path)
        model = Troika_Pro_OT(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_dp_sa':
        shutil.copy('./model/troika_dp_sa.py',config.save_path)
        model = Troika_DPSA(config, attributes=attributes, classes=classes, offset=offset)
    
    elif config.model_name == 'troika_dp_pro_two_stage':
        shutil.copy('./model/troika_dp_sa.py',config.save_path)
        shutil.copy('./model/troika_pro_ot_stageII.py',config.save_path)
        model_original = Troika_DPSA(config, attributes=attributes, classes=classes, offset=offset)

        if config.load_model is not None:
            model_original.load_state_dict(torch.load(config.load_model))
            # print("Evaluating val dataset:")
            # val_result = evaluate(model, val_dataset, config)
            # print("Evaluating test dataset:")
            # test_result = evaluate(model, test_dataset, config)
        model = Troika_Pro_OT_StageII(config, attributes=attributes, classes=classes, offset=offset, original_model=model_original)
    # elif config.model_name == 'clip_multi_path':
    # elif config.model_name == 'clip_multi_path':
    #     model = CLIP_Multi_Path(config, attributes=attributes, classes=classes, offset=offset)
    # elif config.model_name == 'coop_multi_path':
    #     model = COOP_Multi_Path(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'troika_clip':
        shutil.copy('./model/troika_clip.py',config.save_path)
        model = Troika_CLIP(config, attributes=attributes, classes = classes, offset=offset)
        
    elif config.model_name == 'troika_tpt':
        shutil.copy('./model/troika_tpt.py',config.save_path)
        model = Troika_TPT(config, attributes=attributes, classes=classes, offset=offset)
        
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )


    return model