#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

# refine=False model.path=$(pwd)/lama/big-lama indir=$(pwd)/lama/LaMa_test_images outdir=$(pwd)/lama/output
import logging
import os
import sys
import traceback
cur_path = os.path.dirname(os.path.abspath(__file__))+'/..'
sys.path.insert(0,cur_path)
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.evaluation.file_utils import write
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

@hydra.main(config_path=f'{os.path.dirname(os.path.abspath(__file__))}/configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        assert predict_config['root'],'please input root file, use root=XX'
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f'using {device}')
        predict_config.model.path = os.path.dirname(os.path.abspath(__file__))+f'/../../checkpoints/big-lama'
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')

        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        # if not predict_config.indir.endswith('/'):
        #     predict_config.indir += '/'
        predict_config.indir = predict_config['root']
        predict_config.outdir = os.path.join(predict_config['root'],'result') if not predict_config['output'] else predict_config['output']
        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        
        for img_i in tqdm.trange(len(dataset)):
            img_fname = dataset.img_filenames[img_i]
            name = os.path.basename(os.path.dirname(os.path.dirname(img_fname)))
            cur_out_fname = os.path.join(
                predict_config.outdir, name,os.path.basename(img_fname)
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            write(cur_out_fname,cur_res)
            # cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            # cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(cur_out_fname, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
# refine=False model.path=$(pwd)/lama/big-lama indir=$(pwd)/lama/LaMa_test_images outdir=$(pwd)/lama/output