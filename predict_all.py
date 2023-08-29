import argparse
from argparse import Namespace
import dill
import hydra
import tqdm
import torch
import torchaudio
import os
from pathlib import Path
import json
import numpy as np
from utils import (detect_peaks, max_min_norm, replicate_first_k_frames, PrecisionRecallMetric)
from next_frame_classifier import NextFrameClassifier
from torch.utils.data import DataLoader
from dataloader import *


@hydra.main(config_path='conf/config_test.yaml', strict=False)
def main(cfg):
    print(cfg)
    timit_path = Path(cfg.timit_path)
    run_dir = Path(cfg.out_path)

    ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**(ckpt["hparams"]))

    for split in ['train', 'test']:        
        # load weights and peak detection params
        model = NextFrameClassifier(hp)
        weights = ckpt["state_dict"]
        weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
        model.load_state_dict(weights)
        peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']
       
        pr = PrecisionRecallMetric()

        dataset = WavPhnDataset(timit_path / split)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn_padd,
        )
        
        progress = tqdm(ncols=80, total=len(data_loader))
        in_frame_rate = 0.01
        out_frame_rate = 0.02
        with open(run_dir / f"{split}.src", "w") as f_src:
            for b_idx, batch in enumerate(data_loader):
                audios = batch[0]
                segments = batch[1]
                lengths = batch[3]
                fnames = batch[4]

                preds = model(audios)
                preds = preds[1][0]
                preds = replicate_first_k_frames(preds, k=1, dim=1)
                preds = 1 - max_min_norm(preds)

                pr.update(segments, preds, lengths)
                peaks_batch = detect_peaks(
                    x=preds,
                    lengths=lengths,
                    prominence=peak_detection_params["prominence"],
                    width=peak_detection_params["width"],
                    distance=peak_detection_params["distance"],
                )

                for peaks, length, fname in zip(peaks_batch, lengths, fnames):
                    segment_idx = 0
                    offset = 0
                    out_frames = []
                    peaks = peaks.tolist()
                    peaks.append(length)
                    
                    for peak in peaks:
                        size = peak - offset
                        out_frames.extend([str(segment_idx)]*size)
                        offset = peak
                        segment_idx += 1
                    out_frames = out_frames[::int(out_frame_rate / in_frame_rate)]
                    print(" ".join([fname]+out_frames), file=f_src)
                progress.update(1)
            progress.close()
        scores, best_params = pr.get_stats()

        info = f'Boundary Precision -- {scores[0]*100:.2f}\tRecall -- {scores[1]*100:.2f}\tF1 -- {scores[2]*100:.2f}\tRval -- {scores[3]*100:.2f}'
        print(info)
        with open(run_dir / 'results.txt', 'a') as f:
            f.write(info+'\n')

if __name__ == "__main__":
    main()                
