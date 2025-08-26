#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd

# Local imports
from util.visHelper import (
    load_df, rm_cnt_ol_db, get_bbx, suture_pca, load_ALI,
    get_vessel_length, vis_all, reorder_cnt_sutures, vis_clean,
    calculate_and_plot_distributions_with_gaussian, vis_clean_tempt,
    vis_ALI, vis_distribution_with_label, vis_metric_with_label,
    max_curvature_angle, vis_ali_5, vis_ali_3, vis_ali_1, vis_ali_4, vis_ali_6
)


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DATASET_PATH = Path(
    "/home/luyin/Desktop/ly_torch/ly/scripts/Image_Processing/contrast/dataset/single_split_ClujDec/train"
)
INFER_PATH = DATASET_PATH.parent / f"{DATASET_PATH.name}_infer"
NP_PATH = INFER_PATH / "np"
ALI_CSV = Path("/home/luyin/Desktop/0A_new_ALI.csv")

ALI_MISTAKES = {
    "Mistake 1": "(Line Disruption)",
    "Mistake 3": "(Oblique Stitch)",
    "Mistake 4": "(Too Wide Stitch)",
    "Mistake 5": "(Partial Thickness)",
    "Mistake 6": "(Unequal Distancing)",
}


# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------
def run_postprocessing(verbose: bool = False):
    # Load and process contours
    cnt_all = load_df(DATASET_PATH, NP_PATH, verbose)
    # cnt_all = rm_cnt_ol_db(cnt_all)  # optional DBSCAN outlier removal
    cnt_all = get_bbx(cnt_all, "cnt_sutures")
    cnt_all = get_bbx(cnt_all, "cnt_vessels")
    cnt_all = suture_pca(cnt_all, INFER_PATH / "output_pca", vis=False)
    cnt_all = reorder_cnt_sutures(cnt_all)
    cnt_all = get_vessel_length(cnt_all)

    # Visualization: overall sutures & clean view
    vis_all(cnt_all, INFER_PATH / "output_all", vis=True)
    vis_clean_tempt(cnt_all, INFER_PATH / "output_clean", vis=True)

    # Load ALI scoring data
    df_ALI = load_ALI(ALI_CSV)

    # ---------------------------------------------------------------------
    # ALI ERROR VISUALIZATIONS
    # ---------------------------------------------------------------------
    # 1: Line Disruption (Curvature)
    vis_ali_1(
        cnt_all,
        df_ALI,
        1,
        INFER_PATH / "output_ALI_1",
        ALI_MISTAKES,
        curvature_angle_threshold=29.802918037571296,
    )

    # 3: Oblique Stitch
    vis_ali_3(
        cnt_all,
        df_ALI,
        3,
        INFER_PATH / "output_ALI_3",
        ALI_MISTAKES,
        angle_threshold=15.1065227668833,
    )

    # 4: Too Wide Bite
    vis_ali_4(
        cnt_all,
        df_ALI,
        4,
        INFER_PATH / "output_ALI_4",
        ALI_MISTAKES,
        longest_side_threshold=0.12918923070841787,
    )

    # 5: Partial Thickness
    vis_ali_5(
        cnt_all,
        df_ALI,
        5,
        INFER_PATH / "output_ALI_5",
        ALI_MISTAKES,
        aspect_ratio_threshold=2.1855058145303503,
        longest_side_threshold=0.05182086244695942,
    )

    # 6: Unequal Distancing
    vis_ali_6(
        cnt_all,
        df_ALI,
        6,
        INFER_PATH / "output_ALI_6",
        ALI_MISTAKES,
        max_thresh=0.14783247540551675,
        min_thresh=0.06957493424909032,
    )

    print("[INFO] Post-processing complete. Results saved under:", INFER_PATH)


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_postprocessing(verbose=False)
