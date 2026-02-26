import os
import glob
import pandas as pd
from oncotrail.postproc.word_analyzer_utils import process_data_for_volcano_plot
import logging
import sys
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_summary_csv(
    base_dir,
    target,
    note_type,
    save_dir,
    path_to_anchored_notes=None
):

    out_csv = os.path.join(
        save_dir,
        f"word_stats_all_{note_type}_{target.replace('_','-')}_before_pval_adjustment.csv"
    )

    # if os.path.exists(out_csv):
    #     logger.info(f"Skipping {target}: output already exists.")
    #     return

    if note_type == 'Reason':
        df_anchored_notes = []
    else:
        df_anchored_notes = pd.read_csv(path_to_anchored_notes)

    logger.info(f"Processing {target}")
    target_dir = os.path.join(base_dir, target)
    if not os.path.isdir(target_dir):
        logger.info(f"Skipping {target}: directory not found.")
        return

    subdirs = [
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d))
        and ("note_" in d or "note_tabular_" in d)
    ]

    if len(subdirs) != 1:
        logger.info(
            f"Skipping {target}: expected exactly 1 note directory, "
            f"found {len(subdirs)} ({subdirs})"
        )
        return

    note_dir = os.path.join(target_dir, subdirs[0])

    csv_files = glob.glob(os.path.join(note_dir, "summary_*.csv"))
    if len(csv_files) != 1:
        logger.info(
            f"Skipping {target}: expected exactly 1 summary_*.csv, "
            f"found {len(csv_files)}"
        )
        return

    fname = os.path.basename(csv_files[0])
    df_path = note_dir

    df_bow, _ = process_data_for_volcano_plot(
        df_path,
        fname,
        target,
        note_type,
        1,
        'average',
        'predictions',
        df_anchored_notes
    )

    # df_bow = df_bow.loc[df_bow['p_adj'] < 0.05 / n_targets].copy()
    df_bow.to_csv(out_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("note_type", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument(
        "--path_to_anchored_notes",
        type=str,
        default=None,
        help="Required if note_type is not 'Reason'"
    )

    args = parser.parse_args()

    find_summary_csv(
        args.base_dir,
        args.target,
        args.note_type,
        args.save_dir,
        args.path_to_anchored_notes
    )
