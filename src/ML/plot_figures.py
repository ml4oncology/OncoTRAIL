import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from llm_notes_classification.prompt.base_runner import load_shapley_df, load_logistic_df

dict_mapping = {
    'target-hemoglobin-grade2plus': 'Hb',
    'target-neutrophil-grade2plus': 'ANC',
    'target-bilirubin-grade2plus': 'Bili',
    'target-platelet-grade2plus': 'PLT',
    'target-ALT-grade2plus': 'ALT',
    'target-AST-grade2plus': 'AST',
    'target-AKI-grade2plus': 'AKI',
    'target-death-in-30d': '30d death',
    'target-death-in-365d': '1y death',
    'target-ED-visit': 'ED visits',
    'target-esas-depression-3pt-change': 'Depression',
    'target-esas-pain-3pt-change': 'Pain',
    'target-esas-anxiety-3pt-change': 'Anxiety',
    'target-esas-tiredness-3pt-change': 'Tired',
    'target-esas-nausea-3pt-change': 'Nausea',
    'target-esas-drowsiness-3pt-change': 'Drowsy',
    'target-esas-appetite-3pt-change': 'Appetite',
    'target-esas-well-being-3pt-change': 'Well-being',
    'target-esas-shortness-of-breath-3pt-change': 'Dsypnea',
}

def plot_coef_heatmap_with_labels(
    data_dict, save_dir, col_name, file_name, str_title, auc_df,
    top_n=10, cmap="Blues", max_word_len=15, base_font_size=9
):

    col_name_auc = [f for f in auc_df.columns if "auc_test" in f][0]
    auc_dict = auc_df.set_index("target")[col_name_auc].to_dict()

    targets = list(data_dict.keys())
    num_targets = len(targets)
    num_features = top_n

    feature_texts = np.empty((num_features, num_targets), dtype=object)
    coef_magnitudes = np.zeros((num_features, num_targets))

    for j, target in enumerate(targets):
        df = data_dict[target].copy()
        df["abs_coef"] = df[col_name].abs()
        df = df.nlargest(top_n, "abs_coef").reset_index(drop=True)

        print(f"Target: {target}")
        print(df.head(top_n))

        for i in range(len(df)):
            feature_texts[i, j] = df.loc[i, "var_names"]
            coef_magnitudes[i, j] = df.loc[i, "abs_coef"]

    # Normalize per column
    normed_mags = np.zeros_like(coef_magnitudes)
    for j in range(num_targets):
        col = coef_magnitudes[:, j]
        max_val = col.max() if col.max() != 0 else 1.0
        normed_mags[:, j] = col / max_val

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(1.4 * num_targets, 0.7 * num_features))
    cmap = plt.get_cmap(cmap)
    im = ax.imshow(normed_mags, cmap=cmap)

    # Annotate cells with dynamic font size
    max_font = base_font_size
    min_font = 4
    for i in range(num_features):
        for j in range(num_targets):
            text = feature_texts[i, j]
            if text is None:
                text = ""

            # Wrap bigrams: replace space with newline
            if " " in text:
                text = text.replace(" ", "\n")

            # Optionally truncate very long words
            if len(text.replace("\n", "")) > 18:
                text = text[:15] + "..."

            # Dynamically reduce font size for long text
            plain_text_len = len(text.replace("\n", ""))
            font_size = max(min_font, max_font - 0.5 * max(plain_text_len - 5, 0))

            # Draw text
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=font_size,
                color="black" if normed_mags[i, j] < 0.6 else "white",
                linespacing=0.9
            )


    # Clean formatting
    ax.minorticks_off()
    ax.grid(False)
    ax.tick_params(which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(num_targets))
    ax.set_xticklabels([f"{dict_mapping[target]} (AUC: {auc_dict[target]:.3f})" for target in targets], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(num_features))
    ax.set_yticklabels([f"Top {i+1}" for i in range(num_features)], fontsize=9)

    plt.title(str_title, fontsize=12)
    plt.tight_layout()

    # Save
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ngram(max_ngram, shapley_path, log_reg_path, shapley, save_dir):
    df_shapley_all_results = pd.read_csv(shapley_path)
    df_logistic_all_results = pd.read_csv(log_reg_path)

    if shapley == 1:
        col_name = 'shap_values'
        # col_name = 'corr_coeff'
        df_results_merged = df_shapley_all_results
    else:
        col_name = 'lr_values'
        df_results_merged = df_logistic_all_results
    
    # get the column which contains string "auc_test"
    col_name_auc = [col for col in df_results_merged.columns if "auc_test" in col][0]
    df_auc = df_results_merged[['target', col_name_auc]]

    all_results = {}
    # iterate over each row of df_results_merged
    for _, row in df_results_merged.iterrows():
        prediction_path = row['pred_file_name']
        target_name = row['target']
        if shapley == 1:
            df_to_plot = load_shapley_df(prediction_path)
        else:
            # prediction_path = df_shapley_all_results[df_shapley_all_results['target'] == target_name.replace('_', '-')]['pred_file_name'].values[0]
            # doesn't matter because we just need to get the column names
            # it does matter. you should use LR model, not other ML model because under LR, the train set also includes the evaluation set
            # which may affect 
            log_reg_path = row['model_file_name']
            df_to_plot = load_logistic_df(prediction_path, log_reg_path)

        # only take max_ngram rows of df_to_plot
        df_to_plot = df_to_plot.head(max_ngram)
        df_to_plot = df_to_plot[['var_names', col_name]]
        all_results[target_name] = df_to_plot

    str_title = f"Top Features by Target (Color = |{'LR' if shapley == 0 else 'Shapley Corr'} Coeff|)"
    plot_coef_heatmap_with_labels(all_results, save_dir, col_name, 'shapley' if shapley == 1 else 'logistic', str_title, df_auc, top_n=max_ngram, cmap="Blues")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot n-gram analysis with optional Shapley values')
    parser.add_argument('max_ngram', type=int, help='Maximum n-gram size to analyze')
    parser.add_argument('shapley_path', type=str, help='Path to Shapley values file')
    parser.add_argument('log_reg_path', type=str, help='Path to logistic regression file')
    parser.add_argument('shapley', type=int, choices=[0, 1], help='Whether to use Shapley values (0 for False, 1 for True)')
    parser.add_argument('save_dir', type=str, help='Directory to save the plot output')
    args = parser.parse_args()

    # Call the function with parsed arguments
    plot_ngram(args.max_ngram, args.shapley_path, args.log_reg_path, 
            args.shapley, args.save_dir)