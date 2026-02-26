import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path
import numpy as np
import sys
import argparse
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
from oncotrail.constants import target_dict_mapping

def plot_results(results_path, save_dir):
    """
    Plot AUC results from CSV files.
    
    Parameters:
    results_path (str): Path where all result CSV files are saved
    save_dir (str): Path where plots should be saved
    """

    # Plot AUC comparison
    
    # Define target category function
    def target_category(target):
        if 'grade2plus' in target:
            return 'lab'
        elif 'change' in target:
            return 'symptom'
        else:
            return 'clinic'
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all target_*_auc.csv files
    csv_pattern = os.path.join(results_path, "target_*_auc.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        logger.info(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            logger.info(f"Error reading {file}: {e}")
    
    if not dataframes:
        logger.info("No valid dataframes found")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    
    # Apply target categorization
    combined_df['target_category'] = combined_df['target'].apply(target_category)
    
    # Apply target renaming
    def rename_target(target):
        key = target.replace("_", "-")
        return target_dict_mapping.get(key, key)  # Use original if not in mapping
    
    combined_df['target_renamed'] = combined_df['target'].apply(rename_target)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get unique categories
    categories = combined_df['target_category'].unique()
    logger.info(f"Target categories found: {categories}")
    
    # Create plots for each category and metric combination
    metrics = ['train_auc', 'test_auc']
    
    for category in categories:
        category_data = combined_df[combined_df['target_category'] == category]
        
        for metric in metrics:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            plot_data = category_data.pivot_table(
                index='target_renamed', 
                columns='aggregation', 
                values=metric, 
                aggfunc='mean'
            )

            # Reorder columns to put specific methods at the right
            def sort_aggregation_methods(columns):
                # Methods to appear at the right (in this order)
                priority_methods = ['max', 'min', 'mean']
                
                # Separate columns into regular and priority
                regular_cols = []
                priority_cols = []
                autogluon_cols = []
                
                for col in columns:
                    if col in priority_methods:
                        priority_cols.append(col)
                    elif 'autogluon' in col.lower():
                        autogluon_cols.append(col)
                    else:
                        regular_cols.append(col)
                
                # Sort each group
                regular_cols.sort()
                # Sort priority cols by their order in priority_methods
                priority_cols.sort(key=lambda x: priority_methods.index(x))
                autogluon_cols.sort()
                
                # Combine: regular methods first, then priority methods, then autogluon methods
                return regular_cols + priority_cols + autogluon_cols
            
            # Reorder the columns
            ordered_columns = sort_aggregation_methods(plot_data.columns.tolist())
            plot_data = plot_data[ordered_columns]
            
            # Create the grouped bar chart
            ax = plot_data.plot(kind='bar', 
                              figsize=(12, 8), 
                              width=0.8,
                              rot=45)
            
            # Customize plot
            plt.title(f'{metric.replace("_", " ").title()} by Aggregation Method\n({category.title()} Targets)', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Target', fontsize=12, fontweight='bold')
            plt.ylabel(metric.replace("_", " ").title(), fontsize=12, fontweight='bold')
            plt.legend(title='Aggregation Method', 
                      title_fontsize=12, 
                      fontsize=10, 
                      bbox_to_anchor=(1.05, 1), 
                      loc='upper left')
            
            # Improve layout
            plt.tight_layout()
            plt.grid(axis='y', alpha=0.3)
            
            # Save plot
            filename = f"{category}_{metric}_comparison.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved plot: {filepath}")

    # Plot feature importance

    dpi = 300
    figure_size = (14,10)

    # Color scheme
    base_color = '#666666'
    target_type_colors = {
        'clinic': '#d95f02',
        'lab': '#1b9e77',
        'symptom': '#7570b3',
    }
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all target_*_feature_importance.csv files
    csv_pattern = os.path.join(results_path, "target_*_feature_importance.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        logger.info(f"No feature importance CSV files found matching pattern: {csv_pattern}")
        return
    
    logger.info(f"Found {len(csv_files)} feature importance CSV files")
    
    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Keep only the columns we care about
            if all(col in df.columns for col in ['feature', 'importance_score', 'target']):
                df_subset = df[['feature', 'importance_score', 'target']].copy()
                dataframes.append(df_subset)
            else:
                logger.info(f"Warning: Required columns not found in {file}")
        except Exception as e:
            logger.info(f"Error reading {file}: {e}")
    
    if not dataframes:
        logger.info("No valid feature importance dataframes found")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined feature importance dataframe shape: {combined_df.shape}")
    
    # Apply target categorization
    combined_df['target_category'] = combined_df['target'].apply(target_category)
    
    # Get unique features (should be: tabular, nlp, finetune, prompting)
    features = sorted(combined_df['feature'].unique())
    categories = sorted(combined_df['target_category'].unique())
    
    logger.info(f"Features found: {features}")
    logger.info(f"Target categories: {categories}")
    
    # Create the plot
    plt.figure(figsize=figure_size)
    
    # Set up x positions for features
    x_positions = np.arange(len(features))
    jitter_width = 0.15  # Width for jittering
    category_width = 0.08  # Width between category groups
    
    # Plot for each feature
    for i, feature in enumerate(features):
        feature_data = combined_df[combined_df['feature'] == feature]
        
        # Position 1: All targets (center)
        all_scores = feature_data['importance_score'].values
        all_jitter = np.random.uniform(-jitter_width/4, jitter_width/4, len(all_scores))
        x_all = np.full(len(all_scores), x_positions[i]) + all_jitter
        
        plt.scatter(x_all, all_scores, c=base_color, alpha=0.7, s=50, 
                   label='All' if i == 0 else "", zorder=2)
        
        # Positions 2-4: Each category (spread out)
        for j, category in enumerate(categories):
            category_data = feature_data[feature_data['target_category'] == category]
            if len(category_data) == 0:
                continue
                
            scores = category_data['importance_score'].values
            # Offset each category to the right
            offset = (j + 1) * category_width
            jitter = np.random.uniform(-jitter_width/8, jitter_width/8, len(scores))
            x_cat = np.full(len(scores), x_positions[i] + offset) + jitter
            
            plt.scatter(x_cat, scores, c=target_type_colors[category], alpha=0.8, s=50,
                       label=category.title() if i == 0 else "", zorder=3)
    
    # Customize plot
    plt.xlabel('Feature Type', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=14, fontweight='bold')
    plt.title('Feature Importance Distribution Across Target Categories', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions + category_width, [f.title() for f in features])
    
    # Add legend
    plt.legend(title='Target Type', title_fontsize=12, fontsize=10, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    filename = "feature_importance_distribution.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance plot: {filepath}")

    # Plot model correlations

    # Find all target_*_model_correlations.csv files
    csv_pattern = os.path.join(results_path, "target_*_model_correlations.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        logger.info(f"No model correlation CSV files found matching pattern: {csv_pattern}")
        return
    
    logger.info(f"Found {len(csv_files)} model correlation CSV files")
    
    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Keep only the columns we care about
            required_cols = ['model1', 'model2', 'correlation', 'dataset', 'target']
            if all(col in df.columns for col in required_cols):
                df_subset = df[required_cols].copy()
                dataframes.append(df_subset)
            else:
                logger.info(f"Warning: Required columns not found in {file}")
        except Exception as e:
            logger.info(f"Error reading {file}: {e}")
    
    if not dataframes:
        logger.info("No valid model correlation dataframes found")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined model correlation dataframe shape: {combined_df.shape}")
    
    # Apply target categorization
    combined_df['target_category'] = combined_df['target'].apply(target_category)
    
    # Create model pair labels (ensure consistent ordering)
    def create_model_pair_label(row):
        model1, model2 = sorted([row['model1'], row['model2']])
        return f"{model1} vs {model2}"
    
    combined_df['model_pair'] = combined_df.apply(create_model_pair_label, axis=1)
    
    # Get unique model pairs and categories
    model_pairs = sorted(combined_df['model_pair'].unique())
    categories = sorted(combined_df['target_category'].unique())
    datasets = sorted(combined_df['dataset'].unique())
    
    logger.info(f"Model pairs found: {model_pairs}")
    logger.info(f"Target categories: {categories}")
    logger.info(f"Datasets: {datasets}")
    
    # Create plots for each dataset (train/test)
    for dataset in datasets:
        dataset_data = combined_df[combined_df['dataset'] == dataset]
        
        plt.figure(figsize=figure_size)
        
        # Set up x positions for model pairs
        x_positions = np.arange(len(model_pairs))
        jitter_width = 0.2  # Width for jittering
        category_width = 0.12  # Width between category groups
        
        # Plot for each model pair
        for i, model_pair in enumerate(model_pairs):
            pair_data = dataset_data[dataset_data['model_pair'] == model_pair]
            
            if len(pair_data) == 0:
                continue
            
            # Position 1: All targets (center)
            all_correlations = pair_data['correlation'].values
            all_jitter = np.random.uniform(-jitter_width/4, jitter_width/4, len(all_correlations))
            x_all = np.full(len(all_correlations), x_positions[i]) + all_jitter
            
            plt.scatter(x_all, all_correlations, c=base_color, alpha=0.7, s=50, 
                       label='All' if i == 0 else "", zorder=2)
            
            # Positions 2-4: Each category (spread out)
            for j, category in enumerate(categories):
                category_data = pair_data[pair_data['target_category'] == category]
                if len(category_data) == 0:
                    continue
                    
                correlations = category_data['correlation'].values
                # Offset each category to the right
                offset = (j + 1) * category_width
                jitter = np.random.uniform(-jitter_width/8, jitter_width/8, len(correlations))
                x_cat = np.full(len(correlations), x_positions[i] + offset) + jitter
                
                plt.scatter(x_cat, correlations, c=target_type_colors[category], alpha=0.8, s=50,
                           label=category.title() if i == 0 else "", zorder=3)
        
        # Customize plot
        plt.xlabel('Model Pairs', fontsize=14, fontweight='bold')
        plt.ylabel('Correlation', fontsize=14, fontweight='bold')
        plt.title(f'Model Prediction Correlations - {dataset.title()} Dataset\nAcross Target Categories', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis ticks and labels (rotate for better readability)
        plt.xticks(x_positions + category_width, model_pairs, rotation=45, ha='right')
        
        # Add legend
        plt.legend(title='Target Type', title_fontsize=12, fontsize=10, 
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, zorder=1)
        
        # Add grid
        plt.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits to better show correlation range
        plt.ylim(-1.1, 1.1)
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot
        filename = f"model_correlations_{dataset}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model correlation plot: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Plot AUC results from CSV files.")
    parser.add_argument('--results_path', type=str, required=True, help="Path where all result CSV files are saved")
    parser.add_argument('--save_dir', type=str, required=True, help="Path where plots should be saved")
    
    args = parser.parse_args()
    
    plot_results(args.results_path, args.save_dir)

if __name__ == "__main__":
    main()