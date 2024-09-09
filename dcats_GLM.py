import numpy as np
import pandas as pd

def validate_all_cell_groups(ad_mat, dp_mat, clone_mat, min_cells_group):
    # Check if the column names (cells) are identical between AD_mat and DP_mat
    if not np.array_equal(ad_mat.columns, dp_mat.columns):
        raise ValueError("Cells in AD and DP matrices are not identical")
    
    # Check if the row names (variants) are identical between AD_mat and DP_mat
    if not np.array_equal(ad_mat.index, dp_mat.index):
        raise ValueError("Variants in AD and DP matrices are not identical")
    
    mis_label = []
    unique_labels = clone_mat['cell_label'].unique()
    
    for label in unique_labels:
        cells = clone_mat.loc[clone_mat['cell_label'] == label, 'cellID']
        if len(np.intersect1d(cells, ad_mat.columns)) < min_cells_group:
            mis_label.append(label)
    
    if mis_label:
        print(f"The following label(s) have less than the required cells, they will be excluded from further testing: {', '.join(mis_label)}")
    
    return [label for label in unique_labels if label not in mis_label]

# # Sample data
# ad_mat = pd.DataFrame({
#     'Cell1': [10, 20, 30],
#     'Cell2': [15, 25, 35],
#     'Cell3': [10, 20, 30],
#     'Cell4': [15, 25, 35]
# }, index=['Variant1', 'Variant2', 'Variant3'])

# dp_mat = pd.DataFrame({
#     'Cell1': [100, 200, 300],
#     'Cell2': [150, 250, 350],
#     'Cell3': [100, 200, 300],
#     'Cell4': [150, 250, 350]
# }, index=['Variant1', 'Variant2', 'Variant3'])

# clone_mat = pd.DataFrame({
#     'cellID': ['Cell1', 'Cell2', 'Cell3', 'Cell4'],
#     'cell_label': ['Label1', 'Label1', 'Label2', 'Label3']
# })

# min_cells_group = 2

# # Validate cell groups
# valid_labels = validate_all_cell_groups(ad_mat, dp_mat, clone_mat, min_cells_group)
# print("Valid labels:", valid_labels)


