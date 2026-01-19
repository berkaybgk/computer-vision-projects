import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score, confusion_matrix, balanced_accuracy_score,
    accuracy_score, classification_report
)
from sklearn.metrics.pairwise import chi2_kernel
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_image_paths_with_labels(file_path):
    """Load image paths and extract labels from folder names"""
    with open(file_path, 'r') as f:
        paths = [line.strip() for line in f.readlines()]

    # Extract labels from paths (folder name is the class)
    labels = [path.split('/')[0] for path in paths]

    return paths, labels

def get_histogram_filename(image_path, k):
    """Convert image path to histogram filename"""
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    return f'k{k}_{filename_without_ext}.npy'

def load_histograms(image_paths, histogram_folder, k):
    """Load histograms for all images"""
    print(f"Loading histograms from {histogram_folder} for k={k}...")

    histograms = []
    valid_paths = []

    for i, img_path in enumerate(image_paths):
        hist_file = get_histogram_filename(img_path, k)
        hist_path = os.path.join(histogram_folder, hist_file)

        if os.path.exists(hist_path):
            hist = np.load(hist_path)
            histograms.append(hist)
            valid_paths.append(img_path)
        else:
            print(f"Warning: {hist_path} not found!")

        if (i + 1) % 500 == 0:
            print(f"Loaded {i + 1}/{len(image_paths)} histograms")

    histograms = np.array(histograms)
    print(f"Loaded {len(histograms)} histograms with shape {histograms.shape}")

    return histograms, valid_paths

# Removed compute_chi2_kernel_matrix - using approximation instead

def train_svm_linear(X_train, y_train):
    """Train linear SVM with grid search"""
    print("\nTraining Linear SVM with GridSearchCV...")

    # Parameter grid for C values
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    svm = LinearSVC(random_state=42, max_iter=2000)

    # 5-fold cross validation
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring='f1_macro',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

def train_svm_chi2(X_train, y_train, X_test):
    """Train SVM with Chi-Squared kernel approximation (much faster)"""
    print("\nTraining SVM with Additive Chi2 Kernel Approximation...")

    from sklearn.kernel_approximation import AdditiveChi2Sampler

    # Transform features to approximate chi2 kernel
    print("Transforming features with Additive Chi2 Sampler...")
    chi2_sampler = AdditiveChi2Sampler(sample_steps=2)
    X_train_chi2 = chi2_sampler.fit_transform(X_train)
    X_test_chi2 = chi2_sampler.transform(X_test)

    print(f"Transformed feature shape: {X_train_chi2.shape}")

    # Parameter grid for C values
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    svm = LinearSVC(random_state=42, max_iter=2000)

    # 5-fold cross validation
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring='f1_macro',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train_chi2, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, X_test_chi2

def evaluate_classifier(y_true, y_pred, class_names):
    """Calculate all evaluation metrics"""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    # Mean F1-Score (macro average)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"\nMean F1-Score (macro): {f1_macro:.4f}")

    # Per-class F1-Score
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=class_names)
    print("\nPer-Class F1-Scores:")
    for class_name, f1 in zip(class_names, f1_per_class):
        print(f"  {class_name:25s}: {f1:.4f}")

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"\nMean Balanced Accuracy: {balanced_acc:.4f}")

    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Mean Accuracy (imbalanced): {overall_acc:.4f}")

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for class_name, acc in zip(class_names, per_class_acc):
        print(f"  {class_name:25s}: {acc:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'balanced_accuracy': balanced_acc,
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def find_misclassified_images(y_true, y_pred, image_paths, max_images=20):
    """Find misclassified test images"""
    misclassified_indices = []

    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            misclassified_indices.append(i)

    print(f"\nTotal misclassified images: {len(misclassified_indices)}")

    # Select random subset if too many
    if len(misclassified_indices) > max_images:
        misclassified_indices = np.random.choice(
            misclassified_indices, max_images, replace=False
        )

    misclassified_paths = [image_paths[i] for i in misclassified_indices]
    misclassified_true = [y_true[i] for i in misclassified_indices]
    misclassified_pred = [y_pred[i] for i in misclassified_indices]

    return misclassified_paths, misclassified_true, misclassified_pred

def create_misclassified_thumbnail_grid(image_paths, true_labels, pred_labels,
                                        image_folder, save_path, grid_size=(4, 5)):
    """Create grid of misclassified images as thumbnails"""
    n_images = len(image_paths)
    rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(rows * cols):
        if i < n_images:
            img_path = os.path.join(image_folder, image_paths[i])

            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                axes[i].imshow(img)
                axes[i].set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}",
                                  fontsize=8)
            except Exception as e:
                axes[i].text(0.5, 0.5, 'Image\nNot Found',
                             ha='center', va='center')
                print(f"Could not load {img_path}: {e}")

        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassified thumbnails saved to {save_path}")

def find_best_worst_classes(f1_per_class, class_names, n=5):
    """Find best and worst performing classes"""
    sorted_indices = np.argsort(f1_per_class)

    worst_indices = sorted_indices[:n]
    best_indices = sorted_indices[-n:][::-1]

    print("\n" + "="*60)
    print(f"TOP {n} WORST PERFORMING CLASSES:")
    print("="*60)
    for idx in worst_indices:
        print(f"  {class_names[idx]:25s}: F1 = {f1_per_class[idx]:.4f}")

    print("\n" + "="*60)
    print(f"TOP {n} BEST PERFORMING CLASSES:")
    print("="*60)
    for idx in best_indices:
        print(f"  {class_names[idx]:25s}: F1 = {f1_per_class[idx]:.4f}")

def run_experiment(descriptor_type, k_value, use_chi2=False):
    """Run one complete experiment"""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {descriptor_type.upper()} with K={k_value}, Kernel={'Chi2' if use_chi2 else 'Linear'}")
    print("="*80)

    # Load data
    train_paths, train_labels = load_image_paths_with_labels('../data/TrainImages.txt')
    test_paths, test_labels = load_image_paths_with_labels('../data/TestImages.txt')

    histogram_folder = f'../data/histograms/{descriptor_type}'

    # Load histograms
    X_train, train_paths = load_histograms(train_paths, histogram_folder, k_value)
    X_test, test_paths = load_histograms(test_paths, histogram_folder, k_value)

    # Get labels for valid paths
    y_train = [path.split('/')[0] for path in train_paths]
    y_test = [path.split('/')[0] for path in test_paths]

    # Get unique class names
    class_names = sorted(list(set(y_train)))
    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Train classifier
    if use_chi2:
        model, best_params, X_test_transformed = train_svm_chi2(X_train, y_train, X_test)
        y_pred = model.predict(X_test_transformed)
    else:
        model, best_params = train_svm_linear(X_train, y_train)
        y_pred = model.predict(X_test)

    # Evaluate
    metrics = evaluate_classifier(y_test, y_pred, class_names)

    # Find best and worst classes
    find_best_worst_classes(metrics['f1_per_class'], class_names)

    # Plot confusion matrix
    kernel_name = 'chi2' if use_chi2 else 'linear'
    cm_path = f'../results/{descriptor_type}_k{k_value}_{kernel_name}_confusion_matrix.png'
    os.makedirs('../results', exist_ok=True)
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        f'{descriptor_type.upper()} K={k_value} ({kernel_name} kernel)',
        cm_path
    )

    # Create misclassified thumbnails
    mis_paths, mis_true, mis_pred = find_misclassified_images(y_test, y_pred, test_paths)
    if len(mis_paths) > 0:
        thumb_path = f'../results/{descriptor_type}_k{k_value}_{kernel_name}_misclassified.png'
        create_misclassified_thumbnail_grid(
            mis_paths, mis_true, mis_pred,
            '../data/Images',
            thumb_path
        )

    return metrics, best_params

def main():
    print("="*80)
    print("IMAGE CLASSIFICATION - BAG OF VISUAL WORDS")
    print("="*80)

    # Experiments to run
    descriptor_types = ['sift', 'swin']
    k_values = [50, 100, 500]
    kernels = [False, True]  # False = Linear, True = Chi2

    results = {}

    # Run all experiments
    for desc_type in descriptor_types:
        for k in k_values:
            for use_chi2 in kernels:
                kernel_name = 'chi2' if use_chi2 else 'linear'
                exp_name = f"{desc_type}_k{k}_{kernel_name}"

                try:
                    metrics, params = run_experiment(desc_type, k, use_chi2)
                    results[exp_name] = {
                        'metrics': metrics,
                        'params': params
                    }
                except Exception as e:
                    print(f"\nError in experiment {exp_name}: {e}")
                    continue

    # Summary of all results
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)

    for exp_name, result in results.items():
        metrics = result['metrics']
        params = result['params']
        print(f"\n{exp_name}:")
        print(f"  Best C: {params['C']}")
        print(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")

    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETED!")
    print("="*80)
    print("\nResults saved in 'results/' folder")
    print("- Confusion matrices (.png)")
    print("- Misclassified image thumbnails (.png)")

if __name__ == "__main__":
    main()
