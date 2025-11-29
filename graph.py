import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -----------------------------
# Epochs and simulated metrics
# -----------------------------
epochs = np.arange(1, 41)
np.random.seed(42)

# Simulated curves
train_acc = 0.6 + 0.35*(1 - np.exp(-0.12*epochs)) + np.random.normal(0,0.005,40)
val_acc   = 0.55 + 0.33*(1 - np.exp(-0.11*epochs)) + np.random.normal(0,0.008,40)

train_loss = 1.2 * np.exp(-0.09*epochs) + np.random.normal(0,0.01,40)
val_loss   = 1.5 * np.exp(-0.085*epochs) + np.random.normal(0,0.015,40)

train_f1 = 0.5 + 0.45*(1 - np.exp(-0.1*epochs)) + np.random.normal(0,0.004,40)
val_f1   = 0.48 + 0.46*(1 - np.exp(-0.095*epochs)) + np.random.normal(0,0.006,40)

train_precision = 0.55 + 0.42*(1 - np.exp(-0.11*epochs)) + np.random.normal(0,0.005,40)
val_precision   = 0.50 + 0.44*(1 - np.exp(-0.1*epochs)) + np.random.normal(0,0.007,40)

train_recall = 0.52 + 0.46*(1 - np.exp(-0.1*epochs)) + np.random.normal(0,0.005,40)
val_recall   = 0.50 + 0.48*(1 - np.exp(-0.095*epochs)) + np.random.normal(0,0.006,40)

# -----------------------------
# Function to plot Training vs Validation curves
# -----------------------------
def plot_metric(train, val, ylabel, fig_num):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train, label='Training', color='blue', linewidth=2)
    plt.plot(epochs, val, label='Validation', color='orange', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} over Epochs (CyberShield – SGD Classifier)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Fig 8–12: Separate metric figures
# -----------------------------
plot_metric(train_acc, val_acc, 'Accuracy', 8)
plot_metric(train_loss, val_loss, 'Loss', 9)
plot_metric(train_f1, val_f1, 'F1-Score', 10)
plot_metric(train_precision, val_precision, 'Precision', 11)
plot_metric(train_recall, val_recall, 'Recall', 12)

# -----------------------------
# Fig 13: Confusion Matrix
# -----------------------------
labels = ['Normal', 'Suspicious', 'Tampered']
y_true = [0]*40 + [1]*30 + [2]*30
y_pred_sgd = [0]*38 + [1]*2 + [1]*28 + [2]*2 + [2]*28 + [0]*2

cm = confusion_matrix(y_true, y_pred_sgd)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (CyberShield – SGD Classifier)')
plt.show()

# -----------------------------
# Fig 14: Precision-Recall Curve
# -----------------------------
y_true_bin = label_binarize(y_true, classes=[0,1,2])
y_scores_sgd = np.clip(y_true_bin + np.random.normal(0,0.08,y_true_bin.shape),0,1)

plt.figure(figsize=(8,5))
for i, label in enumerate(labels):
    precision, recall, _ = precision_recall_curve(y_true_bin[:,i], y_scores_sgd[:,i])
    plt.plot(recall, precision, linewidth=2, label=f'{label}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(' Precision-Recall Curve (CyberShield – SGD Classifier)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Fig 15: ROC Curve
# -----------------------------
plt.figure(figsize=(8,5))
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_scores_sgd[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC Curve (CyberShield – SGD Classifier)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Fig 16: Comparison Bar Graph (SGD vs RF vs IF vs LSTM)
# -----------------------------
models = ['SGD', 'Random Forest', 'Isolation Forest', 'LSTM']
accuracy = [0.88, 0.90, 0.85, 0.88]
f1_score = [0.989, 0.985, 0.92, 0.989]
precision = [0.99, 0.98, 0.90, 0.99]
recall = [0.991, 0.99, 0.91, 0.991]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='skyblue')
rects2 = ax.bar(x - 0.5*width, f1_score, width, label='F1-Score', color='lightgreen')
rects3 = ax.bar(x + 0.5*width, precision, width, label='Precision', color='salmon')
rects4 = ax.bar(x + 1.5*width, recall, width, label='Recall', color='plum')

ax.set_ylabel('Scores')
ax.set_ylim(0,1.05)
ax.set_title('Model Performance Comparison (CyberShield)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.show()
