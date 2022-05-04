import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from scipy import stats

import collections

import forestci as fci

import json

import warnings
warnings.filterwarnings('ignore')

dataset = 'CIFAR10'

x_train = y_train = x_test = y_test = None

if dataset == 'MNIST':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
elif dataset == 'CIFAR10':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

NUMBER_OF_CLASSES = len(np.unique(y_train))

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))
print("Number of classes:", NUMBER_OF_CLASSES)


# Baseline model

# the batch size will be used as the number of new images to annotate
BATCH_SIZE = 128
BASELINE_EPOCHS = 6
VALIDATION_SPLIT = 0.2

baseline_model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(NUMBER_OF_CLASSES)
])

baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

baseline_model.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=BASELINE_EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=0,
)

_, baseline_metrics = baseline_model.evaluate(
    x=x_test,
    y=y_test,
    verbose=0,
)

print("SparseCategoricalAccuracy", baseline_metrics)


# 90% baseline
P = 0.9
target_accuracy = baseline_metrics * P
print('Target accuracy', target_accuracy, '(', P, '%)')

# Active POC

def norm_ratios(
    labels,
    sum_up_to,
    classes,
    inclusion=True):
    """ Given an array of classes (labels), returns an array of
    elements per class (ratio) that sum up to a given number.
    In other words, the sum of all returned elements is equal to
    'sum_up_to'.
    """
    labels_counts = collections.Counter(labels)    

    # select a repersentative initial set of images to be annotated
    counts = list()
    for i in classes:
        counts.append(labels_counts[i])

    counts = np.array(counts)
    _min = counts.min() if counts.min() > 0 else 1
    _sum = counts.sum() if counts.sum() > 0 else 1
    
    counts = counts / _min
    counts = counts / _sum
    counts = counts * sum_up_to
    counts = counts.astype(int)

    if inclusion:
        # to avoid smalls clusters to be left out, we add at least one element per cluster
        counts[counts == 0] = 1

    while counts.sum() != sum_up_to:
        if counts.sum() > sum_up_to:
            counts[counts.argmax()] = counts[counts.argmax()] - 1
        elif counts.sum() < sum_up_to:
            counts[counts.argmin()] = counts[counts.argmin()] + 1
            
    return counts


def auto_cluster_size(
    x_train_flat,
    max_n_clusters):
    """ Set automatically the optimal number of clusters
    """
    print("Warning: The automatic setting of the number of clusters is not yet implemented")
    return max_n_clusters


def active_cluster(
    x_train,
    y_train,
    p_batch_size,
    p_epochs,
    p_init,
    max_n_clusters,
    target_accuracy,
    custom_acquisition_fn=None):
    """
    """
    x_train_flat = np.reshape(
        x_train, 
        (len(x_train), x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    )
    
    n_clusters = auto_cluster_size(x_train_flat, max_n_clusters)
    p_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x_train_flat)
    
    # we will use the number of elements per cluster to select the initial batch
    counts = norm_ratios(p_kmeans.labels_, p_init, range(n_clusters))

    set_ix = list()
    for i in range(len(counts)):
        pos = np.where(p_kmeans.labels_ == i)[0]
        set_ix = set_ix + np.random.choice(pos, size=counts[i], replace=False).tolist()

    annotated_ix = set_ix.copy()

    x_train_annotated = x_train[annotated_ix]    
    y_train_annotated = y_train[annotated_ix]

    p_model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(NUMBER_OF_CLASSES)
    ])
    
    p_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    p_model.fit(
        x=x_train_annotated,
        y=y_train_annotated,
        batch_size=p_batch_size,
        epochs=p_epochs,
        validation_split=VALIDATION_SPLIT,
        verbose=0,
    )

    p_spent = p_epochs * (1 - VALIDATION_SPLIT) * len(x_train_annotated)

    _, metrics = p_model.evaluate(
        x=x_test,
        y=y_test,
        verbose=0,
    )

    p_metrics = [metrics]
    p_budget = [p_spent]
    p_annotated = [len(x_train_annotated)]

    while p_metrics[-1] < target_accuracy:
        
        # TODO: try alternative balancing methods, e.g.,
        # weak label distribution, match between weak and real label, ...
        preds = p_model.predict(x_train_annotated)
        preds_class = np.argmax(preds, axis=1)
        misclassified_relative = np.where((preds_class - y_train_annotated)!=0)
        misclassified_pos = np.take(annotated_ix, misclassified_relative)
        misclassified_cluster_no = np.take(p_kmeans.labels_, misclassified_pos)

        # compute ratio of misclassified classes
        #counts_active = norm_ratios(
        #    misclassified_cluster_no.ravel(), 
        #    p_batch_size,
        #    range(n_clusters),
        #    inclusion=False)        
        # TODO do this switch using a function
        counts_active = norm_ratios(p_kmeans.labels_, p_batch_size, range(n_clusters))

        # select new images to annotate with the given ratios
        ix_pool = np.delete(range(len(x_train)), annotated_ix)
        weak_labels_ix_pool = np.delete(p_kmeans.labels_, annotated_ix)

        set_ix = list()
        for i in range(len(counts_active)):
            pos = np.where(weak_labels_ix_pool == i)[0]                    
            if len(pos) >= counts_active[i]:
                if custom_acquisition_fn:
                    # results are relative to input positions
                    tmp_ix = custom_acquisition_fn(
                        x_train_flat[annotated_ix], #TODO: filter by weakclass
                        y_train_annotated,
                        x_train_flat[ix_pool[pos]],
                        counts_active[i])
                    # thus, we select from the original indices
                    if len(tmp_ix) > 0:
                        set_ix = set_ix + ix_pool[pos][tmp_ix].tolist()
                else:
                    set_ix = set_ix + np.random.choice(
                        ix_pool[pos], 
                        size=counts_active[i], 
                        replace=False).tolist()
            elif len(pos) > 0:
                set_ix = set_ix + ix_pool[pos].tolist()    

        annotated_ix = annotated_ix + set_ix

        x_train_annotated = x_train[annotated_ix]    
        y_train_annotated = y_train[annotated_ix]

        p_model.fit(
            x=x_train_annotated,
            y=y_train_annotated,
            batch_size=p_batch_size,
            epochs=p_epochs,
            validation_split=VALIDATION_SPLIT,
            verbose=0,
        )

        _, metrics = p_model.evaluate(
            x=x_test,
            y=y_test,
            verbose=0,
        )

        p_metrics.append(metrics)
        p_spent = p_spent + p_epochs * (1 - VALIDATION_SPLIT) * len(x_train_annotated)    
        p_budget.append(p_spent)
        p_annotated.append(len(x_train_annotated))

        if len(p_metrics) % 20 == 0:
            print(p_metrics[-1], p_annotated[-1])
            
    return p_metrics, p_budget, p_annotated



r0 = active_cluster(
    x_train,
    y_train,
    p_batch_size=5,
    p_epochs=6,
    p_init=20,
    max_n_clusters=10,
    target_accuracy=target_accuracy,)

r0_12 = active_cluster(
    x_train,
    y_train,
    p_batch_size=5,
    p_epochs=6,
    p_init=20,
    max_n_clusters=12,
    target_accuracy=target_accuracy,)

r0_14 = active_cluster(
    x_train,
    y_train,
    p_batch_size=5,
    p_epochs=6,
    p_init=20,
    max_n_clusters=14,
    target_accuracy=target_accuracy,)

r0_16 = active_cluster(
    x_train,
    y_train,
    p_batch_size=5,
    p_epochs=6,
    p_init=20,
    max_n_clusters=16,
    target_accuracy=target_accuracy,)

r0_18 = active_cluster(
    x_train,
    y_train,
    p_batch_size=5,
    p_epochs=6,
    p_init=20,
    max_n_clusters=12,
    target_accuracy=target_accuracy,)

r0_20 = active_cluster(
    x_train,
    y_train,
    p_batch_size=5,
    p_epochs=6,
    p_init=20,
    max_n_clusters=12,
    target_accuracy=target_accuracy,)

results = dict()
results['r0'] = r0
results['r0_12'] = r0_12
results['r0_14'] = r0_14
results['r0_16'] = r0_16
results['r0_18'] = r0_18
results['r0_20'] = r0_20

active_file = open("poc.json", "w")
json.dump(results, active_file)
active_file.close()

plt.plot(r0[2], r0[0], color='red')
plt.plot(r0_12[2], r0_12[0], color='orange')
plt.plot(r0_14[2], r0_14[0], color='yellow')
plt.plot(r0_16[2], r0_16[0], color='green')
plt.plot(r0_18[2], r0_18[0], color='cyan')
plt.plot(r0_20[2], r0_20[0], color='blue')

plt.axhline(target_accuracy, color='black')
plt.axhline(baseline_metrics, color='grey')

plt.savefig('poc.png')

