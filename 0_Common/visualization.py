import string
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.colors import Normalize
from matplotlib.pyplot import Figure
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def __plot_confusion_matrix(
    confusion_matrix, display_labels, percent=False, cmap="Blues"
):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.plot
    fig, ax = plt.subplots()
    cm = 100 * confusion_matrix.T if percent else confusion_matrix.T
    n_classes = cm.shape[0]

    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_ = ax.imshow(cm, **default_im_kw)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)
    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_cm = (format(cm[i, j], ".0f") + "%") if percent else format(cm[i, j], "d")
        default_text_kwargs = dict(ha="center", va="center", color=color, fontsize=16)
        text_[i, j] = ax.text(j, i, text_cm, **default_text_kwargs)

    if display_labels is None:
        display_labels = np.arange(n_classes)
    else:
        display_labels = display_labels

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        xlabel="True label",
        ylabel="Predicted label",
    )
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")
    plt.tight_layout()
    return fig, ax


def plot_confusion_matrix(
    endpoints: pd.Series,
    predictions: pd.Series,
    outcome: str,
    save_dir: Path,
    filename: Optional[str] = None,
    save: bool = True,
):
    classes = [outcome, f"No {outcome}"]
    ## Plot absolute counts confusion matrix
    # cm = confusion_matrix(endpoints, predictions, labels=[1, 0])
    # fig, _ = __plot_confusion_matrix(
    #     cm, display_labels=classes, percent=False, cmap="Blues"
    # )
    # if save:
    #     filename_ = f"{filename}_abs" if filename != None else "confusion_matrix_abs"
    #     save_figures(fig, save_dir, filename_)

    cm = confusion_matrix(endpoints, predictions, labels=[1, 0], normalize="true")
    fig, _ = __plot_confusion_matrix(
        cm, display_labels=classes, percent=True, cmap="Blues"
    )
    if save:
        filename_ = (
            f"{filename}_by_true"
            if filename != None
            else "confusion_matrix_percent_by_true"
        )
        save_figures(fig, save_dir, filename_)


def plot_coxph_coefficients(
    model,
    save_dir: Path,
    save: bool = True,
):
    fig, ax = plt.subplots()
    model.plot(ax=ax)
    plt.tight_layout()

    if save:
        filename = "coxph_coefficients"
        save_figures(fig, save_dir, filename)

    ## Save a version of the plot with letters instead for feature names, which are usually long
    # feature_names = ""
    # y_ticklabels = []
    # for letter, feature_name in zip(string.ascii_lowercase, ax.get_yticklabels()):
    #     feature_names += f"{letter}, {feature_name.get_text()}\n"
    #     y_ticklabels.append(f"({letter})")
    # ax.set_yticklabels(y_ticklabels)

    # if save:
    #     filename = "coxph_coefficients_letters"
    #     save_figures(fig, save_dir, filename)
    #     with open(save_dir / "coxph_coefficients_letters.txt", "w") as f:
    #         f.write(feature_names.strip())


def plot_lr_coefficients(
    model: LogisticRegression,
    save_dir: Path,
    save: bool = True,
):
    coefs = model.coef_[0]

    # 2. Sort indices by coefficient value descending
    sorted_idx_desc = np.argsort(coefs)[
        ::-1
    ]  # indices of features sorted by coef descending

    # 3. Prepare labels and values
    feature_names = model.feature_names_in_
    sorted_features = [feature_names[i] for i in sorted_idx_desc]
    sorted_coefs = coefs[sorted_idx_desc]

    # 4. Set up colormap: normalize around zero so that negative vs positive use different ends of a diverging cmap
    cmap = cm.get_cmap("RdBu_r")
    # Normalize such that 0 maps to the center of the colormap
    max_abs = np.max(np.abs(sorted_coefs)) if sorted_coefs.size > 0 else 1.0
    norm = Normalize(vmin=-max_abs, vmax=+max_abs)
    bar_colors = cmap(norm(sorted_coefs))

    # 5. Plot with Matplotlib
    fig, ax = plt.subplots()
    y_positions = np.arange(len(sorted_features))

    # barh: note that if coefficient is negative, bar extends to the left
    bars = ax.barh(
        y_positions, sorted_coefs, color=bar_colors, edgecolor="black", alpha=0.8
    )

    # 6. Annotate coefficient values at end of bars
    # Compute an offset for text: a small fraction of the x-range
    # Here we use 0.01 * range, but ensure positive offset in either direction
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    # offset in data coordinates:
    offset = x_range * 0.01
    ax.set_xlim(x_min - 0.5, x_max + 0.5)  # extend limits for text

    for i, (coef_val, bar) in enumerate(zip(sorted_coefs, bars)):
        y = bar.get_y() + bar.get_height() / 2
        if coef_val >= 0:
            x_text = coef_val + offset
            ha = "left"
        else:
            x_text = coef_val - offset
            ha = "right"
        ax.text(x_text, y, f"{coef_val:.2f}", va="center", ha=ha, fontsize=9)

    # 7. Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_features)

    ax.axvline(0, color="gray", linewidth=1, linestyle="--")  # vertical line at 0
    ax.set_xlabel("Coefficient (log-odds scale)")
    ax.invert_yaxis()  # optional: so that the top positive appears at top. Remove if you prefer bottom-up.

    plt.tight_layout()
    if save:
        filename = "logistic_regression_coefficients"
        save_figures(fig, save_dir, filename)

    ## Save a version of the plot with letters instead for feature names, which are usually long
    # feature_names = ""
    # y_ticklabels = []
    # for letter, feature_name in zip(string.ascii_lowercase, ax.get_yticklabels()):
    #     feature_names += f"{letter}, {feature_name.get_text()}\n"
    #     y_ticklabels.append(f"({letter})")
    # ax.set_yticklabels(y_ticklabels)

    # if save:
    #     filename = "logistic_regression_coefficients_letters"
    #     save_figures(fig, save_dir, filename)
    #     with open(save_dir / "logistic_regression_coefficients_letters.txt", "w") as f:
    #         f.write(feature_names.strip())


def plot_survival_curves(
    model,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    duration_column: str,
    event_column: str,
    save_dir: Path,
    save: bool = True,
):
    fig, ax = plt.subplots()

    kmf_true = KaplanMeierFitter()
    kmf_true.fit(
        endpoints.loc[:, duration_column],
        event_observed=endpoints.loc[:, event_column],
        label="True",
    )
    ax = kmf_true.plot_survival_function(ax=ax)

    x_max = endpoints[duration_column].max()
    # If there are more than 24 months, use 6 months step for x-axis ticks, if not, use 3 months step
    step = 3 if (x_max // 30 + 1 <= 24) else 6

    data = features.assign(
        **{
            duration_column: endpoints[duration_column],
            event_column: endpoints[event_column],
        }
    )
    predicted_survival = model.predict_survival_function(
        data, times=range(0, x_max, 30 * step)
    )
    average_survival = np.mean(predicted_survival.values, axis=1)
    std_survival = np.std(predicted_survival.values, axis=1)
    plt.plot(
        range(0, x_max, 30 * step), average_survival, color="orange", label="Predicted"
    )
    plt.fill_between(
        range(0, x_max, 30 * step),
        average_survival - std_survival,
        average_survival + std_survival,
        color="orange",
        alpha=0.2,
    )

    # ax.set_title(f"Test Set Survival Curves")
    ax.set_xticks(range(0, x_max, 30 * step), labels=range(0, x_max // 30 + 1, step))
    ax.set_xlabel("Months")
    ax.legend()
    plt.tight_layout()

    if save:
        filename = "test_survival_curves"
        save_figures(fig, save_dir, filename)


def plot_predictions_over_sigmoid(
    model: LogisticRegression,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
    threshold: Optional[float],
    save_dir: Path,
    partition: str = "test",
    save: bool = True,
):
    fig = plt.figure()
    model_response = np.matmul(features.values, model.coef_.T) + model.intercept_
    model_probabilities = expit(model_response).ravel()
    x_min, x_max = np.min(model_response) - 1, np.max(model_response) + 1
    sigmoid_x = np.linspace(x_min, x_max, 300)
    sigmoid_y = expit(sigmoid_x).ravel()
    plt.plot(
        sigmoid_x,
        sigmoid_y,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    if not (threshold is None):
        plt.axhline(
            threshold,
            xmin=x_min,
            xmax=x_max,
            color="lightgray",
            linestyle="--",
            linewidth=1,
            label=f"Optimal Threshold = {threshold:.2f}",
        )
    positive = endpoints == 1
    negative = endpoints == 0
    plt.scatter(
        model_response[positive],
        model_probabilities[positive],
        label="Acute Recurrence",
        color="red",
        marker="x",
    )
    plt.scatter(
        model_response[negative],
        model_probabilities[negative],
        label="No Acute Recurrence",
        color="green",
        marker="o",
    )
    plt.legend()
    plt.xlabel("Model Response")
    plt.ylabel("Probability")

    if save:
        filename = f"{partition}_predictions_over_sigmoid"
        save_figures(fig, save_dir, filename)


def save_figures(figure: Figure, path: Path, filename: str):
    if not path.exists():
        path.mkdir()
    (path / "pdf").mkdir(exist_ok=True)
    (path / "png").mkdir(exist_ok=True)
    filename = filename.replace(" ", "_").replace(":", "_")
    figure.savefig(
        (path / "pdf" / filename).with_suffix(".pdf"),
        transparent=None,
        dpi=300,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    figure.savefig(
        (path / "png" / filename).with_suffix(".png"),
        transparent=None,
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
