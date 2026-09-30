"""
Tear-sheet charts written to results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import FEATURE_LABELS

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')  # fixed categorical order; never cycled
POSITIVE, NEGATIVE = '#2a78d6', '#e34948'  # diverging poles (long/short, +/- coef)

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'],
    'font.size': 9,
    'text.color': INK,
    'axes.labelcolor': INK_2,
    'figure.facecolor': SURFACE,
})


def _style(ax, title: str, ylabel: str = '') -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title(title, loc='left', fontsize=10, fontweight='semibold', color=INK, pad=8)
    ax.set_ylabel(ylabel)


def tearsheet(results: dict, mkt: pd.DataFrame, main_set: str, path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.3, 1.6, 1.3], 'hspace': 0.45})
    ax_eq, ax_dd, ax_px, ax_pos = axes

    # Equity: one line per feature set, legend + end value labels.
    for (name, res), color in zip(results.items(), SERIES):
        growth = (np.exp(res['bt']['equity']) - 1) * 100
        lw = 2 if name == main_set else 1.5
        ax_eq.plot(growth.index, growth, color=color, lw=lw, solid_capstyle='round', label=name)
        ax_eq.plot(growth.index[-1], growth.iloc[-1], 'o', ms=8, color=color, mec=SURFACE, mew=2)
        ax_eq.annotate(f'{growth.iloc[-1]:+.0f}%', (growth.index[-1], growth.iloc[-1]),
                       xytext=(8, 0), textcoords='offset points', va='center', color=INK_2)
    ax_eq.axhline(0, color=BASELINE, lw=1)
    _style(ax_eq, 'Cumulative return vs seasonal path, after costs (futures proxy)', '%')
    ax_eq.legend(loc='upper left', frameon=False, labelcolor=INK_2)

    bt = results[main_set]['bt']
    dd = (np.exp(bt['equity'] - bt['equity'].cummax()) - 1) * 100
    ax_dd.fill_between(dd.index, dd, 0, color=NEGATIVE, alpha=0.10, lw=0)
    ax_dd.plot(dd.index, dd, color=NEGATIVE, lw=1.5)
    _style(ax_dd, f'Drawdown: {main_set}', '%')

    px = mkt['price'].loc[bt.index[0]:]
    ax_px.plot(px.index, px, color=INK_2, lw=1.5)
    _style(ax_px, 'SSB fresh salmon export price', 'NOK/kg')

    pos = bt['position']
    for part, color, label in ((pos.clip(lower=0), POSITIVE, 'Long'),
                               (pos.clip(upper=0), NEGATIVE, 'Short')):
        ax_pos.fill_between(pos.index, part, 0, color=color, alpha=0.10, lw=0, step='post')
        ax_pos.step(pos.index, part, where='post', color=color, lw=1.2, label=label)
    ax_pos.axhline(0, color=BASELINE, lw=1)
    lim = max(1.0, pos.abs().max()) * 1.05
    ax_pos.set_ylim(-lim, lim)
    _style(ax_pos, f'Position: {main_set} (notional / capital, after vol cap and leverage)')
    ax_pos.legend(loc='upper left', frameon=False, ncol=2, labelcolor=INK_2)

    ax_eq.margins(x=0.06)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def coefficients(coefs: pd.Series, path: Path) -> None:
    """Standardised ridge coefficients: effect of a 1-sd move on the forecast."""
    coefs = coefs.sort_values()
    labels = [FEATURE_LABELS.get(c, c) for c in coefs.index]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(coefs) + 1.2))
    colors = [POSITIVE if v > 0 else NEGATIVE for v in coefs]
    ax.barh(labels, coefs * 100, height=0.6, color=colors)
    ax.axvline(0, color=BASELINE, lw=1)
    _style(ax, 'Ridge coefficients: forecast change per 1-sd feature move (% over horizon)')
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=1)
    ax.tick_params(axis='y', colors=INK_2)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
