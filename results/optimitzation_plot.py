import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1) Carrega seus dados
df = pd.read_csv('optimization_results/optimization_history.csv')

# 2) Cria colunas de log10
df['lr_log'] = np.log10(df['Learning Rate'])
df['wd_log'] = np.log10(df['Weight Decay'])

# 3) Define bins de log (por exemplo de -8 a -2, passo 1)
log_bins = np.arange(-8, -1, 1)  # [-8, -7, ..., -2]
# Cria categorias de bin com rótulo no centro de cada intervalo
df['lr_bin'] = pd.cut(df['lr_log'], bins=log_bins, labels=(log_bins[:-1]))
df['wd_bin'] = pd.cut(df['wd_log'], bins=log_bins, labels=(log_bins[:-1]))

# 4) Parâmetros para plot
params = [
    ('lr_bin',            'Learning Rate (log)', 'log_binned'),
    ('wd_bin',            'Weight Decay (log)',   'log_binned'),
    ('Batch Size',        'Batch Size',                  'category'),
    ('fold',              'Fold',                        'category'),
    ('Optimizer',         'Optimizer',                   'category'),
    ('Scheduler',         'Scheduler',                   'category'),
    ('Accumulation Steps','Accumulation Steps',          'category'),
]

# 5) Cria subplots 2 colunas
n = len(params)
n_rows = math.ceil(n/2)
fig = make_subplots(
    rows=n_rows, cols=2,
    shared_xaxes=False,
    vertical_spacing=0.15,
    horizontal_spacing=0.02,
    subplot_titles=[lab for _, lab, _ in params]
)

# 6) Configura coloraxis único
cmin, cmax = df['F1 Score'].min(), df['F1 Score'].max()
fig.update_layout(
    coloraxis=dict(
        colorscale='rdbu',
        cmin=cmin, cmax=cmax,
        colorbar=dict(title='F1-Score', len=0.9)
    ),
    height=140*n_rows, width=1000,
)

# 7) Loop para cada parâmetro
for idx, (col, label, scale) in enumerate(params, start=1):
    # agrupamento
    grp = df.groupby(col)['F1 Score'].mean().reset_index().dropna()
    # ordena
    if scale in ('log_binned','linear'):
        grp = grp.sort_values(col)
    # x e z
    x = grp[col].astype(float) if 'log' in scale else grp[col].astype(str)
    z = [grp['F1 Score'].values]

    # posição no grid
    row = math.ceil(idx/2)
    col_nr = 2 if idx%2==0 else 1

    fig.add_trace(
        go.Heatmap(
            x=x, y=[label],
            z=z,
            coloraxis='coloraxis',
            hovertemplate=f'{label}: %{{x}}<br>F1-Score: %{{z:.3f}}<extra></extra>'
        ),
        row=row, col=col_nr
    )

    # eixo X
    if scale=='log_binned':
        fig.update_xaxes(
            type='linear',
            tickmode='array',
            tickvals=log_bins[:-1],
            ticktext=[f'{b:.1f}' for b in (log_bins[:-1])],
            row=row, col=col_nr
        )
    elif scale=='category':
        fig.update_xaxes(type='category', row=row, col=col_nr)
    else:
        fig.update_xaxes(type='linear', row=row, col=col_nr)

    # esconde Y
    fig.update_yaxes(showticklabels=False, row=row, col=col_nr)

# 8) Exibe
fig.show()
