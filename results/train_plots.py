import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Carregando os dados
train_data = pd.read_csv('results/train_result.csv')
val_data = pd.read_csv('results/val_result.csv')
train_loss = pd.read_csv('results/loss_train.csv')
val_loss = pd.read_csv('results/loss_val.csv')

# Configurações de estilo
font_size = 14
line_width = 2
marker_size = 8
template = 'plotly_white'
grid_color = 'rgba(128, 128, 128, 0.2)'  # Cor cinza mais escura para as grades

# Criando o gráfico de acurácia
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(
    x=train_data['Step'],
    y=train_data['Value'],
    name='Training',
    mode='lines+markers',
    line=dict(width=line_width),
    marker=dict(size=marker_size)
))
fig_acc.add_trace(go.Scatter(
    x=val_data['Step'],
    y=val_data['Value'],
    name='Validation',
    mode='lines+markers',
    line=dict(width=line_width),
    marker=dict(size=marker_size)
))

fig_acc.update_layout(
    template=template,
    title=dict(
        x=0.5,
        y=0.95,
        font=dict(size=font_size + 2)
    ),
    xaxis=dict(
        title=dict(
            text='Epochs',
            font=dict(size=font_size)
        ),
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1
    ),
    yaxis=dict(
        title=dict(
            text='Accuracy',
            font=dict(size=font_size)
        ),
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1
    ),
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1
    ),
    margin=dict(l=50, r=20, t=50, b=50),
    width=1500,
    height=600
)

# Criando o gráfico de loss
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(
    x=train_loss['Step'],
    y=train_loss['Value'],
    name='Training',
    mode='lines+markers',
    line=dict(width=line_width),
    marker=dict(size=marker_size)
))
fig_loss.add_trace(go.Scatter(
    x=val_loss['Step'],
    y=val_loss['Value'],
    name='Validation',
    mode='lines+markers',
    line=dict(width=line_width),
    marker=dict(size=marker_size)
))

fig_loss.update_layout(
    template=template,
    title=dict(
        x=0.5,
        y=0.95,
        font=dict(size=font_size + 2)
    ),
    xaxis=dict(
        title=dict(
            text='Epochs',
            font=dict(size=font_size)
        ),
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1
    ),
    yaxis=dict(
        title=dict(
            text='Loss',
            font=dict(size=font_size)
        ),
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1
    ),
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1
    ),
    margin=dict(l=50, r=20, t=50, b=50),
    width=1500,
    height=600
)

# Salvar os gráficos como arquivos HTML
fig_acc.write_html('results/accuracy_plot.html')
fig_loss.write_html('results/loss_plot.html')

# Mostrar os gráficos
fig_acc.show()
fig_loss.show()