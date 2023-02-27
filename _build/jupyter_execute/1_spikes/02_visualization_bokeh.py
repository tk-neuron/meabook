#!/usr/bin/env python
# coding: utf-8

# # 2. Interactive Visualization with Bokeh

# {doc}`01_visualization`に述べたように，基本的にはmatplotlibで可視化するのが便利だが，グラフ上にマウスカーソルをかざしながら，インタラクティブ（対話式）に情報を確認したい場合もある．対話式の描画ツールとしてPlotly, Bokehなどがあるが，今回はBokehを用いた実装例を述べる．

# In[1]:


import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter


# In[2]:


datadir = '../datasets/01/'
df_map = pd.read_csv(datadir + 'mapping.csv', index_col=0)
df_sp = pd.read_csv(datadir + 'spikes.csv', index_col=0)


# In[3]:


from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, TabPanel, Tabs
from bokeh.palettes import Magma256, Viridis256
from bokeh.transform import transform
output_notebook()


# ## 2.1. Raster Plot

# In[4]:


def rastergram(p1, p2, df_sp, start, end, bin_width=0.01, gaussian=True):
    df_ = df_sp.query('@start < spiketime < @end')
    hist, edges = np.histogram(df_.spiketime.values, range=(start, end), bins=int((end - start) / bin_width))

    if gaussian:
        hist = gaussian_filter(hist, sigma=[2], truncate=4)
        
    p1.line(x=edges[:-1], y=hist, line_width=2.0, line_color='black')
    p1.yaxis.ticker.desired_num_ticks = 2
    p1.yaxis.axis_label = 'spikes'
    p1.background_fill_color = "whitesmoke"
    
    p2.xaxis.axis_label = 'time [s]'
    p2.yaxis.axis_label = 'channel #'
    p2.background_fill_color = "whitesmoke"
    p2.circle(df_.spiketime.values, df_.channel.values, fill_alpha=0.2, size=3, line_color='black')
    return p1, p2


# In[5]:


p1 = figure(width=600, height=100)
p2 = figure(width=600, height=300, x_range=p1.x_range)  # sync x-range across two plots
p1, p2 = rastergram(p1=p1, p2=p2, df_sp=df_sp, start=40.0, end=50.0, bin_width=0.01)

p1.xaxis.visible = False
show(column(p1, p2))


# ```{note}
# 右側のツールバーにある虫眼鏡のマークから，ドラッグによりグラフの任意の部分を拡大することができる．
# ```

# ## 2.2. Electrode Mapping

# Bokehでは，マウスをホバーするとデータ点の属性を表示してくれるTooltipsという便利な機能がある．この機能を利用して，channel id等をインタラクティブに取得することができる．

# In[6]:


def channel_stats(df_sp: pd.DataFrame, df_map: pd.DataFrame):
    duration = df_sp.spiketime.max() - df_sp.spiketime.min()
    groups = df_sp[['channel', 'amplitude']].groupby('channel')
    
    df_fr = pd.DataFrame(groups.size() / duration, columns=['firing_rate'])  # firing rate for each channel
    df_amp = groups.mean()  # mean spike amplitude for each channel
    
    df_stat = pd.concat([df_map.set_index('channel'), df_fr, df_amp], axis=1, join='inner')
    return df_stat


# In[7]:


def heatmap(p, df, column, vmin, vmax, colors=Magma256):
    source = ColumnDataSource(df)
    mapper = LinearColorMapper(palette=colors, low=vmin, high=vmax)
    p.rect(x="x", y="y", width=17.5, height=17.5, source=source, line_color=None, fill_color=transform(column, mapper))
    p.background_fill_color = 'black'
    
    cbar = ColorBar(color_mapper=mapper, location=(0, 0), width=8, ticker=BasicTicker(desired_num_ticks=5), title='[Hz]')
    p.add_layout(cbar, 'right')
    return p


# In[8]:


df_stat = channel_stats(df_sp, df_map)

# properties shown interactively
TOOLTIPS = [
    ("channel id", "@channel"),
    ("(x, y)", "(@x, @y)"),
    ("firing rate", "@firing_rate"),
    ("amplitude", "@amplitude")
]
p1 = figure(title='', tooltips=TOOLTIPS, width=600, height=360, x_range=(0.0, 17.5*220), y_range=(0.0, 17.5*120))
heatmap(p=p1, df=df_stat, column='firing_rate', vmin=0.0, vmax=3.0)

p2 = figure(title='', tooltips=TOOLTIPS, width=600, height=360, x_range=p1.x_range, y_range=p1.y_range)
heatmap(p=p2, df=df_stat, column='amplitude', vmin=0.0, vmax=600.0, colors=Viridis256)

tab1 = TabPanel(child=p1, title="Firing Rate")
tab2 = TabPanel(child=p2, title="Mean Amplitude")
show(Tabs(tabs=[tab1, tab2]))


# In[ ]:




