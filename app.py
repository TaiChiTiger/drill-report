import dash, random
from dash.dependencies import Input, Output
from dash import dcc, Dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from os import path
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def data_preparing(dir_name):
    data_file = dir_name + ".csv"
    data_path = path.join('./datasets', dir_name, data_file)
    data_df = pd.read_csv(data_path, names=["depth", "azimuth", "pitch"])
    info_file = dir_name + "_info.csv"
    info_path = os.path.join("./datasets", dir_name, info_file)
    info_df = pd.read_csv(info_path, encoding="gbk")
    data_df["drift"] = data_df["azimuth"] - info_df["design_azi"].values
    df = pd.merge(data_df, info_df, how="left", left_index=True, right_index=True)
    df = df.fillna(method='ffill')

    return df

def transform_data(df):
    A = df["depth"].values
    C = df["pitch"].values
    D = df["drift"].values
    xs = [0]
    for i in range(1, len(df)):
        xi = (A[i] - A[i-1]) / 2 * (np.cos(C[i] * np.pi / 180) * np.cos(D[i] * np.pi / 180) + \
        np.cos(C[i-1] * np.pi / 180) * np.cos(D[i-1] * np.pi / 180)) + xs[i-1]
        xs.append(xi)
    ys = [0]
    for i in range(1, len(df)):
        yi = (A[i] - A[i-1]) / 2 * (np.cos(C[i] * np.pi / 180) * np.sin(D[i] * np.pi / 180) + \
        np.cos(C[i-1] * np.pi / 180) * np.sin(D[i-1] * np.pi / 180)) + ys[i-1]
        ys.append(yi)
    zs = [0]
    for i in range(1, len(df)):
        zi = (A[i] - A[i-1]) / 2 * (np.sin(C[i] * np.pi / 180) + np.sin(C[i-1] * np.pi / 180)) + zs[i-1]
        zs.append(zi)

    new_df = pd.DataFrame()
    new_df["x"] = xs
    new_df["y"] = ys
    new_df["z"] = zs
    new_df["up_down"] = new_df["z"].shift(-1) - new_df["z"]
    new_df["up_down"] = new_df["up_down"].shift(1)
    new_df["up_down"][0] = 0    
    new_df["left_right"] = new_df["y"].shift(-1) - new_df["y"]
    new_df["left_right"] = new_df["left_right"].shift(1)
    new_df["left_right"][0] = 0   

    return new_df.round(2)   

def calc_table_height(df, base=208, height_per_row=20, char_limit=30, height_padding=16.5):
    '''
    df: The dataframe with only the columns you want to plot
    base: The base height of the table (header without any rows)
    height_per_row: The height that one row requires
    char_limit: If the length of a value crosses this limit, the row's height needs to be expanded to fit the value
    height_padding: Extra height in a row when a length of value exceeds char_limit
    '''
    total_height = 0 + base
    for x in range(df.shape[0]):
        total_height += height_per_row
    for y in range(df.shape[1]):
        if len(str(df.iloc[x][y])) > char_limit:
            total_height += height_padding
    return total_height

def plot_report(df, new_df, design_xs, design_ys, design_zs, file_name):
    info_height = 100
    drill3d_height = 350
    drift2d_height = 200
    table_height = calc_table_height(df) + 218
    total_height = info_height + drill3d_height + drift2d_height * 2 + table_height
    data_df = df.iloc[:, :4]
    data_df = pd.concat([data_df, new_df, df.iloc[:, 4:]], axis=1)
    x = new_df["x"]
    y = new_df["y"]
    z = new_df["z"]
    fig = make_subplots(
        rows=5, cols=1,
        vertical_spacing=0.03,
        shared_xaxes=True,
        specs=[[{"type": "table"}],
            [{"type": "surface"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "table"}]],
        column_widths=[800],
        row_heights=[info_height, drill3d_height, 
                    drift2d_height, drift2d_height, table_height],
        subplot_titles=["钻孔基本信息", "钻孔轨迹三维视图", 
                        "上下偏移量曲线", "左右偏移量曲线", "数据表"]
    )

    fig.add_annotation(text=file_name,
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=1.03,
                        y=1.035,
                        font=dict(
                            family="calibre",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>巷道名称</b>：{}'.format(df["drift_name"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=0.995,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>巷道里程</b>：{}'.format(df["drift_length"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.19,
                        y=0.995,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>巷道方位</b>：{}'.format(df["drift_azi"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.48,
                        y=0.995,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>钻孔编号</b>：{}'.format(df["hole_no"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=0.985,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>钻孔直径</b>：{}'.format(df["hole_dia"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.19,
                        y=0.985,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>设计深度</b>：{}'.format(df["design_depth"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.493,
                        y=0.985,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>设计倾角</b>：{}'.format(df["design_pitch"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.77,
                        y=0.985,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>设计方位角</b>：{}'.format(df["design_azi"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.99,
                        y=0.985,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>开孔高度</b>：{}'.format(df["open_height"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=0.975,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>开孔位置</b>：{}'.format(df["open_loc"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.19,
                        y=0.975,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )


    fig.add_annotation(text='<b>测量人员</b>：{}'.format(df["operator_name"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=-0,
                        y=0.965,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>测量班次</b>：{}'.format(df["operator_shift"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.19,
                        y=0.965,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='<b>测量时间</b>：{}'.format(df["sample_time"].values[0]), 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.51,
                        y=0.965,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_trace(
        go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+lines',
        marker=dict(
            size=3,
            color="#008000",
            opacity=0.8,
        ),
        line=dict(color="#008000"),
        name="测量轨迹",
        showlegend=True
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=design_xs,
            y=design_ys,
            z=design_zs,
            mode="lines",
            line=dict(color="red", width=3),
            opacity=0.6,
            name="设计轨迹",
            showlegend=True
        ),
        row=2, col=1
    )
    fig.update_layout(
        margin=dict(l=0, r=00, b=0, t=200),
        legend=dict(orientation="h"),
        showlegend=True,
        scene=dict(aspectmode="data", 
                camera = dict(
                    eye=dict(x=-1.3, y=-1.3, z=0.9),
                    center=dict(x=2, y=3, z=-3))
            )
    )

    fig.add_trace(
        go.Scatter(
            x=df["depth"],
            y=new_df["up_down"],
            mode='markers+lines',
            marker=dict(
                size=5,
                color="#FFAA33",
            ),
            line=dict(width=0.8)
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="上下+/-偏移量（m）", row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=df["depth"],
            y=new_df["left_right"],
            mode='markers+lines',
            marker=dict(
                size=5,
                color="#C70039",
            ),
            line=dict(width=0.8)
        ),
        row=4, col=1
    )
    fig.update_yaxes(title_text="左右+/-偏移量（m）", row=4, col=1)
    fig.update_xaxes(title_text="深度（m）", row=4, col=1)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>   深度</b><br> （m）", "<b>方位角</b>",
                "<b>倾角</b>", "<b>偏差</b>", "<b>x坐标</b>", "<b>y坐标<b>", "<b>z坐标</b>", 
                "<b>上下偏移量</b><br>     （m）", "<b>左右偏移量</b><br>     （m）"],
                font=dict(size=12),
                fill_color="#C8D4E3",
                align="center"
            ),
            cells=dict(
                values=[data_df[k].round(2).tolist() for k in data_df.iloc[:, :9].columns],
                align = "right")
        ),
        row=5, col=1
    )

    # 5. 添加页脚
    fig.add_annotation(text='测定人：',
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.15,
                        y=-0.01,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='科长：',
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.5,
                        y=-0.01,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.add_annotation(text='副总：',
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0.85,
                        y=-0.01,
                        font=dict(
                            family="微软雅黑",
                            size=14,
                            color='#000000')
                    )
    fig.update_layout(width=800, height=total_height,
        margin=dict(l=80, r=40, t=120, b=50),
        showlegend=False,
        title=dict(
            text="钻孔轨迹数据报告",
            x=0.5,
            font=dict(
                family="Arial",
                size=30,
                color='#000000'
            )
        )
    )
    fig.update_xaxes(rangemode="tozero")

    return fig

dir_name = "13_2_20150811_morning"
df = data_preparing(dir_name)
dd = df["design_depth"].values[0]
dp = df["design_pitch"].values[0]
da = df["design_azi"].values[0]
dds = np.linspace(0, dd, 2)
design_xs = dds * np.cos(dp * np.pi / 180) * np.cos((da - 90) * np.pi / 180) 
design_ys = dds * np.cos(dp * np.pi / 180) * np.sin((da - 90) * np.pi / 180) 
design_zs = dds * np.sin(dp * np.pi / 180)

new_df = transform_data(df)
fig = plot_report(df, new_df, design_xs, design_ys, design_zs, dir_name)

# 建立仪表盘
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], update_title='更新中...', title='钻孔报告')
app._favicon = ("./img/drill.png")
server = app.server

app.layout = dbc.Container([
    dbc.Row([dbc.Col([
        html.Button("下载", id="btn_report"),
        dcc.Download(id="download-report"),
        ], width={"size": 1, "offset": 11})
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph("report-graph", figure=fig, config={"displayModeBar": False}), 
                width={"size": 10, "offset": 1})
    ], justify='center')
])

# @callback(
#     Output("download-report", "data"),
#     Input("btn_report", "n_clicks"),
#     prevent_initial_call=True,
# )
def download_report(n_clicks):
    report = file_name + ".pdf" 
    export_path = os.path.join("./", report)
    plotly.io.write_image(fig, export_path, format='pdf')
    return dcc.send_file(report)


def init_callbacks(app):
    app.callback(Output("download-report", "data"),
                    Input("btn_report", "n_clicks"),
                    prevent_initial_call=True,)(
        download_report
    )



if __name__ == '__main__':
    app.run_server(debug=False)
