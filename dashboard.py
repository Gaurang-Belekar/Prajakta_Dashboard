import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from arch import arch_model
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========== Load Data ==========
df_main = pd.read_csv("final_did_panel_fully_cleaned.csv")
df_macro = pd.read_csv("macro_controls_annual.csv")
df_vol = pd.read_csv("overall_avg_annual_volatility.csv")
df_gpr = pd.read_csv("annual_gpr_index.csv")
df_fx = pd.read_csv("annual_forex_volatility_by_currency.csv")

# ========== Preprocessing ==========
hedging_vars = ['HSI', 'HedgingIndex', 'DerivativesFairValue', 'SizNumFXHedge',
                'TypesOfDerivativesUsed', 'ForeignRevenuePct', 'ForeignAssetsPct', 'CountriesOperated']
top_vars = df_main[hedging_vars].corrwith(df_main['AvgAnnualVolatility']).abs().sort_values(ascending=False).head(4).index.tolist()
scaler = StandardScaler()
Z_top = scaler.fit_transform(df_main[top_vars])
df_main['CHS_Top'] = Z_top.mean(axis=1)
df_main['CHS_GPR_Top'] = df_main['CHS_Top'] * df_main['AvgAnnualGPR']

# ========== OLS ==========
X_ols = sm.add_constant(df_main[['AvgAnnualGPR', 'CHS_Top', 'CHS_GPR_Top', 'GES', 'CPI', 'TradeBalance']])
ols_model = sm.OLS(df_main['AvgAnnualVolatility'], X_ols).fit(cov_type='HC3')
ols_summary_df = pd.DataFrame({
    "Variable": X_ols.columns,
    "Coefficient": ols_model.params.values,
    "P-value": ols_model.pvalues.values
})

# ========== VIF ==========
vif_data = pd.DataFrame()
vif_data["Variable"] = X_ols.columns
vif_data["VIF"] = [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])]

# ========== Ridge ==========
X_ridge = df_main[['AvgAnnualGPR', 'CHS_Top', 'CHS_GPR_Top', 'GES', 'CPI', 'InterestRate', 'TradeBalance']]
y_ridge = df_main['AvgAnnualVolatility']
ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(X_ridge, y_ridge)
ridge_coefs = pd.DataFrame({
    "Variable": X_ridge.columns,
    "Coefficient": ridge_model.coef_
})

# ========== GARCH ==========
returns = np.log(df_main['AvgAnnualVolatility']).diff().dropna()
garch_X = df_main['AvgAnnualGPR'].iloc[1:len(returns)+1].values
garch_model = arch_model(returns*10, mean='ARX', x=garch_X, lags=1, vol='GARCH', p=1, q=1, dist='normal')
garch_result = garch_model.fit(disp='off')

# ========== Marginal Effects ==========
df_main['PredictedVolatility'] = ols_model.predict(X_ols)
df_main['CHS_bin'] = pd.qcut(df_main['CHS_Top'], 4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
df_main['GPR_bin'] = pd.qcut(df_main['AvgAnnualGPR'], 5).astype(str)
avg_pred = df_main.groupby(['CHS_bin', 'GPR_bin'], observed=False)['PredictedVolatility'].mean().reset_index()

# ========== App Setup ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Volatility & Hedging Dashboard"

header = dbc.NavbarSimple(
    brand="Volatility & Hedging Analytics Dashboard",
    color="dark",
    dark=True,
    fluid=True,
)

# ========== Layout ==========
app.layout = html.Div([
    header,
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            dbc.Container([
                html.Br(),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=px.line(df_vol, x='Year', y='AvgAnnualVolatility',
                                                     title='Average Annual Volatility',
                                                     markers=True)), md=6),
                    dbc.Col(dcc.Graph(figure=px.line(df_gpr, x='Year', y='AvgAnnualGPR',
                                                     title='Average Annual GPR',
                                                     markers=True)), md=6)
                ])
            ])
        ]),

        dcc.Tab(label='FX Volatility Heatmap', children=[
            dbc.Container([
                html.Br(),
                dcc.Graph(figure=px.imshow(df_fx.set_index('Year').T,
                                           title="FX Volatility by Currency",
                                           color_continuous_scale='Blues'))
            ])
        ]),

        dcc.Tab(label='OLS & Multicollinearity', children=[
            dbc.Container([
                html.Br(),
                html.H5("OLS Regression Summary"),
                dash_table.DataTable(
                    data=ols_summary_df.round(6).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in ols_summary_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                ),
                html.Br(),
                html.H5("Variance Inflation Factors (VIF)"),
                dash_table.DataTable(
                    data=vif_data.round(3).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in vif_data.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                )
            ])
        ]),

        dcc.Tab(label='Ridge & GARCH Model', children=[
            dbc.Container([
                html.Br(),
                html.H5("Ridge Regression Coefficients"),
                dash_table.DataTable(
                    data=ridge_coefs.round(6).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in ridge_coefs.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                ),
                html.Br(),
                html.H5("GARCH(1,1) Model Summary"),
                html.Pre(garch_result.summary().as_text(), style={'whiteSpace': 'pre-wrap', 'fontSize': '13px'})
            ])
        ]),

        dcc.Tab(label='Marginal Effect Visualization', children=[
            dbc.Container([
                html.Br(),
                dcc.Graph(figure=px.line(avg_pred, x='GPR_bin', y='PredictedVolatility',
                                         color='CHS_bin', markers=True,
                                         title="Marginal Effect of CHS on Volatility across GPR Quintiles"))
            ])
        ])
    ])
])

if __name__ == '__main__':
    app.run(debug=True)


# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, dash_table
# from dash.dependencies import Input, Output
# import statsmodels.api as sm
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import RidgeCV
# from arch import arch_model

# # Load data
# df_main = pd.read_csv("final_did_panel_fully_cleaned.csv")
# df_macro = pd.read_csv("macro_controls_annual.csv")
# df_vol = pd.read_csv("overall_avg_annual_volatility.csv")
# df_gpr = pd.read_csv("annual_gpr_index.csv")
# df_fx = pd.read_csv("annual_forex_volatility_by_currency.csv")

# # Top 4 correlated variables with volatility
# hedging_vars = ['HSI', 'HedgingIndex', 'DerivativesFairValue', 'SizNumFXHedge',
#                 'TypesOfDerivativesUsed', 'ForeignRevenuePct', 'ForeignAssetsPct', 'CountriesOperated']
# top_vars = df_main[hedging_vars].corrwith(df_main['AvgAnnualVolatility']).abs().sort_values(ascending=False).head(4).index.tolist()
# scaler = StandardScaler()
# Z_top = scaler.fit_transform(df_main[top_vars])
# df_main['CHS_Top'] = Z_top.mean(axis=1)
# df_main['CHS_GPR_Top'] = df_main['CHS_Top'] * df_main['AvgAnnualGPR']

# # === OLS Model ===
# X_ols = sm.add_constant(df_main[['AvgAnnualGPR', 'CHS_Top', 'CHS_GPR_Top', 'GES', 'CPI', 'TradeBalance']])
# ols_model = sm.OLS(df_main['AvgAnnualVolatility'], X_ols).fit(cov_type='HC3')
# ols_summary_df = pd.DataFrame({
#     "Variable": X_ols.columns,
#     "Coefficient": ols_model.params.values,
#     "P-value": ols_model.pvalues.values
# })

# # === VIF ===
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# vif_data = pd.DataFrame()
# vif_data["Variable"] = X_ols.columns
# vif_data["VIF"] = [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])]

# # === Ridge ===
# X_ridge = df_main[['AvgAnnualGPR', 'CHS_Top', 'CHS_GPR_Top', 'GES', 'CPI', 'InterestRate', 'TradeBalance']]
# y_ridge = df_main['AvgAnnualVolatility']
# ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 100)).fit(X_ridge, y_ridge)
# ridge_coefs = pd.DataFrame({
#     "Variable": X_ridge.columns,
#     "Coefficient": ridge_model.coef_
# })

# # === GARCH ===
# returns = np.log(df_main['AvgAnnualVolatility']).diff().dropna()
# garch_X = df_main['AvgAnnualGPR'].iloc[1:len(returns)+1].values
# garch_model = arch_model(returns*10, mean='ARX', x=garch_X, lags=1, vol='GARCH', p=1, q=1, dist='normal')
# garch_result = garch_model.fit(disp='off')

# # === Marginal Effect Plot Data ===
# df_main['PredictedVolatility'] = ols_model.predict(X_ols)
# df_main['CHS_bin'] = pd.qcut(df_main['CHS_Top'], 4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
# df_main['GPR_bin'] = pd.qcut(df_main['AvgAnnualGPR'], 5).astype(str)
# avg_pred = df_main.groupby(['CHS_bin', 'GPR_bin'], observed=False)['PredictedVolatility'].mean().reset_index()

# # === Dash Layout ===
# app = Dash(__name__)
# app.title = "Volatility & Hedging Analytics Dashboard"

# app.layout = html.Div([
#     html.H1("Volatility and Hedging Dashboard", style={'textAlign': 'center'}),

#     dcc.Tabs([
#         dcc.Tab(label='Time Trends', children=[
#             dcc.Graph(figure=px.line(df_vol, x='Year', y='AvgAnnualVolatility', title='Avg Annual Volatility')),
#             dcc.Graph(figure=px.line(df_gpr, x='Year', y='AvgAnnualGPR', title='Geopolitical Risk Index'))
#         ]),

#         dcc.Tab(label='Forex Heatmap', children=[
#             dcc.Graph(figure=px.imshow(df_fx.set_index('Year').T, 
#                                        title="Annual FX Volatility Heatmap",
#                                        color_continuous_scale='Blues'))
#         ]),

#         dcc.Tab(label='OLS & VIF', children=[
#             html.H3("OLS Summary"),
#             dash_table.DataTable(ols_summary_df.round(6).to_dict('records'), columns=[{"name": i, "id": i} for i in ols_summary_df.columns]),
#             html.H3("VIF Table"),
#             dash_table.DataTable(vif_data.round(3).to_dict('records'), columns=[{"name": i, "id": i} for i in vif_data.columns])
#         ]),

#         dcc.Tab(label='Ridge & GARCH', children=[
#             html.H3("Ridge Coefficients"),
#             dash_table.DataTable(ridge_coefs.round(6).to_dict('records'), columns=[{"name": i, "id": i} for i in ridge_coefs.columns]),
#             html.H3("GARCH Summary (text only)"),
#             html.Pre(garch_result.summary().as_text())
#         ]),

#         dcc.Tab(label='Marginal Effect', children=[
#             dcc.Graph(figure=px.line(avg_pred, x='GPR_bin', y='PredictedVolatility', color='CHS_bin',
#                                      markers=True, title="Marginal Effect of CHS on Volatility Across GPR Quintiles"))
#         ])
#     ])
# ])

# if __name__ == '__main__':
#     app.run(debug=True)

