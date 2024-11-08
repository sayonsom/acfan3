import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression, SelectFromModel, VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression
import shap
from pylatex import Document, Section, Subsection, Command, NoEscape, Figure, Tabular, LongTable

from pylatex.utils import bold
import numpy as np


# Load the data
data = pd.read_csv('synthetic_ac_fan_data.csv')

# Create a LaTeX document
doc = Document("Data_Exploration_Report")
doc.preamble.append(Command('title', 'Data Exploration Report'))
doc.preamble.append(Command('author', 'SRI-B Team'))
doc.preamble.append(NoEscape(r'\usepackage{pdflscape}'))  # Add pdflscape package for landscape support
doc.append(NoEscape(r'\maketitle'))

# 1. Correlation Analysis
doc.append(Section('Correlation Analysis'))

# Compute correlations with target variables
correlation_matrix = data.corr()
correlations_desired_air_velocity = correlation_matrix['Desired_AirVelocity'].sort_values(ascending=False)
correlations_ac_setpoint_mrt = correlation_matrix['AC_Setpoint(MRT)'].sort_values(ascending=False)

# Add correlation tables to LaTeX report
with doc.create(Subsection('Correlations with Desired_AirVelocity')):
    with doc.create(LongTable('|c|c|')) as table:
        table.add_hline()
        table.add_row(["Feature", "Correlation"])
        table.add_hline()
        for feature, corr_value in correlations_desired_air_velocity.items():
            table.add_row([feature, f"{corr_value:.4f}"])
            table.add_hline()

with doc.create(Subsection('Correlations with AC_Setpoint(MRT)')):
    with doc.create(LongTable('|c|c|')) as table:
        table.add_hline()
        table.add_row(["Feature", "Correlation"])
        table.add_hline()
        for feature, corr_value in correlations_ac_setpoint_mrt.items():
            table.add_row([feature, f"{corr_value:.4f}"])
            table.add_hline()

# Plot and add the full correlation matrix heatmap
correlation_img = 'correlation_matrix.png'
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.savefig(correlation_img)
plt.close()
with doc.create(Figure(position='h!')) as correlation_fig:
    correlation_fig.add_image(correlation_img, width='250px')
    correlation_fig.add_caption('Correlation Matrix Heatmap')

# 2. Variance Threshold
doc.append(Section('Variance Threshold Feature Selection'))
selector = VarianceThreshold(threshold=0.01)
selected_variance = selector.fit_transform(data)
selected_features_variance = data.columns[selector.get_support()].tolist()
doc.append(f"The selected features using Variance Threshold are: {', '.join(selected_features_variance)}")

# 3. Univariate Selection
doc.append(Section('Univariate Selection'))
X = data.drop(columns=['Desired_AirVelocity', 'AC_Setpoint(MRT)'])
y = data['Desired_AirVelocity']
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X, y)
selected_features_univariate = X.columns[selector.get_support()].tolist()
doc.append(f"The selected features using Univariate Selection are: {', '.join(selected_features_univariate)}")

# 4. Feature Importance using Random Forest
doc.append(Section('Random Forest Feature Importance'))
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
rf_img = 'rf_feature_importance.png'
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Random Forest Feature Importance')
plt.savefig(rf_img)
plt.close()
with doc.create(Figure(position='h!')) as rf_fig:
    rf_fig.add_image(rf_img, width='250px')
    rf_fig.add_caption('Random Forest Feature Importance')

# 5. Recursive Feature Elimination (RFE)
doc.append(Section('Recursive Feature Elimination (RFE)'))
rfe = RFE(LinearRegression(), n_features_to_select=10)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_].tolist()
doc.append(f"The selected features using RFE are: {', '.join(selected_features_rfe)}")

# 6. Lasso Regression
doc.append(Section('Lasso Regression Feature Selection'))
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
model = SelectFromModel(lasso, prefit=True)
selected_features_lasso = X.columns[model.get_support()].tolist()
doc.append(f"The selected features using Lasso are: {', '.join(selected_features_lasso)}")

# 7. SHAP Analysis
# doc.append(Section('SHAP Feature Importance'))
# explainer = shap.Explainer(rf, X)
# shap_values = explainer(X)
# shap_img = 'shap_summary_plot.png'
# shap.summary_plot(shap_values, X, show=False)
# plt.savefig(shap_img)
# plt.close()
# with doc.create(Figure(position='h!')) as shap_fig:
#     shap_fig.add_image(shap_img, width='250px')
#     shap_fig.add_caption('SHAP Summary Plot for Feature Importance')

# 8. Summary Table of Selected Features in Landscape Mode
doc.append(Section('Summary of Selected Features by Different Methods'))

# Start the landscape environment manually
doc.append(NoEscape(r'\begin{landscape}'))
with doc.create(LongTable('|c|c|')) as summary_table:
    summary_table.add_hline()
    summary_table.add_row(["Method", "Selected Features"])
    summary_table.add_hline()
    summary_table.add_row(["Variance Threshold", ', '.join(selected_features_variance)])
    summary_table.add_hline()
    summary_table.add_row(["Univariate Selection", ', '.join(selected_features_univariate)])
    summary_table.add_hline()
    summary_table.add_row(["Random Forest Importance", ', '.join(feature_importance['Feature'][:10])])
    summary_table.add_hline()
    summary_table.add_row(["RFE", ', '.join(selected_features_rfe)])
    summary_table.add_hline()
    summary_table.add_row(["Lasso", ', '.join(selected_features_lasso)])
    summary_table.add_hline()

# End the landscape environment manually
doc.append(NoEscape(r'\end{landscape}'))

# Save the LaTeX document
doc.generate_tex('data_exploration_report')