import streamlit as st
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None
    }
    return metrics

def plot_metrics_bar(metrics_dict, title="Model Metrics"):
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_names = list(metrics_dict.keys())
    metric_values = [v if v is not None else 0 for v in metrics_dict.values()]
    colors = ['dodgerblue', 'crimson', 'limegreen', 'orange', 'mediumpurple', 'turquoise']
    bars = ax.bar(metric_names, metric_values, color=colors[:len(metric_names)])
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_prediction_distribution(y_true, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    actual_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    ax1.bar(['Low Income (≤50K)', 'High Income (>50K)'], 
            [actual_counts.get(0, 0), actual_counts.get(1, 0)], 
            color=['salmon', 'turquoise'])
    ax1.set_title('Actual Income Distribution')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(['Low Income (≤50K)', 'High Income (>50K)'], 
            [pred_counts.get(0, 0), pred_counts.get(1, 0)], 
            color=['salmon', 'turquoise'])
    ax2.set_title('Predicted Income Distribution')
    ax2.set_ylabel('Count')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_error_analysis(y_true, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correct vs Incorrect
    correct = (y_true == y_pred).sum()
    incorrect = (y_true != y_pred).sum()
    ax1.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
           autopct='%1.1f%%', colors=['limegreen', 'crimson'], startangle=90)
    ax1.set_title('Prediction Accuracy')
    
    # Confusion breakdown
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    labels = [f'True Positive ({tp})', f'True Negative ({tn})', 
             f'False Positive ({fp})', f'False Negative ({fn})']
    sizes = [tp, tn, fp, fn]
    colors = ['green', 'dodgerblue', 'darkorange', 'darkred']
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
           colors=colors, startangle=45)
    ax2.set_title('Prediction Analysis')
    
    plt.tight_layout()
    return fig

st.set_page_config(page_title="Adult Income Classification", layout="wide")
st.title("Adult Income Classification App")

st.subheader("Download Adult Income Test Dataset")
if os.path.exists("model/test.csv"):
    with open("model/test.csv", "rb") as t:
        st.download_button(label="Download test.csv", data=t, file_name="test.csv", mime="text/csv")
else:
    st.warning("Test dataset not found!")
    
st.subheader("Upload a CSV file to evaluate different trained models.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Adult Dataset CSV", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess data
    df = load_and_preprocess_data(uploaded_file)
    
    # Show data preview with toggle
    with st.expander("📊 Preview Dataset", expanded=False):
        st.write(f"**Shape of dataset** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))
        
        # Show basic statistics
        if st.checkbox("Display Statistics"):
            st.write(df.describe())

    # Encode categorical columns
    encoders = joblib.load("model/encoders.pkl")
    for col in df.columns:
        if df[col].dtype == "object":
            le = encoders[col]
            df[col] = df[col].map(lambda x : x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Separate features and target
    X = df.drop("income", axis=1)
    y = df["income"]


    # Load scaler
    scaler = joblib.load("model/scaler.pkl")
    X = scaler.transform(X)

    st.subheader("Models")
    
    # Add tabs for single model vs comparison
    tab1, tab2 = st.tabs(["Evaluate Single Model", "Compare All Models"])
    
    with tab1:
        # Model selection dropdown
        model_choice = st.selectbox(
            "Select your Model",
            ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
        )

        # Load selected model
        model = joblib.load(f"model/{model_choice}.pkl")

        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics = calculate_metrics(y, y_pred, y_prob)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{metrics['f1']:.4f}")
        col5.metric("MCC", f"{metrics['mcc']:.4f}")
        col6.metric("AUC", f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A")
        
        # Metrics bar chart
        st.write("### Overview of Performance")
        fig_metrics = plot_metrics_bar(metrics, f"{model_choice.replace('_', ' ').title()} Performance")
        st.pyplot(fig_metrics)

        # Confusion Matrix with heatmap
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.write("**Matrix Values:**")
            st.write(cm)
        
        with col_cm2:
            st.write("**Heatmap:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_choice.replace("_", " ").title()} - Confusion Matrix')
            st.pyplot(fig)
        
        # Additional Visualizations
        with st.expander("📈 Show Additional Visualizations", expanded=False):
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                show_dist = st.checkbox("Prediction Distribution", value=True)
                show_error = st.checkbox("Error Analysis", value=True)
            
            with viz_col2:
                show_feature = st.checkbox("Feature Importance", value=False)
            
            if show_dist:
                st.write("### Prediction Distribution")
                fig_dist = plot_prediction_distribution(y, y_pred)
                st.pyplot(fig_dist)
            
            if show_error:
                st.write("### Error Analysis")
                fig_error = plot_error_analysis(y, y_pred)
                st.pyplot(fig_error)
                
                # Error summary
                tp = ((y == 1) & (y_pred == 1)).sum()
                tn = ((y == 0) & (y_pred == 0)).sum()
                fp = ((y == 0) & (y_pred == 1)).sum()
                fn = ((y == 1) & (y_pred == 0)).sum()
                
                st.write("**Summary:**")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("True Positives", tp)
                col_s2.metric("True Negatives", tn)
                col_s3.metric("False Positives", fp)
                col_s4.metric("False Negatives", fn)
            
            if show_feature:
                st.write("### Top 5 Important Features")
                feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                               'marital_status', 'occupation', 'relationship', 'race', 'sex',
                               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:5]
                    
                    for rank, idx in enumerate(indices, 1):
                        st.write(f"{rank}. **{feature_names[idx]}** - Importance: {importances[idx]:.4f}")
                        
                elif hasattr(model, 'coef_'):
                    coefficients = np.abs(model.coef_[0])
                    indices = np.argsort(coefficients)[::-1][:5]
                    
                    for rank, idx in enumerate(indices, 1):
                        st.write(f"{rank}. **{feature_names[idx]}** - Coefficient: {coefficients[idx]:.4f}")
                else:
                    st.info("Feature importance not available for this model type.")
        
        # Download predictions
        st.subheader("Download Predictions")
        results_df = pd.DataFrame({
            'Actual': y.values,
            'Predicted': y_pred,
            'Correct': y.values == y_pred
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"{model_choice}_predictions.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.write("### Compare All Models")
        
        if st.button("🔄 Run All Models"):
            model_names = ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
            results = []
            results_numeric = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(model_names):
                status_text.text(f"Evaluating {model_name}...")
                
                model = joblib.load(f"model/{model_name}.pkl")
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
                
                metrics = calculate_metrics(y, y_pred, y_prob)
                
                results.append({
                    'Model': model_name.replace("_", " ").title(),
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1']:.4f}",
                    'MCC': f"{metrics['mcc']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
                })
                
                results_numeric.append({
                    'Model': model_name.replace("_", " ").title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1'],
                    'MCC': metrics['mcc'],
                    'AUC': metrics['auc'] if metrics['auc'] else 0
                })
                
                progress_bar.progress((idx + 1) / len(model_names))
            
            status_text.text("✅ All models evaluated!")
            
            # Display comparison table
            comparison_df = pd.DataFrame(results)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create comparison charts
            st.write("### Performance Comparison Charts")
            
            numeric_df = pd.DataFrame(results_numeric)
            
            # Chart 1: Main metrics comparison
            fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            colors_palette = ['dodgerblue', 'crimson', 'limegreen', 'orange']
            
            for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors_palette)):
                ax = axes[idx // 2, idx % 2]
                bars = ax.bar(numeric_df['Model'], numeric_df[metric], color=color, alpha=0.8)
                ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim([0, 1])
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig1)
            
            # Chart 2: All metrics in one grouped bar chart
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            
            x = np.arange(len(numeric_df['Model']))
            width = 0.12
            metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'AUC']
            colors_all = ['dodgerblue', 'crimson', 'limegreen', 'orange', 'mediumpurple', 'turquoise']
            
            for idx, (metric, color) in enumerate(zip(metrics_list, colors_all)):
                offset = width * (idx - len(metrics_list) / 2)
                ax2.bar(x + offset, numeric_df[metric], width, label=metric, color=color, alpha=0.8)
            
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Score')
            ax2.set_title('Complete Metrics Comparison Across All Models', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(numeric_df['Model'], rotation=45, ha='right')
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax2.set_ylim([0, 1.1])
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Chart 3: Heatmap of all metrics
            st.write("### Metrics Heatmap")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            heatmap_data = numeric_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'AUC']].T
            heatmap_data.columns = numeric_df['Model']
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0.5, vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Score'})
            ax3.set_title('Performance Heatmap - All Models', fontweight='bold')
            ax3.set_ylabel('Metric')
            ax3.set_xlabel('Model')
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Best model identification
            st.write("### Best Performing Models")
            best_col1, best_col2, best_col3 = st.columns(3)
            
            best_accuracy_idx = numeric_df['Accuracy'].idxmax()
            best_f1_idx = numeric_df['F1 Score'].idxmax()
            best_auc_idx = numeric_df['AUC'].idxmax()
            
            best_col1.success(f"**Best Accuracy:** {numeric_df.loc[best_accuracy_idx, 'Model']} ({numeric_df.loc[best_accuracy_idx, 'Accuracy']:.4f})")
            best_col2.success(f"**Best F1 Score:** {numeric_df.loc[best_f1_idx, 'Model']} ({numeric_df.loc[best_f1_idx, 'F1 Score']:.4f})")
            best_col3.success(f"**Best AUC:** {numeric_df.loc[best_auc_idx, 'Model']} ({numeric_df.loc[best_auc_idx, 'AUC']:.4f})")
            
            # Download comparison
            csv_comparison = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison CSV",
                data=csv_comparison,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload a CSV file to proceed.")