import os
import shutil
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import shutil
import random
from collections import defaultdict
import json
import os
import sys
import io
from contextlib import contextmanager
import base64
import io
from IPython.display import display, HTML, SVG
import matplotlib.pyplot as plt
import PIL.Image

import pandas as pd
import numpy as np
import json
import re
from collections import Counter, defaultdict
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from settings import Settings

settings = Settings()

def setup_plotly_for_notebook():
    """
    Setup Plotly for proper display in Jupyter notebooks.
    Call this function at the beginning of your notebook if you're having display issues.
    """
    try:
        pio.renderers.default = "notebook"
        print("Plotly configured for notebook display")
    except Exception as e:
        try:
            pio.renderers.default = "plotly_mimetype+notebook"
            print("Plotly configured with plotly_mimetype+notebook renderer")
        except Exception as e2:
            print(f"Warning: Could not configure Plotly renderer. Using default. Error: {e}")
    
    # Also set the template for better appearance
    pio.templates.default = "plotly_white"

def delete_all_files_and_folders(folder_path):
    """
    Delete all files and folders in the specified directory.
    
    Args:
        folder_path (str): Path to the directory to be cleared
    """
    # Check if the path exists
    if not os.path.exists(folder_path):
        print(f"Error: The path '{folder_path}' does not exist.")
        return False
    
    # Check if the path is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return False
    
    # Get the list of all items in the directory
    items = os.listdir(folder_path)
    
    if not items:
        print(f"The directory '{folder_path}' is already empty.")
        return True
    
    # Display what will be deleted
    print(f"The following items will be deleted from '{folder_path}':")
    for item in items:
        print(f"- {item}")
    
    # # Ask for confirmation
    # confirmation = input("\nAre you sure you want to delete all these items? (yes/no): ").lower()
    
    # if confirmation != "yes":
    #     print("Operation cancelled.")
    #     return False
    
    # Delete all items
    error_count = 0
    for item in items:
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")
            error_count += 1
    
    if error_count == 0:
        print("\nAll items were successfully deleted.")
        return True
    else:
        print(f"\nOperation completed with {error_count} errors.")
        return False


def evaluation_metrics(log_path):
    folders = os.listdir(log_path)
    if 'evaluation.log' in folders:
        folders.remove('evaluation.log')

    # Initialize an empty list to store all evaluation metrics
    all_metrics = []
    print("Folders:",folders)
    for folder in folders:
        result_path = os.path.join(log_path, folder, 'result.json')
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            
            # Extract evaluations for each folder
            if 'evaluations' in data and len(data['evaluations']) > 0:
                for metrics in data['evaluations'][0]:
                    # Add the folder ID to each metric entry
                    metric_data = {
                        'folder_id': folder,
                        'aspect': metrics['aspect'],
                        'answer': metrics['answer'],
                        'rationale': metrics['rationale']
                    }
                    all_metrics.append(metric_data)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")

    # Create DataFrame from the collected metrics
    metrics_df = pd.DataFrame(all_metrics)

    # Save the DataFrame to CSV (optional)
    metrics_df.to_csv(os.path.join(log_path, 'evaluation_metrics.csv'), index=False)

    index_metrics = pd.DataFrame(metrics_df.groupby('folder_id')['answer'].sum()).reset_index()
    return index_metrics

def remove_svg_files():
    svg_files_path = settings.SVG_FILES_PATH
    svg_files = os.listdir(svg_files_path)
    svg_files = [file_path for file_path in svg_files if os.path.splitext(file_path)[1] =='.svg']
    for file in svg_files:
        if os.path.exists(file):
            os.remove(file)
        else:
            continue

def remove_logs_folder():
    folder_path = settings.LOG_FOLDER
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return False
        
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return False
        
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully removed folder and all its contents: {folder_path}")
        return True
    except Exception as e:
        print(f"Error removing folder '{folder_path}': {e}")
        return False


def metric_visual(results_df, metric):
    # Validate metric input
    valid_metrics = ['pass_rate', 'readability_score', 'quality_score']
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}, got '{metric}'")
    
    # Check if required columns exist
    required_cols = ['model', 'agent', metric]
    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"DataFrame is missing required column: {col}")
    
    # Get unique models and agents
    models = sorted(results_df['model'].unique())
    agents = sorted(results_df['agent'].unique())
    
    # Set up plot 
    plt.figure(figsize=(12, 6))
    
    # Width of a bar group
    width = 0.8 / len(agents)
    
    # Title and label mapping for better display
    metric_titles = {
        'pass_rate': 'Pass Rate',
        'readability_score': 'Readability Score',
        'quality_score': 'Quality Score'
    }
    title = metric_titles.get(metric, metric.replace('_', ' ').title())
    
    # Plot bars for each agent
    for i, agent in enumerate(agents):
        # Filter data for this agent
        agent_data = results_df[results_df['agent'] == agent]
        
        # Calculate position offset for this agent's bars
        offset = (i - len(agents)/2 + 0.5) * width
        
        # Get data for each model for this agent
        heights = []
        x_positions = []
        for j, model in enumerate(models):
            model_data = agent_data[agent_data['model'] == model]
            if not model_data.empty:
                heights.append(model_data[metric].mean())
                x_positions.append(j + offset)
        
        # Plot bars for this agent
        bars = plt.bar(x_positions, heights, width, label=agent)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (results_df[metric].max() * 0.01),
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set x-axis ticks at model positions
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    
    # Customize plot
    plt.title(f'{title} Comparison Across Models and Agents', fontsize=15)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel(title, fontsize=12)
    
    # Set y-axis limits based on metric
    if metric == 'pass_rate':
        plt.ylim(0, min(1.0, results_df[metric].max() * 1.15))  # Pass rate typically 0-1
    else:
        # For readability and quality scores, set based on data range
        min_val = max(0, results_df[metric].min() * 0.9)  # Lower bound, not below 0
        max_val = results_df[metric].max() * 1.15  # Upper bound with 15% padding
        plt.ylim(min_val, max_val)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Agents", loc='best')
    
    # Improve layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def get_equal_samples(viseval_json_path:str, num_samples: int) -> list:
    with open(viseval_json_path, 'r') as f:
        data = json.load(f)

    # Group IDs by hardness level and chart type
    hardness_groups = defaultdict(list)
    chart_groups = defaultdict(list)
    
    for sample_id, sample_data in data.items():
        hardness = sample_data.get("hardness")
        chart_type = sample_data.get("chart")
        
        if hardness:
            hardness_groups[hardness].append(sample_id)
        if chart_type:
            chart_groups[chart_type].append(sample_id)

    required_levels = ['Easy', 'Medium', 'Hard', 'Extra Hard']
    required_charts = ['Grouping Scatter', 'Grouping Line', 'Bar', 'Line', 'Pie', 'Scatter', 'Stacked Bar']

    # Randomly select samples from each hardness level (as many as available, up to num_samples)
    selected_by_hardness = []
    for level in required_levels:
        available_samples = len(hardness_groups[level])
        samples_to_take = min(num_samples, available_samples)
        if available_samples > 0:
            selected_by_hardness.extend(random.sample(hardness_groups[level], samples_to_take))
        # print(f"Hardness level '{level}': {samples_to_take}/{num_samples} samples selected")
    
    # Randomly select samples from each chart type (as many as available, up to num_samples)
    selected_by_chart = []
    for chart in required_charts:
        available_samples = len(chart_groups[chart])
        samples_to_take = min(num_samples, available_samples)
        if available_samples > 0:
            selected_by_chart.extend(random.sample(chart_groups[chart], samples_to_take))
        # print(f"Chart type '{chart}': {samples_to_take}/{num_samples} samples selected")
    
    # Combine both selections, ensuring no duplicates
    selected_ids = list(set(selected_by_hardness + selected_by_chart))
    
    # print(f"Total unique samples selected: {len(selected_ids)}")
    return selected_ids


@contextmanager
def suppress_output():
    # Save the original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Redirect stdout and stderr to a null device
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        yield  # Run the code block inside the with statement
    finally:
        # Restore original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr



# print(svg_output)
def display_svg(svg_content):
    from IPython.display import SVG, display
    
    # Remove any XML declaration if present to avoid display issues
    if '<?xml' in svg_content:
        svg_content = svg_content.split('>', 1)[1]
    
    # Display the SVG
    display(SVG(svg_content))



def display_base64(base64_string, format_type="svg"):
    # Remove the data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    try:
        if format_type.lower() == "svg":
            # For SVG format
            decoded = base64.b64decode(base64_string).decode('utf-8')
            display(SVG(data=decoded))
        
        elif format_type.lower() in ["png", "jpg", "jpeg"]:
            # For image formats
            decoded = base64.b64decode(base64_string)
            img = PIL.Image.open(io.BytesIO(decoded))
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        
        else:
            print(f"Unsupported format: {format_type}")
    
    except Exception as e:
        print(f"Error displaying visualization: {e}")


def plot(data, chart_name, x_col=None, y_col=None, color_col=None, title=None):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import templates
    import plotly.io as pio
    import pandas as pd
    import numpy as np
    import random
    
    # Configure Plotly for notebook display
    try:
        pio.renderers.default = "notebook"
    except:
        try:
            pio.renderers.default = "plotly_mimetype+notebook"
        except:
            pass  # Use default renderer
    
    # Define visually attractive color palettes optimized for plotly_white
    qualitative_colors = [
        # Modern, vibrant colors that stand out on white background
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        # Additional attractive colors
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
    ]
    
    # Always shuffle the qualitative colors
    random.shuffle(qualitative_colors)
    
    # Alternative attractive sequential and diverging color schemes for plotly_white
    color_scales = {
        'qualitative': qualitative_colors,
        'sequential': px.colors.sequential.Viridis,  # Better contrast than Plasma
        'diverging': px.colors.diverging.RdBu      # Clean, professional red-blue scale
    }
    
    # Set template
    template = templates.default = "plotly_white"
    
    # Set default title if none provided
    if title is None:
        title = f"{chart_name.capitalize()} Chart"
    
    # Font styles
    title_font = "Raleway, sans-serif"  # Modern sans-serif font for titles
    text_font = "Open Sans, sans-serif"  # Clean font for axis labels and text
    
    # Determine if the data is integer or float (if applicable)
    def get_text_template(column):
        if column is None:
            return '%{text}'
        
        # Check if column exists in dataframe
        if column in data.columns:
            # Check data type
            if pd.api.types.is_integer_dtype(data[column]) or data[column].apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all():
                return '%{text:.0f}'  # Integer format
            else:
                return '%{text:.2f}'  # Float format with 2 decimal places
        return '%{text}'  # Default
    
    # Create figure based on chart type
    fig = None
    
    # Handle different chart types
    if chart_name.lower() == 'bar':
        if x_col is None or y_col is None:
            raise ValueError("Bar chart requires x_col and y_col")
        
        # Determine text template based on y_col data type
        text_template = get_text_template(y_col)
        
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col or x_col,
            title=title,
            text=y_col,
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        # Configure text position
        fig.update_traces(
            texttemplate=text_template,
            textposition='outside',
            cliponaxis=False,
            marker_line_width=1,  # Add subtle border for better definition
            opacity=0.85  # Slightly transparent for a softer look
        )
        
    elif chart_name.lower() == 'scatter':
        if x_col is None or y_col is None:
            raise ValueError("Scatter plot requires x_col and y_col")
        
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            title=title,
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        
        # Enhanced scatter plot styling
        fig.update_traces(
            marker=dict(
                size=8,
                line=dict(width=1, color='white'),  # White border for better definition
                opacity=0.8  # Slightly transparent for overlapping points
            )
        )
        
        # Add trend line if no color grouping
        if color_col is None:
            # Calculate trend line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_col], data[y_col])
            line_x = [data[x_col].min(), data[x_col].max()]
            line_y = [slope * x + intercept for x in line_x]
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f'Trend (R² = {r_value**2:.3f})',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            ))
        
    # Other chart types remain the same but use color_scales instead of dark_colors
    elif chart_name.lower() == 'line':
        if x_col is None or y_col is None:
            raise ValueError("Line chart requires x_col and y_col")
            
        text_template = get_text_template(y_col)
        
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col, 
            title=title,
            text=y_col,
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        fig.update_traces(
            mode='lines+markers+text',
            texttemplate=text_template,
            textposition='top center',
            marker=dict(size=8, line=dict(width=1, color='white')),
            line=dict(width=3)  # Thicker lines for better visibility
        )
            
    elif chart_name.lower() == 'pie':
        if x_col is None or y_col is None:
            raise ValueError("Pie chart requires names column (x_col) and values column (y_col)")
        
        # For pie charts, determine if values are integers or floats
        is_integer = False
        try:
            if pd.api.types.is_integer_dtype(data[y_col]) or data[y_col].apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all():
                is_integer = True
        except:
            pass
        
        fig = px.pie(
            data, 
            names=x_col, 
            values=y_col,
            title=title,
            color_discrete_sequence=color_scales['qualitative'],
            template=template,
            hole=0.4
        )
        
        # Format text based on data type
        if is_integer:
            fig.update_traces(texttemplate='%{percent:.1%}<br>%{value:.0f}', textinfo='label+percent+value')
        else:
            fig.update_traces(texttemplate='%{percent:.1%}<br>%{value:.2f}', textinfo='label+percent+value')
        
        fig.update_traces(textposition='inside')
    
    elif chart_name.lower() == 'box':
        if x_col is None or y_col is None:
            raise ValueError("Box plot requires x_col and y_col")
        fig = px.box(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            title=title,
            points='all',  # Show all points for better data visibility
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        # Box plots don't support direct data labels, but we show all points
        
    elif chart_name.lower() == 'histogram':
        if x_col is None:
            raise ValueError("Histogram requires x_col")
        fig = px.histogram(
            data, 
            x=x_col, 
            color=color_col,
            title=title,
            text_auto=True,  # Add count labels
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        # Histogram counts are always integers
        fig.update_traces(
            texttemplate='%{y:.0f}',
            textposition='outside'
        )
        
    elif chart_name.lower() == 'heatmap':
        data = pd.crosstab(data[x_col], data[y_col])

        # Create a high-resolution heatmap visualization with larger fonts
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='Blues',
            colorbar=dict(
                title=dict(text='Count', font=dict(size=24)),  # ✅ fixed
                tickfont=dict(size=20)                         # ✅ fixed
            ),
            text=data.values,
            texttemplate="%{text}",
            textfont={"size": 24}  # Increase heatmap cell text size
        ))





    # Apply common styling to all charts with updated fonts and styling for white theme
    fig.update_layout(
        title_font_size=28,
        title_font_family=title_font,
        title_font_color="#222222",  # Slightly darker for better contrast on white
        title_x=0.5,
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'xanchor': 'center'
        },
        legend_title_font_size=14,
        legend_title_font_family=text_font,
        font_family=text_font,
        width=900,
        height=700,
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            title_font=dict(family=text_font, size=24),
            tickfont=dict(family=text_font, size=24),
            gridcolor='rgba(220,220,220,0.4)'  # Lighter grid lines
        ),
        yaxis=dict(
            title_font=dict(family=text_font, size=24),
            tickfont=dict(family=text_font, size=24),
            gridcolor='rgba(220,220,220,0.4)'  # Lighter grid lines
        ),
        # Add subtle box around plot area
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        legend=dict(
            bordercolor='rgba(220,220,220,0.4)',
            borderwidth=1
        )
    )
    
    # # Show the figure to ensure it displays in Jupyter notebooks
    # # Use alternative display method if the default one fails
    # try:
    #     fig.show()
    # except Exception as e:
    #     print(f"Warning: Could not display figure with default method: {e}")
    #     try:
    #         # Try alternative display method
    #         from IPython.display import display, HTML
    #         display(HTML(fig.to_html()))
    #     except Exception as e2:
    #         print(f"Warning: Could not display figure with HTML method: {e2}")
    #         print("Figure created successfully but display may not work in this environment.")
    
    return fig




# def plot(data, chart_name, x_col=None, y_col=None, color_col=None, title=None, export=True, filename="plot.png"):
#     import plotly.express as px
#     import plotly.graph_objects as go
#     from plotly.io import templates
#     import plotly.io as pio
#     import pandas as pd
#     import numpy as np
#     import random

#     # Configure Plotly renderer
#     try:
#         pio.renderers.default = "notebook_connected"
#     except:
#         pio.renderers.default = "browser"

#     # High-DPI scaling
#     pio.templates.default = "plotly_white"
#     pio.kaleido.scope.default_format = "png"
#     pio.kaleido.scope.default_width = 1600
#     pio.kaleido.scope.default_height = 1000
#     pio.kaleido.scope.default_scale = 6  # Simulate high DPI (300+)

#     # Color setup
#     qualitative_colors = [
#         '#00549F', '#00A1DE', '#FBB513', '#BB29BB', '#19D3F3',
#         '#FF6692', '#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD',
#         '#E377C2', '#17BECF', '#FF97FF', '#7F7F7F', '#BCBD22'
#     ]
#     random.shuffle(qualitative_colors)
#     color_scales = {
#         'qualitative': qualitative_colors,
#         'sequential': px.colors.sequential.Viridis,
#         'diverging': px.colors.diverging.RdBu
#     }

#     title = title or f"{chart_name.capitalize()} Chart"
#     title_font = "Raleway, sans-serif"
#     text_font = "Open Sans, sans-serif"

#     def get_text_template(column):
#         if column is None:
#             return '%{text}'
#         if column in data.columns:
#             if pd.api.types.is_integer_dtype(data[column]) or data[column].apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all():
#                 return '%{text:.0f}'
#             else:
#                 return '%{text:.2f}'
#         return '%{text}'

#     fig = None

#     # === BAR CHART ===
#     if chart_name.lower() == 'bar':
#         if x_col is None or y_col is None:
#             raise ValueError("Bar chart requires x_col and y_col")
#         text_template = get_text_template(y_col)
#         fig = px.bar(
#             data, x=x_col, y=y_col,
#             color=color_col or x_col,
#             title=title, text=y_col,
#             color_discrete_sequence=color_scales['qualitative']
#         )
#         fig.update_traces(
#             texttemplate=text_template,
#             textfont=dict(size=20, family=text_font, color="black"),
#             textposition='outside',
#             cliponaxis=False,
#             marker_line_width=1.8,
#             opacity=0.9
#         )

#     elif chart_name.lower() == 'scatter':
#         fig = px.scatter(
#             data, x=x_col, y=y_col,
#             color=color_col, title=title,
#             color_discrete_sequence=color_scales['qualitative']
#         )
#         fig.update_traces(marker=dict(size=12, line=dict(width=2, color='white'), opacity=0.85))

#     elif chart_name.lower() == 'line':
#         text_template = get_text_template(y_col)
#         fig = px.line(
#             data, x=x_col, y=y_col, color=color_col,
#             title=title, text=y_col,
#             color_discrete_sequence=color_scales['qualitative']
#         )
#         fig.update_traces(
#             mode='lines+markers+text',
#             texttemplate=text_template,
#             textfont=dict(size=20, family=text_font),
#             textposition='top center',
#             marker=dict(size=10, line=dict(width=2, color='white')),
#             line=dict(width=3.8)
#         )

#     elif chart_name.lower() == 'pie':
#         is_integer = pd.api.types.is_integer_dtype(data[y_col]) if y_col in data.columns else False
#         fig = px.pie(
#             data, names=x_col, values=y_col, title=title,
#             color_discrete_sequence=color_scales['qualitative'], hole=0.4
#         )
#         fmt = '%{percent:.1%}<br>%{value:.0f}' if is_integer else '%{percent:.1%}<br>%{value:.2f}'
#         fig.update_traces(
#             texttemplate=fmt,
#             textfont=dict(size=20, family=text_font),
#             textinfo='label+percent+value',
#             textposition='inside'
#         )

#     elif chart_name.lower() == 'box':
#         fig = px.box(
#             data, x=x_col, y=y_col, color=color_col,
#             title=title, points='all',
#             color_discrete_sequence=color_scales['qualitative']
#         )

#     elif chart_name.lower() == 'histogram':
#         fig = px.histogram(
#             data, x=x_col, color=color_col,
#             title=title, text_auto=True,
#             color_discrete_sequence=color_scales['qualitative']
#         )
#         fig.update_traces(
#             texttemplate='%{y:.0f}',
#             textfont=dict(size=20, family=text_font),
#             textposition='outside'
#         )

#     elif chart_name.lower() == 'heatmap':
#         data = pd.crosstab(data[x_col], data[y_col])
#         fig = go.Figure(data=go.Heatmap(
#             z=data.values, x=data.columns, y=data.index,
#             colorscale='Viridis',
#             colorbar=dict(title='Count', titlefont=dict(size=16)),
#             text=data.values, texttemplate="%{text}",
#             textfont={"size":18}
#         ))
#         fig.update_layout(
#             title=title, xaxis=dict(title=x_col), yaxis=dict(title=y_col),
#             width=1600, height=1000, margin=dict(l=100, r=50, t=80, b=60)
#         )

#     # === UNIVERSAL LAYOUT ENHANCEMENTS ===
#     fig.update_layout(
#         title=dict(
#             text=f"<b>{title}</b>", x=0.5, y=0.96,
#             font=dict(size=36, family=title_font, color="#111111")
#         ),

#         # 🔹 Legend: now size 24
#         legend=dict(
#             title_font=dict(size=24, family=text_font),  # CHANGED 22 → 24
#             font=dict(size=24, family=text_font),        # CHANGED 20 → 24
#             bordercolor='rgba(200,200,200,0.5)', borderwidth=1
#         ),

#         # 🔹 Axis fonts: now all 24
#         xaxis=dict(
#             title_font=dict(size=24),     # CHANGED 24 → stays 24
#             tickfont=dict(size=24),       # CHANGED 20 → 24
#             gridcolor='rgba(210,210,210,0.4)'
#         ),
#         yaxis=dict(
#             title_font=dict(size=24),     # CHANGED 24 → stays 24
#             tickfont=dict(size=24),       # CHANGED 20 → 24
#             gridcolor='rgba(210,210,210,0.4)'
#         ),

#         font=dict(family=text_font, size=20, color="#222222"),
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         width=1600,
#         height=1000,
#         margin=dict(l=80, r=80, t=100, b=80)
#     )

#     # Export high-DPI image if requested
#     if export:
#         fig.write_image(filename, scale=3)
#         print(f"✅ High-resolution image saved as: {filename}")

#     return fig


def plot_figure_only(data, chart_name, x_col=None, y_col=None, color_col=None, title=None):
    """
    Same as plot() function but returns figure without attempting to display it.
    Use this if you want to handle the display yourself or avoid display issues.
    
    Example usage:
        fig = plot_figure_only(data, 'bar', x_col='category', y_col='values')
        fig.show()  # or display(fig) or fig.write_html('chart.html')
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import templates
    import plotly.io as pio
    import pandas as pd
    import numpy as np
    import random
    
    # Configure Plotly for notebook display
    try:
        pio.renderers.default = "notebook"
    except:
        try:
            pio.renderers.default = "plotly_mimetype+notebook"
        except:
            pass  # Use default renderer
    
    # Define visually attractive color palettes optimized for plotly_white
    qualitative_colors = [
        # Modern, vibrant colors that stand out on white background
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        # Additional attractive colors
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
    ]
    
    # Always shuffle the qualitative colors
    random.shuffle(qualitative_colors)
    
    # Alternative attractive sequential and diverging color schemes for plotly_white
    color_scales = {
        'qualitative': qualitative_colors,
        'sequential': px.colors.sequential.Viridis,  # Better contrast than Plasma
        'diverging': px.colors.diverging.RdBu      # Clean, professional red-blue scale
    }
    
    # Set template
    template = templates.default = "plotly_white"
    
    # Set default title if none provided
    if title is None:
        title = f"{chart_name.capitalize()} Chart"
    
    # Font styles
    title_font = "Raleway, sans-serif"  # Modern sans-serif font for titles
    text_font = "Open Sans, sans-serif"  # Clean font for axis labels and text
    
    # Determine if the data is integer or float (if applicable)
    def get_text_template(column):
        if column is None:
            return '%{text}'
        
        # Check if column exists in dataframe
        if column in data.columns:
            # Check data type
            if pd.api.types.is_integer_dtype(data[column]) or data[column].apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all():
                return '%{text:.0f}'  # Integer format
            else:
                return '%{text:.2f}'  # Float format with 2 decimal places
        return '%{text}'  # Default
    
    # Create figure based on chart type
    fig = None
    
    # Handle different chart types
    if chart_name.lower() == 'bar':
        if x_col is None or y_col is None:
            raise ValueError("Bar chart requires x_col and y_col")
        
        # Determine text template based on y_col data type
        text_template = get_text_template(y_col)
        
        fig = px.bar(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col or x_col,
            title=title,
            text=y_col,
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        # Configure text position
        fig.update_traces(
            texttemplate=text_template,
            textposition='outside',
            cliponaxis=False,
            marker_line_width=1,  # Add subtle border for better definition
            opacity=0.85,  # Slightly transparent for a softer look
            textfont=dict(size=22, family="Open Sans, sans-serif", color="black")
        )
        
    elif chart_name.lower() == 'scatter':
        if x_col is None or y_col is None:
            raise ValueError("Scatter plot requires x_col and y_col")
        
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            title=title,
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        
        # Enhanced scatter plot styling
        fig.update_traces(
            marker=dict(
                size=8,
                line=dict(width=1, color='white'),  # White border for better definition
                opacity=0.8  # Slightly transparent for overlapping points
            )
        )
        
        # Add trend line if no color grouping
        if color_col is None:
            # Calculate trend line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_col], data[y_col])
            line_x = [data[x_col].min(), data[x_col].max()]
            line_y = [slope * x + intercept for x in line_x]
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f'Trend (R² = {r_value**2:.3f})',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            ))
        
    # Other chart types remain the same but use color_scales instead of dark_colors
    elif chart_name.lower() == 'line':
        if x_col is None or y_col is None:
            raise ValueError("Line chart requires x_col and y_col")
            
        text_template = get_text_template(y_col)
        
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col, 
            title=title,
            text=y_col,
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        fig.update_traces(
            mode='lines+markers+text',
            texttemplate=text_template,
            textposition='top center',
            marker=dict(size=8, line=dict(width=1, color='white')),
            line=dict(width=3)  # Thicker lines for better visibility
        )
            
    elif chart_name.lower() == 'pie':
        if x_col is None or y_col is None:
            raise ValueError("Pie chart requires names column (x_col) and values column (y_col)")
        
        # For pie charts, determine if values are integers or floats
        is_integer = False
        try:
            if pd.api.types.is_integer_dtype(data[y_col]) or data[y_col].apply(lambda x: float(x).is_integer() if pd.notnull(x) else True).all():
                is_integer = True
        except:
            pass
        
        fig = px.pie(
            data, 
            names=x_col, 
            values=y_col,
            title=title,
            color_discrete_sequence=color_scales['qualitative'],
            template=template,
            hole=0.4
        )
        
        # Format text based on data type
        if is_integer:
            fig.update_traces(texttemplate='%{percent:.1%}<br>%{value:.0f}', textinfo='label+percent+value')
        else:
            fig.update_traces(texttemplate='%{percent:.1%}<br>%{value:.2f}', textinfo='label+percent+value')
        
        fig.update_traces(textposition='inside')
    
    elif chart_name.lower() == 'box':
        if x_col is None or y_col is None:
            raise ValueError("Box plot requires x_col and y_col")
        fig = px.box(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            title=title,
            points='all',  # Show all points for better data visibility
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        # Box plots don't support direct data labels, but we show all points
        
    elif chart_name.lower() == 'histogram':
        if x_col is None:
            raise ValueError("Histogram requires x_col")
        fig = px.histogram(
            data, 
            x=x_col, 
            color=color_col,
            title=title,
            text_auto=True,  # Add count labels
            color_discrete_sequence=color_scales['qualitative'],
            template=template
        )
        # Histogram counts are always integers
        fig.update_traces(
            texttemplate='%{y:.0f}',
            textposition='outside'
        )

    elif chart_name.lower() == 'heatmap':
        import plotly.graph_objects as go
        data = pd.crosstab(data[x_col], data[y_col])

        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='Viridis',

            # ✅ Correct syntax for Plotly ≥5 (NO 'titlefont')
            colorbar=dict(
                title=dict(
                    text='Count',
                    font=dict(size=24, family='Open Sans, sans-serif', color='black')
                ),
                tickfont=dict(size=24, family='Open Sans, sans-serif', color='black'),
                thickness=25,
                outlinewidth=1
            ),

            text=data.values,
            texttemplate="%{text}",
            textfont={"size":18, "color": "black"}
        ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=36, family='Raleway, sans-serif', color='black'),
                x=0.5
            ),
            xaxis=dict(
                title=x_col,
                title_font=dict(size=24, family='Open Sans, sans-serif', color='black'),
                tickfont=dict(size=24, family='Open Sans, sans-serif', color='black')
            ),
            yaxis=dict(
                title=y_col,
                title_font=dict(size=24, family='Open Sans, sans-serif', color='black'),
                tickfont=dict(size=24, family='Open Sans, sans-serif', color='black')
            ),
            width=1600,
            height=1000,
            margin=dict(l=100, r=50, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )



    # Apply common styling to all charts with updated fonts and styling for white theme
    fig.update_layout(
        title_font_size=28,
        title_font_family=title_font,
        title_font_color="#222222",  # Slightly darker for better contrast on white
        title_x=0.5,
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'xanchor': 'center'
        },
        legend_title_font_size=14,
        legend_title_font_family=text_font,
        font_family=text_font,
        height=600,
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            title_font=dict(family=text_font, size=14),
            tickfont=dict(family=text_font, size=12),
            gridcolor='rgba(220,220,220,0.4)'  # Lighter grid lines
        ),
        yaxis=dict(
            title_font=dict(family=text_font, size=14),
            tickfont=dict(family=text_font, size=12),
            gridcolor='rgba(220,220,220,0.4)'  # Lighter grid lines
        ),
        # Add subtle box around plot area
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        legend=dict(
            bordercolor='rgba(220,220,220,0.4)',
            borderwidth=1
        )
    )
    
    return fig




# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

# Assuming your dataframe is named 'df'
# df = pd.read_your_data()

class VisEvalAnalyzer:
    def __init__(self, df):
        self.df = df
        self.stop_words = set(stopwords.words('english'))
    
    # ============================================================================
    # 1. DATASET OVERVIEW & COMPOSITION ANALYSIS
    # ============================================================================
    
    def get_chart_type_distribution(self):
        """1.1 Distribution of Visualization Types"""
        chart_dist = self.df['chart'].value_counts().reset_index()
        chart_dist.columns = ['chart_type', 'count']
        chart_dist['percentage'] = (chart_dist['count'] / len(self.df) * 100).round(2)
        return chart_dist
    
    def get_hardness_distribution(self):
        """1.2 Query Complexity Distribution"""
        # Overall hardness distribution
        hardness_dist = self.df['hardness'].value_counts().reset_index()
        hardness_dist.columns = ['hardness', 'count']
        
        # Hardness by chart type
        hardness_by_chart = pd.crosstab(self.df['chart'], self.df['hardness']).reset_index()
        
        return {
            'overall': hardness_dist,
            'by_chart': hardness_by_chart
        }
    
    def get_database_coverage(self):
        """1.3 Database Coverage Analysis"""
        # Queries per database
        db_coverage = self.df['db_id'].value_counts().reset_index()
        db_coverage.columns = ['db_id', 'query_count']
        
        # Database vs Chart Type matrix
        db_chart_matrix = pd.crosstab(self.df['db_id'], self.df['chart']).reset_index()
        
        return {
            'coverage': db_coverage,
            'matrix': db_chart_matrix
        }
    
    # ============================================================================
    # 2. NATURAL LANGUAGE QUERY ANALYSIS
    # ============================================================================
    
    def get_nl_query_diversity(self):
        """2.1 Query Linguistic Diversity"""
        nl_stats = []
        
        for idx, row in self.df.iterrows():
            nl_queries = ast.literal_eval(row['nl_queries']) if isinstance(row['nl_queries'], str) else row['nl_queries']
            
            nl_stats.append({
                'index': idx,
                'chart_type': row['chart'],
                'hardness': row['hardness'],
                'num_variants': len(nl_queries),
                'avg_length': np.mean([len(q.split()) for q in nl_queries]),
                'total_words': sum([len(q.split()) for q in nl_queries]),
                'unique_words': len(set(' '.join(nl_queries).lower().split()))
            })
        
        nl_diversity_df = pd.DataFrame(nl_stats)
        
        # Word frequency analysis
        all_queries = []
        for idx, row in self.df.iterrows():
            nl_queries = ast.literal_eval(row['nl_queries']) if isinstance(row['nl_queries'], str) else row['nl_queries']
            all_queries.extend(nl_queries)
        
        # Clean and tokenize
        words = []
        for query in all_queries:
            tokens = word_tokenize(query.lower())
            words.extend([w for w in tokens if w.isalpha() and w not in self.stop_words])
        
        word_freq = pd.DataFrame(Counter(words).most_common(50), columns=['word', 'frequency'])
        
        return {
            'diversity_stats': nl_diversity_df,
            'word_frequency': word_freq
        }
    
    def get_query_similarity_analysis(self):
        """2.2 Query Similarity Analysis"""
        similarity_data = []
        
        for idx, row in self.df.iterrows():
            nl_queries = ast.literal_eval(row['nl_queries']) if isinstance(row['nl_queries'], str) else row['nl_queries']
            
            if len(nl_queries) > 1:
                # Calculate pairwise similarity
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(nl_queries)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Get average similarity (excluding diagonal)
                upper_triangle = np.triu(similarity_matrix, k=1)
                avg_similarity = np.mean(upper_triangle[upper_triangle > 0])
                
                similarity_data.append({
                    'index': idx,
                    'chart_type': row['chart'],
                    'hardness': row['hardness'],
                    'num_queries': len(nl_queries),
                    'avg_similarity': avg_similarity,
                    'max_similarity': np.max(upper_triangle),
                    'min_similarity': np.min(upper_triangle[upper_triangle > 0]) if len(upper_triangle[upper_triangle > 0]) > 0 else 0
                })
        
        return pd.DataFrame(similarity_data)
    
    def get_command_patterns(self):
        """2.3 Command Pattern Analysis"""
        command_patterns = []
        chart_mentions = []
        
        for idx, row in self.df.iterrows():
            nl_queries = ast.literal_eval(row['nl_queries']) if isinstance(row['nl_queries'], str) else row['nl_queries']
            
            for query in nl_queries:
                query_lower = query.lower()
                
                # Extract command patterns
                commands = []
                if 'show' in query_lower: commands.append('show')
                if 'visualize' in query_lower or 'visualise' in query_lower: commands.append('visualize')
                if 'create' in query_lower: commands.append('create')
                if 'generate' in query_lower: commands.append('generate')
                if 'display' in query_lower: commands.append('display')
                if 'plot' in query_lower: commands.append('plot')
                if 'draw' in query_lower: commands.append('draw')
                
                command_patterns.extend(commands)
                
                # Check for chart type mentions
                chart_types = ['pie', 'bar', 'line', 'scatter', 'histogram', 'box']
                for chart_type in chart_types:
                    if chart_type in query_lower:
                        chart_mentions.append(chart_type)
        
        command_df = pd.DataFrame(Counter(command_patterns).most_common(), columns=['command', 'frequency'])
        chart_mention_df = pd.DataFrame(Counter(chart_mentions).most_common(), columns=['chart_mention', 'frequency'])
        
        return {
            'commands': command_df,
            'chart_mentions': chart_mention_df
        }
    
    # ============================================================================
    # 3. SQL QUERY COMPLEXITY ANALYSIS
    # ============================================================================
    
    def get_sql_complexity(self):
        """3.1 SQL Operation Complexity"""
        sql_stats = []
        
        for idx, row in self.df.iterrows():
            vis_query = ast.literal_eval(row['vis_query']) if isinstance(row['vis_query'], str) else row['vis_query']
            sql_query = vis_query.get('data_part', {}).get('sql_part', '') if isinstance(vis_query, dict) else ''
            
            if sql_query:
                sql_upper = sql_query.upper()
                
                operations = {
                    'SELECT': sql_upper.count('SELECT'),
                    'FROM': sql_upper.count('FROM'),
                    'WHERE': sql_upper.count('WHERE'),
                    'GROUP BY': sql_upper.count('GROUP BY'),
                    'ORDER BY': sql_upper.count('ORDER BY'),
                    'HAVING': sql_upper.count('HAVING'),
                    'JOIN': sql_upper.count('JOIN'),
                    'INNER JOIN': sql_upper.count('INNER JOIN'),
                    'LEFT JOIN': sql_upper.count('LEFT JOIN'),
                    'RIGHT JOIN': sql_upper.count('RIGHT JOIN'),
                    'UNION': sql_upper.count('UNION'),
                    'DISTINCT': sql_upper.count('DISTINCT')
                }
                
                # Count aggregation functions
                agg_functions = {
                    'COUNT': sql_upper.count('COUNT('),
                    'SUM': sql_upper.count('SUM('),
                    'AVG': sql_upper.count('AVG('),
                    'MAX': sql_upper.count('MAX('),
                    'MIN': sql_upper.count('MIN(')
                }
                
                # Calculate complexity score
                complexity_score = (
                    operations['JOIN'] * 2 +
                    operations['GROUP BY'] * 1.5 +
                    operations['WHERE'] * 1 +
                    sum(agg_functions.values()) * 1.5 +
                    operations['HAVING'] * 2
                )
                
                sql_stats.append({
                    'index': idx,
                    'chart_type': row['chart'],
                    'hardness': row['hardness'],
                    'complexity_score': complexity_score,
                    'num_joins': operations['JOIN'],
                    'has_group_by': operations['GROUP BY'] > 0,
                    'has_where': operations['WHERE'] > 0,
                    'num_aggregations': sum(agg_functions.values()),
                    **operations,
                    **{f'agg_{k.lower()}': v for k, v in agg_functions.items()}
                })
        
        sql_complexity_df = pd.DataFrame(sql_stats)
        
        # Operation frequency
        operations_freq = []
        for op in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'JOIN']:
            operations_freq.append({
                'operation': op,
                'frequency': (sql_complexity_df[op] > 0).sum(),
                'avg_count': sql_complexity_df[op].mean()
            })
        
        operations_df = pd.DataFrame(operations_freq)
        
        return {
            'complexity_stats': sql_complexity_df,
            'operations_frequency': operations_df
        }
    
    def get_table_analysis(self):
        """3.2 Table Relationship Analysis"""
        table_usage = defaultdict(int)
        table_pairs = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            vis_query = ast.literal_eval(row['vis_query']) if isinstance(row['vis_query'], str) else row['vis_query']
            sql_query = vis_query.get('data_part', {}).get('sql_part', '') if isinstance(vis_query, dict) else ''
            
            if sql_query:
                # Extract table names (simplified - assumes standard SQL format)
                tables = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                tables.extend(re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE))
                
                for table in tables:
                    table_usage[table] += 1
                
                # Table co-occurrence
                for i in range(len(tables)):
                    for j in range(i+1, len(tables)):
                        pair = tuple(sorted([tables[i], tables[j]]))
                        table_pairs[pair] += 1
        
        table_usage_df = pd.DataFrame(list(table_usage.items()), columns=['table_name', 'usage_count'])
        table_usage_df = table_usage_df.sort_values('usage_count', ascending=False)
        
        table_pairs_df = pd.DataFrame([
            {'table1': pair[0], 'table2': pair[1], 'co_occurrence': count}
            for pair, count in table_pairs.items()
        ]).sort_values('co_occurrence', ascending=False)
        
        return {
            'table_usage': table_usage_df,
            'table_pairs': table_pairs_df
        }
    
    # ============================================================================
    # 4. VISUALIZATION DESIGN ANALYSIS
    # ============================================================================
    
    def get_channel_analysis(self):
        """4.1 Channel Specification Patterns"""
        channel_data = []
        
        for idx, row in self.df.iterrows():
            vis_obj = ast.literal_eval(row['vis_obj']) if isinstance(row['vis_obj'], str) else row['vis_obj']
            
            if isinstance(vis_obj, dict):
                x_name = vis_obj.get('x_name', '')
                y_name = vis_obj.get('y_name', '')
                x_data = vis_obj.get('x_data', [])
                y_data = vis_obj.get('y_data', [])
                
                # Determine data types
                x_type = 'categorical' if x_data and isinstance(x_data[0], list) and isinstance(x_data[0][0], str) else 'numerical'
                y_type = 'categorical' if y_data and isinstance(y_data[0], list) and isinstance(y_data[0][0], str) else 'numerical'
                
                # Count data points
                x_cardinality = len(x_data[0]) if x_data and len(x_data) > 0 else 0
                y_cardinality = len(y_data[0]) if y_data and len(y_data) > 0 else 0
                
                channel_data.append({
                    'index': idx,
                    'chart_type': row['chart'],
                    'hardness': row['hardness'],
                    'x_name': x_name,
                    'y_name': y_name,
                    'x_type': x_type,
                    'y_type': y_type,
                    'x_cardinality': x_cardinality,
                    'y_cardinality': y_cardinality
                })
        
        return pd.DataFrame(channel_data)
    
    def get_data_cardinality_analysis(self):
        """4.2 Data Cardinality Analysis"""
        cardinality_stats = []
        
        for idx, row in self.df.iterrows():
            vis_obj = ast.literal_eval(row['vis_obj']) if isinstance(row['vis_obj'], str) else row['vis_obj']
            
            if isinstance(vis_obj, dict):
                x_data = vis_obj.get('x_data', [])
                y_data = vis_obj.get('y_data', [])
                
                x_points = len(x_data[0]) if x_data and len(x_data) > 0 else 0
                y_points = len(y_data[0]) if y_data and len(y_data) > 0 else 0
                total_points = max(x_points, y_points)
                
                cardinality_stats.append({
                    'index': idx,
                    'chart_type': row['chart'],
                    'hardness': row['hardness'],
                    'total_data_points': total_points,
                    'x_cardinality': x_points,
                    'y_cardinality': y_points
                })
        
        return pd.DataFrame(cardinality_stats)
    
    # ============================================================================
    # 5. DATASET QUALITY & COVERAGE ANALYSIS
    # ============================================================================
    
    def get_irrelevant_table_analysis(self):
        """5.1 Irrelevant Table Analysis"""
        irrelevant_stats = []
        
        for idx, row in self.df.iterrows():
            irrelevant_tables = ast.literal_eval(row['irrelevant_tables']) if isinstance(row['irrelevant_tables'], str) else row['irrelevant_tables']
            
            irrelevant_stats.append({
                'index': idx,
                'chart_type': row['chart'],
                'hardness': row['hardness'],
                'db_id': row['db_id'],
                'num_irrelevant_tables': len(irrelevant_tables) if irrelevant_tables else 0,
                'has_irrelevant_tables': len(irrelevant_tables) > 0 if irrelevant_tables else False
            })
        
        return pd.DataFrame(irrelevant_stats)
    
    def get_query_completeness_analysis(self):
        """5.2 Query Completeness Analysis"""
        completeness_stats = []
        
        for idx, row in self.df.iterrows():
            vis_query = ast.literal_eval(row['vis_query']) if isinstance(row['vis_query'], str) else row['vis_query']
            vis_obj = ast.literal_eval(row['vis_obj']) if isinstance(row['vis_obj'], str) else row['vis_obj']
            nl_queries = ast.literal_eval(row['nl_queries']) if isinstance(row['nl_queries'], str) else row['nl_queries']
            
            # Check completeness
            has_sql = bool(vis_query.get('data_part', {}).get('sql_part', '') if isinstance(vis_query, dict) else False)
            has_vis_spec = bool(vis_obj.get('x_name', '') and vis_obj.get('y_name', '') if isinstance(vis_obj, dict) else False)
            has_nl_queries = bool(nl_queries and len(nl_queries) > 0)
            
            completeness_stats.append({
                'index': idx,
                'chart_type': row['chart'],
                'hardness': row['hardness'],
                'has_sql': has_sql,
                'has_vis_spec': has_vis_spec,
                'has_nl_queries': has_nl_queries,
                'is_complete': has_sql and has_vis_spec and has_nl_queries,
                'num_nl_variants': len(nl_queries) if nl_queries else 0
            })
        
        return pd.DataFrame(completeness_stats)
    
    # ============================================================================
    # 6. CROSS-DIMENSIONAL ANALYSIS
    # ============================================================================
    
    def get_hardness_complexity_correlation(self):
        """6.1 Hardness vs Complexity Correlation"""
        sql_complexity = self.get_sql_complexity()['complexity_stats']
        nl_diversity = self.get_nl_query_diversity()['diversity_stats']
        
        # Merge data
        correlation_data = sql_complexity.merge(
            nl_diversity[['index', 'avg_length', 'num_variants', 'unique_words']], 
            on='index'
        )
        
        # Encode hardness as numeric
        hardness_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        correlation_data['hardness_numeric'] = correlation_data['hardness'].map(hardness_map)
        
        return correlation_data
    
    def get_chart_effectiveness_analysis(self):
        """6.2 Chart Type Effectiveness"""
        effectiveness_data = []
        
        # Group by chart type and calculate various metrics
        for chart_type in self.df['chart'].unique():
            chart_data = self.df[self.df['chart'] == chart_type]
            
            # Calculate complexity metrics for this chart type
            sql_complexity = self.get_sql_complexity()['complexity_stats']
            chart_sql = sql_complexity[sql_complexity['chart_type'] == chart_type]
            
            effectiveness_data.append({
                'chart_type': chart_type,
                'total_queries': len(chart_data),
                'avg_complexity': chart_sql['complexity_score'].mean() if not chart_sql.empty else 0,
                'hard_queries_pct': (chart_data['hardness'] == 'Hard').mean() * 100,
                'medium_queries_pct': (chart_data['hardness'] == 'Medium').mean() * 100,
                'easy_queries_pct': (chart_data['hardness'] == 'Easy').mean() * 100,
                'avg_joins': chart_sql['num_joins'].mean() if not chart_sql.empty else 0,
                'avg_aggregations': chart_sql['num_aggregations'].mean() if not chart_sql.empty else 0
            })
        
        return pd.DataFrame(effectiveness_data)
    
    # ============================================================================
    # 7. COMPREHENSIVE ANALYSIS RUNNER
    # ============================================================================
    
    def run_all_analyses(self):
        """Run all analyses and return organized results"""
        print("Running comprehensive VisEval dataset analysis...")
        
        results = {
            'dataset_overview': {
                'chart_distribution': self.get_chart_type_distribution(),
                'hardness_distribution': self.get_hardness_distribution(),
                'database_coverage': self.get_database_coverage()
            },
            'nl_analysis': {
                'query_diversity': self.get_nl_query_diversity(),
                'query_similarity': self.get_query_similarity_analysis(),
                'command_patterns': self.get_command_patterns()
            },
            'sql_analysis': {
                'complexity': self.get_sql_complexity(),
                'table_analysis': self.get_table_analysis()
            },
            'visualization_design': {
                'channel_analysis': self.get_channel_analysis(),
                'cardinality_analysis': self.get_data_cardinality_analysis()
            },
            'quality_analysis': {
                'irrelevant_tables': self.get_irrelevant_table_analysis(),
                'completeness': self.get_query_completeness_analysis()
            },
            'cross_dimensional': {
                'hardness_correlation': self.get_hardness_complexity_correlation(),
                'chart_effectiveness': self.get_chart_effectiveness_analysis()
            }
        }
        
        print("Analysis complete!")
        return results

# ============================================================================
# VISUALIZATION-READY DATAFRAME GENERATORS
# ============================================================================

class VisualizationDataFrames:
    """Generate specific dataframes for different types of visualizations"""
    
    def __init__(self, analyzer_results):
        self.results = analyzer_results
    
    def get_pie_chart_data(self):
        """Data for pie charts"""
        return {
            'chart_type_distribution': self.results['dataset_overview']['chart_distribution'],
            'hardness_distribution': self.results['dataset_overview']['hardness_distribution']['overall'],
            'command_patterns': self.results['nl_analysis']['command_patterns']['commands']
        }
    
    def get_bar_chart_data(self):
        """Data for bar charts"""
        return {
            'database_coverage': self.results['dataset_overview']['database_coverage']['coverage'],
            'table_usage': self.results['sql_analysis']['table_analysis']['table_usage'],
            'word_frequency': self.results['nl_analysis']['query_diversity']['word_frequency'],
            'sql_operations': self.results['sql_analysis']['complexity']['operations_frequency']
        }
    
    def get_heatmap_data(self):
        """Data for heatmaps"""
        return {
            'hardness_by_chart': self.results['dataset_overview']['hardness_distribution']['by_chart'],
            'db_chart_matrix': self.results['dataset_overview']['database_coverage']['matrix']
        }
    
    def get_scatter_plot_data(self):
        """Data for scatter plots"""
        return {
            'hardness_vs_complexity': self.results['cross_dimensional']['hardness_correlation'],
            'cardinality_analysis': self.results['visualization_design']['cardinality_analysis']
        }
    
    def get_box_plot_data(self):
        """Data for box plots"""
        complexity_stats = self.results['sql_analysis']['complexity']['complexity_stats']
        diversity_stats = self.results['nl_analysis']['query_diversity']['diversity_stats']
        
        return {
            'complexity_by_hardness': complexity_stats[['hardness', 'complexity_score', 'num_joins', 'num_aggregations']],
            'query_length_by_chart': diversity_stats[['chart_type', 'avg_length', 'num_variants']],
            'irrelevant_tables': self.results['quality_analysis']['irrelevant_tables']
        }
    
    def get_correlation_matrix_data(self):
        """Data for correlation analysis"""
        correlation_data = self.results['cross_dimensional']['hardness_correlation']
        
        # Select numeric columns for correlation
        numeric_cols = ['hardness_numeric', 'complexity_score', 'num_joins', 'num_aggregations', 
                       'avg_length', 'num_variants', 'unique_words']
        
        return correlation_data[numeric_cols]

# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================

def create_plotting_dataframes(df):
    """
    Master function to create all plotting dataframes
    
    Usage:
    plotting_dfs = create_plotting_dataframes(your_dataframe)
    """
    
    # Initialize analyzer
    analyzer = VisEvalAnalyzer(df)
    
    # Run all analyses
    results = analyzer.run_all_analyses()
    
    # Create visualization dataframes
    viz_dfs = VisualizationDataFrames(results)
    
    # Organize all plotting dataframes
    plotting_dataframes = {
        
        # PIE CHARTS
        'pie_charts': {
            'chart_type_dist': viz_dfs.get_pie_chart_data()['chart_type_distribution'],
            'hardness_dist': viz_dfs.get_pie_chart_data()['hardness_distribution'],
            'command_patterns': viz_dfs.get_pie_chart_data()['command_patterns']
        },
        
        # BAR CHARTS
        'bar_charts': {
            'database_coverage': viz_dfs.get_bar_chart_data()['database_coverage'],
            'table_usage': viz_dfs.get_bar_chart_data()['table_usage'],
            'word_frequency': viz_dfs.get_bar_chart_data()['word_frequency'],
            'sql_operations': viz_dfs.get_bar_chart_data()['sql_operations'],
            'chart_effectiveness': results['cross_dimensional']['chart_effectiveness']
        },
        
        # HEATMAPS
        'heatmaps': {
            'hardness_by_chart': viz_dfs.get_heatmap_data()['hardness_by_chart'],
            'db_chart_matrix': viz_dfs.get_heatmap_data()['db_chart_matrix']
        },
        
        # SCATTER PLOTS
        'scatter_plots': {
            'hardness_vs_complexity': viz_dfs.get_scatter_plot_data()['hardness_vs_complexity'],
            'cardinality_analysis': viz_dfs.get_scatter_plot_data()['cardinality_analysis']
        },
        
        # BOX PLOTS
        'box_plots': viz_dfs.get_box_plot_data(),
        
        # CORRELATION ANALYSIS
        'correlation_matrix': viz_dfs.get_correlation_matrix_data(),
        
        # DETAILED STATS
        'detailed_stats': {
            'sql_complexity': results['sql_analysis']['complexity']['complexity_stats'],
            'nl_diversity': results['nl_analysis']['query_diversity']['diversity_stats'],
            'query_similarity': results['nl_analysis']['query_similarity'],
            'channel_analysis': results['visualization_design']['channel_analysis'],
            'completeness_analysis': results['quality_analysis']['completeness']
        }
    }
    
    return plotting_dataframes


# def error_rate_visualization(df):
#     import plotly.express as px
#     import random

#     # --- Consistent Color Palette ---
#     hex_colors = [
#         "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
#         "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
#     ]
#     random.shuffle(hex_colors)

#     # --- Color Map for Agents ---
#     unique_agents = df["agent"].unique()
#     color_map = {agent: hex_colors[i % len(hex_colors)] for i, agent in enumerate(unique_agents)}

#     # --- Fonts ---
#     title_font = "Raleway, sans-serif"
#     text_font = "Open Sans, sans-serif"

#     # --- Melted Data ---
#     melted_df = df.melt(
#         id_vars=["agent", "model", "library"],
#         value_vars=["invalid_rate", "illegal_rate"]
#     )

#     # --- Create Non-Stacked Area Chart (Vertical Facets with Spacing) ---
#     fig = px.area(
#         melted_df,
#         x="model",
#         y="value",
#         color="agent",
#         facet_row="variable",
#         facet_row_spacing=0.1,
#         color_discrete_map=color_map,
#         title="Error Rate Comparison by Agent and Model",
#         labels={"value": "Error Rate", "variable": "Error Type"},
#     )

#     # --- Customize lines, markers, and transparency ---
#     fig.for_each_trace(lambda t: t.update(
#         stackgroup=None,
#         mode="lines+markers",
#         fill="tozeroy",
#         line=dict(width=3),
#         marker=dict(size=8, line=dict(width=1, color="white")),
#         opacity=0.02
#     ))

#     # --- Layout ---
#     fig.update_layout(
#         title={
#             "text": "<b>Error Rate Comparison by Agents and Models</b>",
#             "x": 0.5,
#             "xanchor": "center",
#             "font": dict(family=title_font, size=26, color="black")
#         },
#         font=dict(family=text_font, size=14, color="black"),
#         plot_bgcolor="white",
#         paper_bgcolor="white",
#         legend_title_text="Agent",
#         legend=dict(
#             orientation="h",
#             yanchor="top",
#             y=-0.25,
#             xanchor="center",
#             x=0.5,
#             font=dict(family=text_font, size=13)
#         ),
#         margin=dict(t=90, b=150),
#         height=800
#     )

#     # --- Gridlines ---
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot", zeroline=False)
#     fig.update_xaxes(showgrid=False)

#     # --- Remove default facet titles ---
#     fig.for_each_annotation(lambda a: a.update(text=''))

#     # --- Custom horizontal facet titles ---
#     for i, variable in enumerate(melted_df["variable"].unique()):
#         fig.add_annotation(
#             text=f"<b>{variable.replace('_', ' ').title()}</b>",
#             x=0.5,
#             y=1.05 - (i * 0.55),
#             xref="paper",
#             yref="paper",
#             showarrow=False,
#             font=dict(family="Raleway, sans-serif", size=16, color="black"),
#             xanchor="center"
#         )

#     # --- Add values dynamically ---
#     for trace in fig.data:
#         fig.add_trace(
#             dict(
#                 type='scatter',
#                 x=trace.x,
#                 y=[y + 0.02 * max(trace.y) for y in trace.y],
#                 mode='text',
#                 text=[f"{y:.2f}" for y in trace.y],
#                 textposition="top center",
#                 textfont=dict(family="Open Sans, sans-serif", size=12, color="black"),
#                 showlegend=False,
#                 xaxis=trace.xaxis,
#                 yaxis=trace.yaxis
#             )
#         )

#     fig.show()



def error_rate_visualization(df):
    import plotly.express as px
    import plotly.graph_objects as go
    import random

    # --- Consistent Color Palette ---
    hex_colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]
    random.shuffle(hex_colors)

    # --- Color Map for Agents ---
    unique_agents = df["agent"].unique()
    color_map = {agent: hex_colors[i % len(hex_colors)] for i, agent in enumerate(unique_agents)}

    # --- Fonts ---
    title_font = "Raleway, sans-serif"
    text_font = "Open Sans, sans-serif"

    # --- Melted Data ---
    melted_df = df.melt(
        id_vars=["agent", "model", "library"],
        value_vars=["invalid_rate", "illegal_rate"]
    )

    # --- Create Area Chart (Vertical Facets) ---
    fig = px.area(
        melted_df,
        x="model",
        y="value",
        color="agent",
        facet_row="variable",
        facet_row_spacing=0.1,
        color_discrete_map=color_map,
        title="<b>Error Rate Comparison by Agents and Models</b>",
        labels={"value": "Error Rate", "variable": "Error Type"},
    )

    # --- Customize Lines, Markers, and Transparency ---
    fig.for_each_trace(lambda t: t.update(
        stackgroup=None,
        mode="lines+markers",
        fill="tozeroy",
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=1, color="white")),
        opacity=0.65
    ))

    # --- Layout Settings ---
    fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "font": dict(family=title_font, size=24, color="#222222")
        },
        font=dict(family=text_font, size=16, color="#222222"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Agent",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=16),
            title_font=dict(size=16)
        ),
        margin=dict(t=100, b=150, l=80, r=50),
        height=800
    )

    # --- Gridlines ---
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(180,180,180,0.6)", zeroline=False)
    fig.update_xaxes(showgrid=False)

    # --- Remove default facet titles ---
    fig.for_each_annotation(lambda a: a.update(text=''))

    # --- Custom horizontal facet titles ---
    for i, variable in enumerate(melted_df["variable"].unique()):
        fig.add_annotation(
            text=f"<b>{variable.replace('_', ' ').title()}</b>",
            x=0.5,
            y=1.05 - (i * 0.55),
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(family=title_font, size=18, color="#222222"),
            xanchor="center"
        )

    # --- Add data values on top of points (no background) ---
    for trace in fig.data:
        # Ensure we match correct subplot (facet)
        fig.add_trace(
            go.Scatter(
                x=trace.x,
                y=trace.y,
                mode="text",
                text=[f"{y:.2f}" if y is not None else "" for y in trace.y],
                textposition="top center",
                textfont=dict(family=text_font, size=16, color="black"),
                showlegend=False,
                xaxis=trace.xaxis,
                yaxis=trace.yaxis
            )
        )

    # --- Save High-Resolution Output (600 DPI) ---
    fig.write_image(
        "ErrorRate_Comparison.png",
        scale=6,   # ~600 DPI
        width=1200,
        height=800
    )

    # --- Display ---
    fig.show()



# ---------------------------------------------------------------------
# ✅ ADDING color_map logic to performance_comparison_plot
# ---------------------------------------------------------------------
def performance_comparison_plot(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import random

    # --- Consistent Color Palette ---
    hex_colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]
    random.shuffle(hex_colors)

    # --- Color Map for Agents ---
    unique_agents = df["agent"].unique()
    color_map = {agent: hex_colors[i % len(hex_colors)] for i, agent in enumerate(unique_agents)}

    # Fonts
    title_font = "Raleway, sans-serif"
    text_font = "Open Sans, sans-serif"

    # --- Subplots ---
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        subplot_titles=["Pass Rate", "Readability Score", "Quality Score"]
    )

    def label_every_n(values, n=2):
        return [f"{v:.2f}" if i % n == 0 else "" for i, v in enumerate(values)]

    # 1️⃣ Pass Rate
    for agent, color in color_map.items():
        subset = df[df["agent"] == agent]
        fig.add_trace(
            go.Bar(
                x=subset["model"],
                y=subset["pass_rate"],
                name=agent,
                marker_color=color,
                text=label_every_n(subset["pass_rate"], 1),
                textposition="outside",
                textfont=dict(size=12),
                cliponaxis=False
            ),
            row=1, col=1
        )

    # 2️⃣ Readability
    for idx, (agent, color) in enumerate(color_map.items()):
        subset = df[df["agent"] == agent]
        fig.add_trace(
            go.Scatter(
                x=subset["model"],
                y=subset["readability_score"],
                mode="lines+markers+text",
                name=agent,
                line=dict(width=3, color=color, shape="spline"),
                marker=dict(size=9, line=dict(width=1, color="black")),
                text=label_every_n(subset["readability_score"], 2),
                textposition="top center",
                textfont=dict(size=12),
            ),
            row=2, col=1
        )

    # 3️⃣ Quality
    for agent, color in color_map.items():
        subset = df[df["agent"] == agent]
        fig.add_trace(
            go.Scatter(
                x=subset["model"],
                y=subset["quality_score"],
                mode="lines+markers+text",
                name=agent,
                fill="tozeroy",
                line=dict(width=3, color=color, shape="spline"),
                marker=dict(size=8, line=dict(width=1, color="black")),
                opacity=0.35,
                text=label_every_n(subset["quality_score"], 2),
                textposition="top center",
                textfont=dict(size=12)
            ),
            row=3, col=1
        )

    fig.update_layout(
        title=dict(
            text="<b>Comparison of Pass Rate, Readability, and Quality Scores across Agents and Models</b>",
            x=0.5,
            font=dict(family=title_font, size=26)
        ),
        font=dict(family=text_font, size=14, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        barmode="group",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        height=1350,
        margin=dict(l=80, r=40, t=100, b=100)
    )

    fig.update_yaxes(showgrid=True, gridcolor="lightgray", griddash="dot")
    fig.update_xaxes(showgrid=False, tickangle=-25)
    fig.show()


# ---------------------------------------------------------------------
# ✅ ADDING color_map logic to performance_heatmap
# ---------------------------------------------------------------------
def performance_heatmap(df):
    import plotly.graph_objects as go
    import numpy as np
    import random

    # --- Consistent Color Palette ---
    hex_colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]
    random.shuffle(hex_colors)

    # --- Color Map for Agents ---
    unique_agents = df["agent"].unique()
    color_map = {agent: hex_colors[i % len(hex_colors)] for i, agent in enumerate(unique_agents)}

    # Fonts
    title_font = "Raleway, sans-serif"
    text_font = "Open Sans, sans-serif"

    # --- Reshape ---
    df_long = df.melt(
        id_vars=["agent", "model"],
        value_vars=["pass_rate", "readability_score", "quality_score"],
        var_name="metric",
        value_name="score"
    )

    df_long["metric_title"] = df_long["metric"].str.replace("_", " ").str.title()
    df_long["agent_metric"] = df_long.apply(
        lambda x: f"<b>{x['agent']}</b><br><span style='font-size:11px;color:gray;'>{x['metric_title']}</span>",
        axis=1
    )

    # --- Pivot ---
    heatmap_df = df_long.pivot(index="model", columns="agent_metric", values="score")

    z = heatmap_df.values
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)

    # --- Plot ---
    fig = go.Figure(data=go.Heatmap(
        z=z_norm,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=16, color="black"),
        colorscale="Tealrose_r",
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="<b>Score</b>", font=dict(size=16, family=title_font)),
            tickfont=dict(size=16, family=text_font)
        )
    ))
    title = "<b>Agents and Models Comparison on Performance Metrics</b>"

    fig.update_layout(
        title_font_size=20,
        title_font_family=title_font,
        title_font_color="#222222",  # Slightly darker for better contrast on white
        title_x=0.5,
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'xanchor': 'center'
        },
        legend_title_font_size=16,
        legend_title_font_family=text_font,
        font_family=text_font,
        width=1200,
        height=800,
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            title_font=dict(family=text_font, size=16),
            tickfont=dict(family=text_font, size=16),
            gridcolor='rgba(220,220,220,0.4)'  # Lighter grid lines
        ),
        yaxis=dict(
            title_font=dict(family=text_font, size=16),
            tickfont=dict(family=text_font, size=16),
            gridcolor='rgba(220,220,220,0.4)'  # Lighter grid lines
        ),
        # Add subtle box around plot area
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        legend=dict(
            bordercolor='rgba(220,220,220,0.4)',
            borderwidth=1
        )
    )

    # fig.update_layout(
    #     title=dict(text="<b>Agents and Models Comparison on Performance Metrics</b>", x=0.5, font=dict(size=26)),
    #     font=dict(family=text_font, size=13),
    #     plot_bgcolor="white",
    #     paper_bgcolor="white",
    #     height=950,
    #     margin=dict(l=120, r=180, t=100, b=120)
    # )

    fig.show()

