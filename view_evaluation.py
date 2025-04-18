import os
import json
import webbrowser
import io
import base64
from jinja2 import Template
from run_evaluation import run_evaluation

def create_html_report(evaluation_results):
    """
    Create an HTML report from the evaluation results.
    
    Args:
        evaluation_results (dict): The evaluation results
        
    Returns:
        str: The HTML report content
    """
    # Load the template
    template_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CV Information Extraction Evaluation Results</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3, h4 {
                color: #333;
                margin-bottom: 20px;
            }
            h1 {
                font-size: 32px;
                font-weight: 700;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 15px;
                margin-bottom: 30px;
            }
            h2 {
                font-size: 24px;
                font-weight: 600;
                margin-top: 40px;
            }
            .chart-container {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.05);
                margin-bottom: 30px;
            }
            .metrics-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.05);
                height: 100%;
            }
            .metrics-card h4 {
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #495057;
            }
            .model-label {
                font-weight: 600;
                font-size: 18px;
                display: inline-block;
                margin-right: 15px;
            }
            .model-badge {
                display: inline-block;
                padding: 8px 15px;
                border-radius: 30px;
                color: white;
                font-weight: 500;
                margin-bottom: 20px;
            }
            .badge-llama3 {
                background-color: #4285F4;
            }
            .badge-mistral {
                background-color: #0F9D58;
            }
            .badge-phi {
                background-color: #F4B400;
            }
            .metric-row {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .metric-label {
                width: 100px;
                font-weight: 500;
                color: #6c757d;
            }
            .metric-value {
                font-weight: 600;
                font-size: 18px;
            }
            .field-metrics {
                margin-top: 30px;
            }
            .section-divider {
                height: 2px;
                background-color: #f0f0f0;
                margin: 40px 0;
            }
            .text-center {
                text-align: center;
            }
            footer {
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                text-align: center;
                color: #6c757d;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center">CV Information Extraction Evaluation</h1>
            
            <div class="alert alert-info">
                <p><strong>Evaluated CVs:</strong> {{ evaluated_cvs|join(', ') }}</p>
                <p>The charts and metrics below represent the average performance across all evaluated CVs.</p>
            </div>
            
            <div class="row">
                {% for model in active_models %}
                <div class="col-md-4 mb-4">
                    <div class="metrics-card">
                        <div class="model-badge badge-{{ model }}">{{ model_names[model] }}</div>
                        <div class="metric-row">
                            <span class="metric-label">Precision:</span>
                            <span class="metric-value">{{ "%.2f"|format(overall_results[model]['precision']) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Recall:</span>
                            <span class="metric-value">{{ "%.2f"|format(overall_results[model]['recall']) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">F1 Score:</span>
                            <span class="metric-value">{{ "%.2f"|format(overall_results[model]['f1']) }}</span>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <h2 class="text-center">Overall Performance</h2>
            
            <!-- Overall Performance Chart -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="chart-container">
                        <img src="data:image/png;base64,{{ charts['overall_chart'] }}" alt="Overall Performance Comparison" class="img-fluid">
                    </div>
                </div>
            </div>
            
            <!-- Precision, Recall, F1 Charts Row -->
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="chart-container">
                        <h4 class="text-center">Precision</h4>
                        <img src="data:image/png;base64,{{ charts['precision_chart'] }}" alt="Precision Comparison" class="img-fluid">
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="chart-container">
                        <h4 class="text-center">Recall</h4>
                        <img src="data:image/png;base64,{{ charts['recall_chart'] }}" alt="Recall Comparison" class="img-fluid">
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="chart-container">
                        <h4 class="text-center">F1 Score</h4>
                        <img src="data:image/png;base64,{{ charts['f1_chart'] }}" alt="F1 Score Comparison" class="img-fluid">
                    </div>
                </div>
            </div>
            
            <h2 class="text-center">Field-Level Performance</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts['field_comparison_chart'] }}" alt="Field-Level Performance Comparison" class="img-fluid">
            </div>
            
            <div class="section-divider"></div>
            
            <h2>Field-Level Details</h2>
            <div class="row">
                {% for field in field_results %}
                <div class="col-md-4 mb-4">
                    <div class="metrics-card">
                        <h4>{{ field|capitalize }}</h4>
                        {% for model in active_models %}
                        <div class="metric-row">
                            <span class="metric-label">{{ model_names[model] }}:</span>
                            <span class="metric-value">{{ "%.2f"|format(field_results[field][model]) }}</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="section-divider"></div>
            
            <h2>Technical Information</h2>
            <div class="row">
                <div class="col-md-6">
                    <h4>Models Evaluated</h4>
                    <ul>
                        {% for model in active_models %}
                        <li><strong>{{ model_names[model] }}</strong></li>
                        {% endfor %}
                    </ul>
                </div>
                <div class="col-md-6">
                    <h4>Technologies Used</h4>
                    <ul>
                        <li><strong>Evaluation:</strong> Custom Python metrics</li>
                        <li><strong>Visualization:</strong> Matplotlib</li>
                        <li><strong>Reporting:</strong> Jinja2 + Bootstrap</li>
                    </ul>
                </div>
            </div>
            
            <footer>
                <p>Generated on {{ generation_date }}</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Create the template
    template = Template(template_html)
    
    # Extract data
    overall_results = evaluation_results.get('overall_results', {})
    field_results = evaluation_results.get('field_results', {})
    active_models = evaluation_results.get('active_models', [])
    evaluated_cvs = evaluation_results.get('evaluated_cvs', [])
    charts = evaluation_results.get('charts', {})
    
    # Get current date and time
    from datetime import datetime
    generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Model display names
    model_names = {'phi': 'Phi-2', 'llama3': 'LLaMA 3', 'mistral': 'Mistral'}
    
    # Render the template
    html = template.render(
        overall_results=overall_results,
        field_results=field_results,
        active_models=active_models,
        evaluated_cvs=evaluated_cvs,
        charts=charts,
        model_names=model_names,
        generation_date=generation_date
    )
    
    return html

def main():
    """
    Main function to run the evaluation and generate the HTML report.
    """
    # Force a new evaluation by removing existing results
    if os.path.exists('evaluation_results.json'):
        os.remove('evaluation_results.json')
        print("Removed existing evaluation results. Running new evaluation...")
    
    # Run the evaluation
    evaluation_results = run_evaluation()
    
    if not evaluation_results:
        print("No evaluation results available.")
        return
    
    # Check if we need to generate charts
    if 'charts' not in evaluation_results or not evaluation_results['charts']:
        from run_evaluation import generate_charts
        charts = generate_charts(evaluation_results)
        evaluation_results['charts'] = charts
    
    # Create the HTML report
    html = create_html_report(evaluation_results)
    
    # Save the HTML report
    report_file = 'evaluation_report.html'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Report saved to {report_file}")
    
    # Open the report in the default browser
    try:
        webbrowser.open('file://' + os.path.realpath(report_file))
        print("Opening report in browser...")
    except Exception as e:
        print(f"Error opening browser: {str(e)}")
        print(f"Please open {report_file} manually.")

if __name__ == "__main__":
    main() 