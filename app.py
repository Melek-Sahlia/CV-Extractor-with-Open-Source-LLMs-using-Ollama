from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session
import os
import time
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from werkzeug.utils import secure_filename
from pdf_processing import extract_text
from llm_integration import extract_with_llm
from evaluation import evaluate_extraction, load_ground_truth, compare_models

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'uploads'
GROUND_TRUTH_FOLDER = 'ground_truth'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf', 'png'}  # Add PNG to allowed extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GROUND_TRUTH_FOLDER'] = GROUND_TRUTH_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# LLM and OCR model configuration
LLM_MODELS = ['phi', 'llama3', 'mistral']  # Use 'phi' to match Ollama's model name
MODEL_DISPLAY_NAMES = {'phi': 'Phi-2', 'llama3': 'LLaMA 3', 'mistral': 'Mistral'}  # Display names for UI
OCR_MODELS = ['llava', 'mistral-vision']  # Available OCR-capable multimodal models
DEFAULT_OCR_MODEL = 'llava'

# Create necessary directories if they don't exist
for folder in [UPLOAD_FOLDER, GROUND_TRUTH_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', llm_models=LLM_MODELS, ocr_models=OCR_MODELS, default_ocr=DEFAULT_OCR_MODEL)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    model = request.form.get('model', 'phi')  # Default to phi instead of phi2
    use_ocr = 'use_ocr' in request.form  # This checks if the checkbox was checked
    ocr_model = request.form.get('ocr_model', DEFAULT_OCR_MODEL)
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Store the file path and model choices in session
        session['file_path'] = file_path
        session['model'] = model
        session['use_ocr'] = use_ocr
        session['ocr_model'] = ocr_model
        
        return redirect(url_for('process_file'))
    
    flash('Invalid file type. Please upload a PDF or PNG file.')
    return redirect(request.url)

@app.route('/process')
def process_file():
    file_path = session.get('file_path')
    model = session.get('model', 'phi')  # Default to phi
    use_ocr = session.get('use_ocr', False)
    ocr_model = session.get('ocr_model', DEFAULT_OCR_MODEL)
    
    if not file_path or not os.path.exists(file_path):
        flash('File not found. Please upload a file first.')
        return redirect(url_for('index'))
    
    return render_template('processing.html', 
                          filename=os.path.basename(file_path), 
                          model=model,
                          use_ocr=use_ocr,
                          ocr_model=ocr_model)

@app.route('/progress')
def progress():
    # Simulate progress updates (in a real app, this would track actual processing progress)
    return jsonify(progress=50)

@app.route('/extract', methods=['POST'])
def extract():
    file_path = session.get('file_path')
    model = session.get('model', 'phi')  # Default to phi
    use_ocr = session.get('use_ocr', False)
    ocr_model = session.get('ocr_model', DEFAULT_OCR_MODEL)
    
    if not file_path or not os.path.exists(file_path):
        return jsonify(error='File not found. Please upload a file first.')
    
    try:
        # Extract text from the PDF
        text = extract_text(file_path, use_mistral_ocr=use_ocr, ocr_model=ocr_model)
        
        # Safely extract structured information using the selected LLM
        try:
            extracted_data = extract_with_llm(text, model)
        except Exception as e:
            # If the selected model fails, try phi2 as fallback
            if model != 'phi':
                flash(f'Selected model {model} failed, using phi as backup. Error: {str(e)}')
                model = 'phi'
                extracted_data = extract_with_llm(text, model)
            else:
                raise e
        
        # Save the extracted data
        result_filename = os.path.basename(file_path).rsplit('.', 1)[0] + '_' + model + '.json'
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(extracted_data, f, indent=4)
        
        # Store the result path in session
        session['result_path'] = result_path
        
        return jsonify(success=True, redirect=url_for('show_results'))
    
    except Exception as e:
        return jsonify(error=str(e))

@app.route('/results')
def show_results():
    result_path = session.get('result_path')
    
    if not result_path or not os.path.exists(result_path):
        flash('Results not found. Please process a file first.')
        return redirect(url_for('index'))
    
    with open(result_path, 'r') as f:
        extracted_data = json.load(f)
    
    # Create default structure for incomplete data
    if not isinstance(extracted_data, dict):
        extracted_data = {
            "name": "",
            "email": "",
            "phone": "",
            "education": [],
            "experience": [],
            "skills": [],
            "error": "Invalid response format from model"
        }
    
    # Ensure all required fields exist
    required_fields = ["name", "email", "phone", "education", "experience", "skills"]
    for field in required_fields:
        if field not in extracted_data:
            extracted_data[field] = [] if field in ["education", "experience", "skills"] else ""
    
    return render_template('results.html', data=extracted_data)

@app.route('/evaluation_dashboard')
def evaluation_dashboard():
    # Check if we have ground truth and results for evaluation
    if not os.path.exists(GROUND_TRUTH_FOLDER) or not os.path.exists(RESULTS_FOLDER):
        flash('Evaluation data not found. Please ensure you have ground truth and results data.')
        return redirect(url_for('index'))
    
    # Load ground truth data
    ground_truth_path = os.path.join(GROUND_TRUTH_FOLDER, 'ground_truth.json')
    if not os.path.exists(ground_truth_path):
        flash('Ground truth file not found. Please ensure ground_truth.json is in the ground_truth folder.')
        return redirect(url_for('index'))
    
    try:
        with open(ground_truth_path, 'r') as f:
            ground_truth_data = json.load(f)
    except Exception as e:
        flash(f'Error loading ground truth file: {str(e)}')
        return redirect(url_for('index'))
    
    # Get list of result files
    result_files = [f for f in os.listdir(RESULTS_FOLDER) 
                    if f.endswith('.json') and os.path.isfile(os.path.join(RESULTS_FOLDER, f))]
    
    if not result_files:
        flash('No result files found. Please add result data to evaluate models.')
        return redirect(url_for('index'))
    
    # Debug what we have
    print(f"Found {len(result_files)} result files.")
    
    # Initialize results storage
    all_comparisons = []
    overall_results = {
        'llama3': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        'mistral': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        'phi': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    }
    field_results = {field: {'llama3': 0.0, 'mistral': 0.0, 'phi': 0.0} 
                    for field in ['name', 'email', 'phone', 'education', 'experience', 'skills']}
    
    # List of CVs we've evaluated
    evaluated_cvs = set()
    
    # Print ground truth CVs for debugging
    ground_truth_cvs = set()
    
    # Create a mapping between base names and their ground truth data
    # This will handle both cv_1.pdf and cv_1.png pointing to the same ground truth
    base_name_to_gt = {}
    for cv_filename, ground_truth in ground_truth_data.items():
        base_name = os.path.splitext(cv_filename)[0]
        # Remove file extension and any suffix like _1, _2 etc.
        core_name = base_name.split('_')[0] + '_' + base_name.split('_')[1] if '_' in base_name else base_name
        ground_truth_cvs.add(core_name)
        base_name_to_gt[base_name] = ground_truth

    print(f"Ground truth CVs: {sorted(list(ground_truth_cvs))}")
    
    # Process each CV in ground truth data by their base names
    for base_name, ground_truth in base_name_to_gt.items():
        print(f"Processing ground truth CV: {base_name}")
        
        # Find corresponding result files for each model
        model_results = {}
        
        # Look for results with this CV name for each model
        for model in LLM_MODELS:
            # Try different naming patterns for result files
            result_found = False
            
            # Various pattern matching approaches:
            result_patterns = [
                f"{base_name}_{model}.json",  # Standard pattern: cv_1_llama3.json
                f"{base_name}.{model}.json",  # Alternative: cv_1.llama3.json
                f"{base_name}-{model}.json",  # Alternative: cv_1-llama3.json
            ]
            
            # Try each pattern
            for pattern in result_patterns:
                if pattern in result_files:
                    result_path = os.path.join(RESULTS_FOLDER, pattern)
                    try:
                        with open(result_path, 'r') as f:
                            model_results[model] = json.load(f)
                        result_found = True
                        print(f"  Found result for {model}: {pattern}")
                        break  # Found a match, no need to try other patterns
                    except Exception as e:
                        flash(f'Error loading results file {pattern}: {str(e)}')
                        print(f"  Error loading {pattern}: {str(e)}")
            
            # If not found, look for files containing the base name in a more flexible way
            if not result_found:
                for result_file in result_files:
                    if base_name in result_file and model in result_file and result_file.endswith('.json'):
                        result_path = os.path.join(RESULTS_FOLDER, result_file)
                        try:
                            with open(result_path, 'r') as f:
                                model_results[model] = json.load(f)
                            result_found = True
                            print(f"  Found fuzzy match for {model}: {result_file}")
                            break
                        except Exception as e:
                            flash(f'Error loading results file {result_file}: {str(e)}')
                            print(f"  Error loading {result_file}: {str(e)}")
            
            if not result_found:
                print(f"  No results found for {model}")
        
        # Check if we have results for at least one model
        if model_results:
            evaluated_cvs.add(base_name)
            
            # Compare models and store results
            comparison = compare_models(ground_truth, 
                                       model_results.get('llama3', {}), 
                                       model_results.get('mistral', {}), 
                                       model_results.get('phi', {}))
            
            all_comparisons.append({
                'cv_name': base_name,
                'results': comparison
            })
            
            # Aggregate overall results for each model
            for model in LLM_MODELS:
                if model in model_results and model in comparison:
                    overall_results[model]['precision'] += comparison[model]['overall']['precision']
                    overall_results[model]['recall'] += comparison[model]['overall']['recall'] 
                    overall_results[model]['f1'] += comparison[model]['overall']['f1']
                    
                    # Aggregate field results
                    for field in field_results:
                        field_results[field][model] += comparison[model]['fields'][field]['f1']
        else:
            print(f"  No results found for any model for {base_name}")
    
    # Calculate averages if we have comparisons
    num_comparisons = len(all_comparisons)
    print(f"Total comparisons made: {num_comparisons}")
    if num_comparisons > 0:
        # Average the overall results
        for model in overall_results:
            overall_results[model]['precision'] /= num_comparisons
            overall_results[model]['recall'] /= num_comparisons
            overall_results[model]['f1'] /= num_comparisons
            
        # Average the field results
        for field in field_results:
            for model in field_results[field]:
                field_results[field][model] /= num_comparisons
    else:
        flash('No complete evaluations could be performed. Please check your data.')
        return redirect(url_for('index'))
    
    print(f"Models evaluated: {LLM_MODELS}")
    print(f"CVs evaluated: {sorted(list(evaluated_cvs))}")
    
    # Check which models have results
    active_models = []
    active_display_models = []
    
    for model in LLM_MODELS:
        if overall_results[model]['f1'] > 0:
            active_models.append(model)
            active_display_models.append(MODEL_DISPLAY_NAMES[model])
    
    print(f"Active models in evaluation: {active_models}")
    
    # Only show models that have results
    if not active_models:
        flash('No models have results available for evaluation.')
        return redirect(url_for('index'))
    
    # Prepare data for charts with only active models
    precision_values = [overall_results[model]['precision'] for model in active_models]
    recall_values = [overall_results[model]['recall'] for model in active_models]
    f1_values = [overall_results[model]['f1'] for model in active_models]
    
    # Generate charts with active models only
    precision_chart = generate_precision_chart_from_values(active_display_models, precision_values)
    recall_chart = generate_recall_chart_from_values(active_display_models, recall_values)
    f1_chart = generate_f1_chart_from_values(active_display_models, f1_values)
    overall_chart = generate_overall_chart_from_values(active_display_models, precision_values, recall_values, f1_values)
    
    # Generate field comparison chart from field results
    field_comparison_chart = generate_field_comparison_chart_from_values(field_results, active_models)
    
    # Create the combined results structure for the template
    comparison_results = {
        'llama3': {
            'overall': overall_results['llama3'],
            'fields': {field: {
                'precision': 0, 
                'recall': 0, 
                'f1': field_results[field]['llama3']
            } for field in field_results}
        },
        'mistral': {
            'overall': overall_results['mistral'],
            'fields': {field: {
                'precision': 0, 
                'recall': 0, 
                'f1': field_results[field]['mistral']
            } for field in field_results}
        },
        'phi': {
            'overall': overall_results['phi'],
            'fields': {field: {
                'precision': 0, 
                'recall': 0, 
                'f1': field_results[field]['phi']
            } for field in field_results}
        }
    }
    
    # Calculate field-level precision and recall for the average comparison
    if all_comparisons:
        for model_key in ['llama3', 'mistral', 'phi']:
            for field in field_results:
                total_precision = 0
                total_recall = 0
                count = 0
                for comp in all_comparisons:
                    comp_results = comp['results']
                    if model_key in comp_results:
                        total_precision += comp_results[model_key]['fields'][field]['precision']
                        total_recall += comp_results[model_key]['fields'][field]['recall']
                        count += 1
                
                if count > 0:
                    comparison_results[model_key]['fields'][field]['precision'] = total_precision / count
                    comparison_results[model_key]['fields'][field]['recall'] = total_recall / count
    
    return render_template('evaluation.html', 
                           precision_chart=precision_chart,
                           recall_chart=recall_chart,
                           f1_chart=f1_chart,
                           overall_chart=overall_chart,
                           field_comparison_chart=field_comparison_chart,
                           comparison_results=comparison_results,
                           all_comparisons=all_comparisons,
                           num_cvs_evaluated=len(evaluated_cvs),
                           active_models=active_models)

def generate_precision_chart_from_values(models, precision_values):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, precision_values, color=['#4285F4', '#0F9D58', '#F4B400'])
    plt.ylim(0, 1.0)
    plt.title('Precision Comparison')
    plt.ylabel('Precision')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Save the plot to a bytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes to base64
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Return the base64 encoded string
    return graphic

def generate_recall_chart_from_values(models, recall_values):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, recall_values, color=['#4285F4', '#0F9D58', '#F4B400'])
    plt.ylim(0, 1.0)
    plt.title('Recall Comparison')
    plt.ylabel('Recall')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Save the plot to a bytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes to base64
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Return the base64 encoded string
    return graphic

def generate_f1_chart_from_values(models, f1_values):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, f1_values, color=['#4285F4', '#0F9D58', '#F4B400'])
    plt.ylim(0, 1.0)
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Save the plot to a bytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes to base64
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Return the base64 encoded string
    return graphic

def generate_overall_chart_from_values(models, precision_values, recall_values, f1_values):
    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_values, width, label='Precision', color='#4285F4')
    plt.bar(x, recall_values, width, label='Recall', color='#0F9D58')
    plt.bar(x + width, f1_values, width, label='F1 Score', color='#F4B400')
    
    plt.ylabel('Score')
    plt.title('Overall Model Performance Comparison')
    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Save the plot to a bytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the bytes to base64
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Return the base64 encoded string
    return graphic

def generate_field_comparison_chart_from_values(field_results, active_models):
    # If only one model is active, create a simplified chart
    if len(active_models) == 1:
        model = active_models[0]
        # Create a chart comparing field performances for only one model
        fields = list(field_results.keys())
        f1_values = [field_results[field][model] for field in fields]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fields, f1_values, color='#F4B400')
        
        plt.ylabel('F1 Score')
        plt.title(f'Field-level F1 Scores for {MODEL_DISPLAY_NAMES[model]} Model')
        plt.xticks(rotation=0)
        plt.ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Save the plot to a bytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Encode the bytes to base64
        graphic = base64.b64encode(image_png).decode('utf-8')
        
        return graphic
    else:
        # Original code for multiple models
        fields = list(field_results.keys())
        
        # Only use data for active models
        x = np.arange(len(fields))  # the label locations
        width = 0.8 / len(active_models)  # Scale the width based on number of models
        offset_multiplier = np.linspace(-(len(active_models)-1)/2, (len(active_models)-1)/2, len(active_models))
        
        plt.figure(figsize=(12, 6))
        
        # Use different colors for each model
        colors = ['#4285F4', '#0F9D58', '#F4B400']
        
        # Plot bars for each active model
        for i, model in enumerate(active_models):
            model_f1 = [field_results[field][model] for field in fields]
            offset = offset_multiplier[i] * width
            plt.bar(x + offset, model_f1, width, label=MODEL_DISPLAY_NAMES[model], color=colors[i % len(colors)])
        
        plt.ylabel('F1 Score')
        plt.title('Field-level F1 Score Comparison')
        plt.xticks(x, [field.capitalize() for field in fields])
        plt.ylim(0, 1.0)
        plt.legend()
        
        # Save the plot to a bytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Encode the bytes to base64
        graphic = base64.b64encode(image_png).decode('utf-8')
        
        return graphic

if __name__ == '__main__':
    app.run(debug=True) 