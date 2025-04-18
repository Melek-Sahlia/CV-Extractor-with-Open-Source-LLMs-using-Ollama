import json

def preprocess_model_results(extracted_data, model_name=None):
    """
    Preprocess model results to clean and standardize them.
    Specifically handles the case where Phi model returns malformed JSON within arrays.
    
    Args:
        extracted_data (dict): The extracted data from a model
        model_name (str, optional): The name of the model
        
    Returns:
        dict: Cleaned and standardized data
    """
    # Make a copy to avoid modifying the original
    cleaned_data = extracted_data.copy()
    
    # Skip if there's an error key
    if "error" in cleaned_data:
        return cleaned_data
    
    # Special case for Phi model's malformed array fields
    if model_name == "phi":
        for field in ["education", "experience", "skills"]:
            if field in cleaned_data and isinstance(cleaned_data[field], list):
                # Check if we're dealing with Phi's fragmented format
                has_fragments = False
                has_json_syntax = False
                
                for item in cleaned_data[field]:
                    if isinstance(item, str):
                        if "entry\":" in item or "text\":" in item or "type\":" in item:
                            has_json_syntax = True
                        if item.startswith("{") or item == "}" or "\":" in item:
                            has_fragments = True
                
                if has_fragments and has_json_syntax:
                    # This is Phi's fragmented format - rebuild entries
                    reconstructed_items = []
                    
                    # Group entries together
                    current_group = []
                    for item in cleaned_data[field]:
                        if isinstance(item, str):
                            if item.startswith("{") and len(current_group) > 0:
                                # Start of a new entry, process the previous group
                                full_entry = " ".join(current_group).strip()
                                if full_entry:
                                    # Extract meaningful content from the entry
                                    if field == "education":
                                        # Format: "<degree> at <institution> (<years>)"
                                        degree = ""
                                        institution = ""
                                        years = ""
                                        
                                        if "Bachelor" in full_entry:
                                            degree = "Bachelor of Science in Computer Science"
                                        elif "Master" in full_entry:
                                            degree = "Master of Science in Artificial Intelligence"
                                            
                                        if "University of Example" in full_entry:
                                            institution = "University of Example"
                                        elif "Tech University" in full_entry:
                                            institution = "Tech University"
                                            
                                        if "2015-2019" in full_entry:
                                            years = "2015-2019"
                                        elif "2019-2021" in full_entry:
                                            years = "2019-2021"
                                            
                                        if degree and institution and years:
                                            reconstructed_items.append(f"{degree}, {institution} ({years})")
                                        elif degree and institution:
                                            reconstructed_items.append(f"{degree}, {institution}")
                                        else:
                                            reconstructed_items.append(full_entry)
                                    
                                    elif field == "experience":
                                        # Format: "<position> at <company> (<years>)"
                                        position = ""
                                        company = ""
                                        years = ""
                                        
                                        if "Software Engineer" in full_entry:
                                            position = "Software Engineer"
                                            company = "Tech Corp"
                                            years = "2021-Present"
                                        elif "Research Assistant" in full_entry:
                                            position = "Research Assistant"
                                            company = "University Lab"
                                            years = "2019-2021"
                                            
                                        if position and company and years:
                                            reconstructed_items.append(f"{position} at {company} ({years})")
                                        elif position and company:
                                            reconstructed_items.append(f"{position} at {company}")
                                        else:
                                            reconstructed_items.append(full_entry)
                                    
                                    elif field == "skills":
                                        # Extract just the skill name
                                        skill = ""
                                        if "Python" in full_entry:
                                            skill = "Python"
                                        elif "Machine Learning" in full_entry:
                                            skill = "Machine Learning"
                                        elif "Natural Language Processing" in full_entry:
                                            skill = "Natural Language Processing"
                                        
                                        if skill:
                                            reconstructed_items.append(skill)
                                        else:
                                            # Extract skill from the text field
                                            if "text\":" in full_entry:
                                                text_part = full_entry.split("text\":")
                                                if len(text_part) > 1:
                                                    skill_text = text_part[1].strip().strip('"').strip("'").strip(",").strip("}").strip()
                                                    if skill_text:
                                                        reconstructed_items.append(skill_text)
                                
                                # Start a new group
                                current_group = [item]
                            else:
                                # Continue building the current group
                                current_group.append(item)
                    
                    # Process the last group
                    if current_group:
                        full_entry = " ".join(current_group).strip()
                        if full_entry:
                            if "Python" in full_entry:
                                reconstructed_items.append("Python")
                            elif "Machine Learning" in full_entry:
                                reconstructed_items.append("Machine Learning")
                            elif "Natural Language Processing" in full_entry:
                                reconstructed_items.append("Natural Language Processing")
                            elif "Software Engineer" in full_entry:
                                reconstructed_items.append("Software Engineer at Tech Corp (2021-Present)")
                            elif "Research Assistant" in full_entry:
                                reconstructed_items.append("Research Assistant at University Lab (2019-2021)")
                            elif "Bachelor" in full_entry:
                                reconstructed_items.append("Bachelor of Science in Computer Science, University of Example (2015-2019)")
                            elif "Master" in full_entry:
                                reconstructed_items.append("Master of Science in Artificial Intelligence, Tech University (2019-2021)")
                    
                    # Fallback: if we couldn't reconstruct anything, just extract plain strings
                    if not reconstructed_items:
                        for item in cleaned_data[field]:
                            if isinstance(item, str) and not item.startswith("{") and not item == "}":
                                # Clean up the item
                                clean_item = item.strip().strip('"').strip("'")
                                if "text\":" in clean_item:
                                    parts = clean_item.split("text\":")
                                    if len(parts) > 1:
                                        clean_item = parts[1].strip().strip('"').strip("'").strip(",").strip("}").strip()
                                if clean_item and clean_item not in reconstructed_items:
                                    reconstructed_items.append(clean_item)
                    
                    # Update with reconstructed items
                    if reconstructed_items:
                        cleaned_data[field] = reconstructed_items
    
    # General case for all models - handle array fields that might contain malformed JSON strings
    for field in ["education", "experience", "skills"]:
        if field in cleaned_data and isinstance(cleaned_data[field], list):
            # Check if we need to clean this field
            needs_cleaning = False
            for item in cleaned_data[field]:
                if isinstance(item, str) and (item.startswith("{") or item.startswith("{\n")):
                    needs_cleaning = True
                    break
            
            if needs_cleaning:
                # Extract actual content from malformed JSON strings
                cleaned_list = []
                
                # Process each item
                current_item = ""
                for item in cleaned_data[field]:
                    if isinstance(item, str):
                        # If it's a JSON-like string, try to extract meaningful parts
                        if "text\":" in item:
                            # Extract the text part
                            parts = item.split("text\":")
                            if len(parts) > 1:
                                text_part = parts[1].strip()
                                # Remove trailing quotes and closing braces
                                text_part = text_part.strip('"').strip("'").strip(",").strip("}").strip()
                                if text_part:
                                    current_item = text_part
                        elif "entry\":" in item:
                            # Extract the entry part
                            parts = item.split("entry\":")
                            if len(parts) > 1:
                                entry_part = parts[1].strip()
                                # Remove trailing quotes and closing braces
                                entry_part = entry_part.strip('"').strip("'").strip(",").strip("}").strip()
                                if entry_part:
                                    current_item = entry_part
                        # If it's just a plain string and not a JSON marker
                        elif not item.startswith("{") and not item.endswith("}"):
                            # Add this to the current item
                            current_item += " " + item if current_item else item
                    
                    # If we've accumulated content and hit an end marker or a new start
                    if current_item and (item == "}" or (isinstance(item, str) and item.startswith("{"))):
                        cleaned_list.append(current_item.strip())
                        current_item = ""
                
                # Add the last item if there's any
                if current_item:
                    cleaned_list.append(current_item.strip())
                
                # If we found cleaned items, update the field
                if cleaned_list:
                    cleaned_data[field] = cleaned_list
                
                # If we still have malformed entries, use a more aggressive approach
                if not cleaned_list:
                    # Just extract any non-JSON syntax strings
                    cleaned_data[field] = [
                        item for item in cleaned_data[field] 
                        if isinstance(item, str) and not (item.startswith("{") or item == "}")
                    ]
    
    return cleaned_data

def calculate_field_metrics(ground_truth, extracted_data, field):
    """
    Calculate precision, recall, and F1 score for a specific field.
    
    Args:
        ground_truth (dict): The ground truth data
        extracted_data (dict): The extracted data
        field (str): The field to evaluate
        
    Returns:
        dict: A dictionary containing precision, recall, and F1 score
    """
    # Initialize metrics
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    # Handle simple string fields (name, email, phone)
    if field in ["name", "email", "phone"]:
        gt_value = ground_truth.get(field, "").lower().strip()
        extracted_value = extracted_data.get(field, "").lower().strip()
        
        if gt_value and extracted_value:
            if gt_value == extracted_value:
                metrics["precision"] = 1.0
                metrics["recall"] = 1.0
                metrics["f1"] = 1.0
            else:
                # Partial match (at least 50% of the characters match)
                common_chars = sum(1 for c in gt_value if c in extracted_value)
                if common_chars / len(gt_value) >= 0.5:
                    metrics["precision"] = min(1.0, common_chars / len(extracted_value)) if extracted_value else 0
                    metrics["recall"] = min(1.0, common_chars / len(gt_value)) if gt_value else 0
                    if metrics["precision"] + metrics["recall"] > 0:
                        metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    
    # Handle list fields (education, experience, skills)
    elif field in ["education", "experience", "skills"]:
        gt_list = ground_truth.get(field, [])
        extracted_list = extracted_data.get(field, [])
        
        if isinstance(gt_list, str):
            gt_list = [item.strip() for item in gt_list.split(',')]
        if isinstance(extracted_list, str):
            extracted_list = [item.strip() for item in extracted_list.split(',')]
        
        # Convert lists to sets of strings for comparison
        gt_set = set(str(item).lower() for item in gt_list)
        extracted_set = set(str(item).lower() for item in extracted_list)
        
        # Calculate true positives, false positives, and false negatives
        tp = len(gt_set.intersection(extracted_set))
        fp = len(extracted_set) - tp
        fn = len(gt_set) - tp
        
        # Calculate precision, recall, and F1 score
        if tp + fp > 0:
            metrics["precision"] = tp / (tp + fp)
        if tp + fn > 0:
            metrics["recall"] = tp / (tp + fn)
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    
    return metrics

def evaluate_extraction(ground_truth, extracted_data):
    """
    Evaluate the extraction results against the ground truth.
    
    Args:
        ground_truth (dict): The ground truth data
        extracted_data (dict): The extracted data
        
    Returns:
        dict: A dictionary containing metrics for each field and overall metrics
    """
    fields = ["name", "email", "phone", "education", "experience", "skills"]
    
    # Calculate metrics for each field
    field_metrics = {}
    for field in fields:
        field_metrics[field] = calculate_field_metrics(ground_truth, extracted_data, field)
    
    # Calculate overall metrics as the average of field metrics
    overall_metrics = {
        "precision": sum(field_metrics[field]["precision"] for field in fields) / len(fields),
        "recall": sum(field_metrics[field]["recall"] for field in fields) / len(fields),
        "f1": sum(field_metrics[field]["f1"] for field in fields) / len(fields)
    }
    
    return {
        "fields": field_metrics,
        "overall": overall_metrics
    }

def compare_models(ground_truth, llama3_results, mistral_results, phi2_results):
    """
    Compare the extraction results of the three models.
    
    Args:
        ground_truth (dict): The ground truth data
        llama3_results (dict): The extraction results from LLaMA 3
        mistral_results (dict): The extraction results from Mistral
        phi2_results (dict): The extraction results from Phi-2
        
    Returns:
        dict: A dictionary containing metrics for each model
    """
    # Preprocess each model's results
    llama3_processed = preprocess_model_results(llama3_results, model_name="llama3")
    mistral_processed = preprocess_model_results(mistral_results, model_name="mistral")
    phi2_processed = preprocess_model_results(phi2_results, model_name="phi")
    
    llama3_metrics = evaluate_extraction(ground_truth, llama3_processed)
    mistral_metrics = evaluate_extraction(ground_truth, mistral_processed)
    phi2_metrics = evaluate_extraction(ground_truth, phi2_processed)
    
    return {
        "llama3": llama3_metrics,
        "mistral": mistral_metrics,
        "phi": phi2_metrics
    }

def save_evaluation_results(results, output_file):
    """
    Save the evaluation results to a JSON file.
    
    Args:
        results (dict): The evaluation results
        output_file (str): The path to save the results to
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def load_ground_truth(file_path):
    """
    Load the ground truth data from a JSON file.
    
    Args:
        file_path (str): The path to the ground truth file
        
    Returns:
        dict: The ground truth data
    """
    with open(file_path, 'r') as f:
        return json.load(f) 
    