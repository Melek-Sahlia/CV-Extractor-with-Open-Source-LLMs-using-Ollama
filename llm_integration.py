# Placeholder functions for LLM integration

import requests
import json
import time
import re

# Base URL for Ollama API
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Timeout settings
DEFAULT_TIMEOUT = 300  # 5 minutes as a default

# Function to extract CV data using LLaMA 3 via Ollama
def run_llama3_extraction(text, timeout=DEFAULT_TIMEOUT):
    prompt = f"""
    EXTRACT INFORMATION FROM THIS CV AND FORMAT AS JSON.
    
    CRITICAL INSTRUCTIONS (FOLLOW PRECISELY):
    1. YOU MUST RETURN ONLY A VALID JSON OBJECT WITH DOUBLE QUOTES
    2. DO NOT RETURN ANY PYTHON CODE, FUNCTIONS, OR CLASSES
    3. DO NOT USE CODE BLOCKS OR MARKDOWN FORMAT. NO ```
    4. DO NOT RETURN IMPORT STATEMENTS
    5. DO NOT SUGGEST CODE OR FUNCTIONS TO PROCESS THE CV
    6. YOUR ENTIRE RESPONSE SHOULD BE *JUST* THE JSON OBJECT
    7. ONLY EXTRACT REAL DATA FROM THE CV TEXT
    8. DO NOT USE PLACEHOLDERS OR EXAMPLE DATA
    9. FIELDS MISSING FROM THE CV SHOULD BE EMPTY STRINGS OR ARRAYS
    
    Expected fields:
    - name: The person's full name 
    - email: Email address from the CV
    - phone: Phone number from the CV
    - education: List of education entries
    - experience: List of work experiences
    - skills: List of skills mentioned
    
    REQUIRED FORMAT (USE DOUBLE QUOTES, NOT SINGLE QUOTES):
    {{
      "name": "Real name from CV",
      "email": "Real email from CV",
      "phone": "Real phone from CV",
      "education": ["Real education 1", "Real education 2"],
      "experience": ["Real experience 1", "Real experience 2"],
      "skills": ["Real skill 1", "Real skill 2"]
    }}
    
    CV TEXT TO EXTRACT FROM:
    
    {text}
    
    REMINDER: RETURN ONLY THE JSON OBJECT WITH REAL DATA.
    NO CODE BLOCKS, NO PYTHON CODE, NO FUNCTIONS, NO MARKDOWN.
    YOUR ENTIRE RESPONSE SHOULD BE JUST THE JSON OBJECT AND NOTHING ELSE.
    """
    
    try:
        print("Running llama3 extraction with Ollama...")
        model_name = "llama3"
        
        # Set a temperature parameter to reduce randomness and increase parameter settings
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Slight temperature to allow creativity but not too much
                    "num_predict": 2048,  # Increase token limit for complete response
                    "top_p": 0.9,        # Reduce randomness
                    "top_k": 30          # Focus on more likely tokens
                }
            },
            timeout=timeout
        )
        
        # Use the same error-resistant JSON extraction logic
        if response.status_code == 200:
            try:
                result = response.json()
                extracted_text = result.get("response", "")
                print(f"Raw response length: {len(extracted_text)}")
                print(f"Raw response first 100 chars: {extracted_text[:100]}")
                
                # More thorough check for Python code patterns
                code_indicators = [
                    "import ", "def ", "```", "class ", 
                    "print(", "return ", "function", 
                    "# Your code", "# This function", 
                    "if __name__", "for ", "while ",
                    "try:", "except:", " = function",
                    "@param", "params", "# Test"
                ]
                
                for indicator in code_indicators:
                    if indicator in extracted_text:
                        print(f"Detected code indicator: '{indicator}' in response")
                        return {"error": "Model returned Python code instead of JSON", "raw_response": extracted_text[:200]}
                
                # Extra cleaning to handle potential code blocks
                extracted_text = extracted_text.replace("```json", "").replace("```", "")
                
                # Check if response contains our example data which would indicate the model just repeated our example
                example_data_indicators = [
                    "Extracted Name Here", 
                    "actual.email@fromcv.com", 
                    "Real phone from CV",
                    "Actual education entry",
                    "Actual experience entry",
                    "Actual skill"
                ]
                
                for indicator in example_data_indicators:
                    if indicator in extracted_text:
                        print(f"Model returned example data ({indicator}), rejecting response")
                        return {"error": "Model returned example data instead of extraction", "raw_response": extracted_text[:200]}
                
                # Try to parse the JSON response
                try:
                    # First, try to find a complete JSON object with improved regex
                    json_match = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', extracted_text)
                    if json_match:
                        json_str = json_match.group(1)
                        print(f"Found JSON pattern, length: {len(json_str)}")
                        
                        # Clean the JSON
                        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)  # Ensure property names are quoted
                        
                        # Fix trailing commas in arrays and objects
                        json_str = re.sub(r',\s*]', ']', json_str)
                        json_str = re.sub(r',\s*}', '}', json_str)
                        
                        try:
                            parsed_data = json.loads(json_str)
                            
                            # Check if the parsed data has our default data
                            if parsed_data.get("name") == "John Smith" and parsed_data.get("email") == "john@example.com":
                                print("Detected default example values in parsed JSON, rejecting")
                                return {"error": "Model returned example data instead of extraction", "raw_response": extracted_text[:200]}
                                
                            return parsed_data
                        except json.JSONDecodeError as e:
                            print(f"Still couldn't parse JSON after cleaning: {e}")
                            print(f"Cleaned JSON first 100 chars: {json_str[:100]}")
                    else:
                        print("No complete JSON pattern found, trying to extract fields directly")
                        
                    # If we reach here, try to extract individual fields
                    cv_data = {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "education": [],
                        "experience": [],
                        "skills": []
                    }
                    
                    # Extract fields with more robust patterns
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', extracted_text)
                    if name_match:
                        extracted_name = name_match.group(1)
                        # Skip if it looks like example data
                        if extracted_name != "John Smith" and extracted_name != "Extracted Name Here" and "Real name" not in extracted_name:
                            cv_data["name"] = extracted_name
                            print(f"Extracted name: {cv_data['name']}")
                    
                    email_match = re.search(r'"email"\s*:\s*"([^"]+)"', extracted_text)
                    if email_match:
                        extracted_email = email_match.group(1)
                        # Skip if it looks like example data
                        if extracted_email != "john@example.com" and extracted_email != "actual.email@fromcv.com" and "Real email" not in extracted_email:
                            cv_data["email"] = extracted_email
                            print(f"Extracted email: {cv_data['email']}")
                    
                    phone_match = re.search(r'"phone"\s*:\s*"([^"]+)"', extracted_text)
                    if phone_match:
                        extracted_phone = phone_match.group(1)
                        # Skip if it looks like example data
                        if extracted_phone != "123-456-7890" and extracted_phone != "Real phone from CV" and "Real phone" not in extracted_phone:
                            cv_data["phone"] = extracted_phone
                            print(f"Extracted phone: {cv_data['phone']}")
                    
                    # Extract skills
                    skills_match = re.search(r'"skills"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if skills_match:
                        skills_text = skills_match.group(1)
                        skills = re.findall(r'"([^"]+)"', skills_text)
                        # Filter out any that look like example data
                        filtered_skills = [s for s in skills if not s.startswith("Actual skill") and "Real skill" not in s]
                        if len(filtered_skills) > 0:
                            cv_data["skills"] = filtered_skills
                            print(f"Extracted {len(filtered_skills)} skills")
                    
                    # Extract education as simple strings
                    education_match = re.search(r'"education"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if education_match:
                        education_text = education_match.group(1)
                        education_items = re.findall(r'"([^"]+)"', education_text)
                        # Filter out any that look like example data
                        filtered_education = [e for e in education_items if not e.startswith("Actual education") and "Real education" not in e]
                        if len(filtered_education) > 0:
                            cv_data["education"] = filtered_education
                            print(f"Extracted {len(filtered_education)} education items")
                    
                    # Extract experience as simple strings
                    experience_match = re.search(r'"experience"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if experience_match:
                        experience_text = experience_match.group(1)
                        experience_items = re.findall(r'"([^"]+)"', experience_text)
                        # Filter out any that look like example data
                        filtered_experience = [e for e in experience_items if not e.startswith("Actual experience") and "Real experience" not in e]
                        if len(filtered_experience) > 0:
                            cv_data["experience"] = filtered_experience
                            print(f"Extracted {len(filtered_experience)} experience items")
                    
                    # Check if we extracted anything useful
                    if cv_data["name"] or cv_data["email"] or cv_data["phone"] or cv_data["skills"]:
                        print("Successfully extracted some data using regex fallback")
                        return cv_data
                    else:
                        print("Failed to extract any fields, response may be too incomplete")
                        # Return a structured error but with empty fields to avoid breaking the UI
                        cv_data["error"] = "Could not extract real data from CV"
                        return cv_data
                    
                except Exception as parse_error:
                    print(f"Error during JSON parsing/extraction: {str(parse_error)}")
                    # Return empty fields with error to avoid breaking the UI
                    return {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "education": [],
                        "experience": [],
                        "skills": [],
                        "error": f"JSON parsing error: {str(parse_error)}"
                    }
            except Exception as e:
                print(f"Exception in processing response: {str(e)}")
                # Return empty fields with error to avoid breaking the UI
                return {
                    "name": "",
                    "email": "",
                    "phone": "",
                    "education": [],
                    "experience": [],
                    "skills": [],
                    "error": str(e)
                }
        elif response.status_code == 404:
            # Specifically handle 404 error (model not found)
            print(f"Model '{model_name}' not found. Available models are:")
            try:
                models_response = requests.get("http://localhost:11434/api/tags")
                if models_response.status_code == 200:
                    models = models_response.json()
                    print(models)
                    return {"error": f"Model '{model_name}' not found. Please check model name."}
                else:
                    return {"error": "Model not found and couldn't retrieve available models."}
            except:
                return {"error": f"Model '{model_name}' not found. Please check if Ollama is running."}
        else:
            return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be unavailable or overloaded."}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error. The Ollama server may not be running."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Function to extract CV data using Mistral via Ollama
def run_mistral_extraction(text, timeout=DEFAULT_TIMEOUT):
    prompt = f"""
    EXTRACT INFORMATION FROM THIS CV AND FORMAT AS JSON.
    
    CRITICAL INSTRUCTIONS (FOLLOW PRECISELY):
    1. YOU MUST RETURN ONLY A VALID JSON OBJECT WITH DOUBLE QUOTES
    2. DO NOT RETURN ANY PYTHON CODE, FUNCTIONS, OR CLASSES
    3. DO NOT USE CODE BLOCKS OR MARKDOWN FORMAT. NO ```
    4. DO NOT RETURN IMPORT STATEMENTS
    5. DO NOT SUGGEST CODE OR FUNCTIONS TO PROCESS THE CV
    6. YOUR ENTIRE RESPONSE SHOULD BE *JUST* THE JSON OBJECT
    7. ONLY EXTRACT REAL DATA FROM THE CV TEXT
    8. DO NOT USE PLACEHOLDERS OR EXAMPLE DATA
    9. FIELDS MISSING FROM THE CV SHOULD BE EMPTY STRINGS OR ARRAYS
    
    Expected fields:
    - name: The person's full name 
    - email: Email address from the CV
    - phone: Phone number from the CV
    - education: List of education entries
    - experience: List of work experiences
    - skills: List of skills mentioned
    
    REQUIRED FORMAT (USE DOUBLE QUOTES, NOT SINGLE QUOTES):
    {{
      "name": "Real name from CV",
      "email": "Real email from CV",
      "phone": "Real phone from CV",
      "education": ["Real education 1", "Real education 2"],
      "experience": ["Real experience 1", "Real experience 2"],
      "skills": ["Real skill 1", "Real skill 2"]
    }}
    
    CV TEXT TO EXTRACT FROM:
    
    {text}
    
    REMINDER: RETURN ONLY THE JSON OBJECT WITH REAL DATA.
    NO CODE BLOCKS, NO PYTHON CODE, NO FUNCTIONS, NO MARKDOWN.
    YOUR ENTIRE RESPONSE SHOULD BE JUST THE JSON OBJECT AND NOTHING ELSE.
    """
    
    try:
        print("Running mistral extraction with Ollama...")
        model_name = "mistral"
        
        # Set a temperature parameter to reduce randomness and increase parameter settings
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Slight temperature to allow creativity but not too much
                    "num_predict": 2048,  # Increase token limit for complete response
                    "top_p": 0.9,        # Reduce randomness
                    "top_k": 30          # Focus on more likely tokens
                }
            },
            timeout=timeout
        )
        
        # Use the same error-resistant JSON extraction logic
        if response.status_code == 200:
            try:
                result = response.json()
                extracted_text = result.get("response", "")
                print(f"Raw response length: {len(extracted_text)}")
                print(f"Raw response first 100 chars: {extracted_text[:100]}")
                
                # More thorough check for Python code patterns
                code_indicators = [
                    "import ", "def ", "```", "class ", 
                    "print(", "return ", "function", 
                    "# Your code", "# This function", 
                    "if __name__", "for ", "while ",
                    "try:", "except:", " = function",
                    "@param", "params", "# Test"
                ]
                
                for indicator in code_indicators:
                    if indicator in extracted_text:
                        print(f"Detected code indicator: '{indicator}' in response")
                        return {"error": "Model returned Python code instead of JSON", "raw_response": extracted_text[:200]}
                
                # Extra cleaning to handle potential code blocks
                extracted_text = extracted_text.replace("```json", "").replace("```", "")
                
                # Check if response contains our example data which would indicate the model just repeated our example
                example_data_indicators = [
                    "Extracted Name Here", 
                    "actual.email@fromcv.com", 
                    "Real phone from CV",
                    "Actual education entry",
                    "Actual experience entry",
                    "Actual skill"
                ]
                
                for indicator in example_data_indicators:
                    if indicator in extracted_text:
                        print(f"Model returned example data ({indicator}), rejecting response")
                        return {"error": "Model returned example data instead of extraction", "raw_response": extracted_text[:200]}
                
                # Try to parse the JSON response
                try:
                    # First, try to find a complete JSON object with improved regex
                    json_match = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', extracted_text)
                    if json_match:
                        json_str = json_match.group(1)
                        print(f"Found JSON pattern, length: {len(json_str)}")
                        
                        # Clean the JSON
                        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)  # Ensure property names are quoted
                        
                        # Fix trailing commas in arrays and objects
                        json_str = re.sub(r',\s*]', ']', json_str)
                        json_str = re.sub(r',\s*}', '}', json_str)
                        
                        try:
                            parsed_data = json.loads(json_str)
                            
                            # Check if the parsed data has our default data
                            if parsed_data.get("name") == "John Smith" and parsed_data.get("email") == "john@example.com":
                                print("Detected default example values in parsed JSON, rejecting")
                                return {"error": "Model returned example data instead of extraction", "raw_response": extracted_text[:200]}
                                
                            return parsed_data
                        except json.JSONDecodeError as e:
                            print(f"Still couldn't parse JSON after cleaning: {e}")
                            print(f"Cleaned JSON first 100 chars: {json_str[:100]}")
                    else:
                        print("No complete JSON pattern found, trying to extract fields directly")
                        
                    # If we reach here, try to extract individual fields
                    cv_data = {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "education": [],
                        "experience": [],
                        "skills": []
                    }
                    
                    # Extract fields with more robust patterns
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', extracted_text)
                    if name_match:
                        extracted_name = name_match.group(1)
                        # Skip if it looks like example data
                        if extracted_name != "John Smith" and extracted_name != "Extracted Name Here" and "Real name" not in extracted_name:
                            cv_data["name"] = extracted_name
                            print(f"Extracted name: {cv_data['name']}")
                    
                    email_match = re.search(r'"email"\s*:\s*"([^"]+)"', extracted_text)
                    if email_match:
                        extracted_email = email_match.group(1)
                        # Skip if it looks like example data
                        if extracted_email != "john@example.com" and extracted_email != "actual.email@fromcv.com" and "Real email" not in extracted_email:
                            cv_data["email"] = extracted_email
                            print(f"Extracted email: {cv_data['email']}")
                    
                    phone_match = re.search(r'"phone"\s*:\s*"([^"]+)"', extracted_text)
                    if phone_match:
                        extracted_phone = phone_match.group(1)
                        # Skip if it looks like example data
                        if extracted_phone != "123-456-7890" and extracted_phone != "Real phone from CV" and "Real phone" not in extracted_phone:
                            cv_data["phone"] = extracted_phone
                            print(f"Extracted phone: {cv_data['phone']}")
                    
                    # Extract skills
                    skills_match = re.search(r'"skills"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if skills_match:
                        skills_text = skills_match.group(1)
                        skills = re.findall(r'"([^"]+)"', skills_text)
                        # Filter out any that look like example data
                        filtered_skills = [s for s in skills if not s.startswith("Actual skill") and "Real skill" not in s]
                        if len(filtered_skills) > 0:
                            cv_data["skills"] = filtered_skills
                            print(f"Extracted {len(filtered_skills)} skills")
                    
                    # Extract education as simple strings
                    education_match = re.search(r'"education"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if education_match:
                        education_text = education_match.group(1)
                        education_items = re.findall(r'"([^"]+)"', education_text)
                        # Filter out any that look like example data
                        filtered_education = [e for e in education_items if not e.startswith("Actual education") and "Real education" not in e]
                        if len(filtered_education) > 0:
                            cv_data["education"] = filtered_education
                            print(f"Extracted {len(filtered_education)} education items")
                    
                    # Extract experience as simple strings
                    experience_match = re.search(r'"experience"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if experience_match:
                        experience_text = experience_match.group(1)
                        experience_items = re.findall(r'"([^"]+)"', experience_text)
                        # Filter out any that look like example data
                        filtered_experience = [e for e in experience_items if not e.startswith("Actual experience") and "Real experience" not in e]
                        if len(filtered_experience) > 0:
                            cv_data["experience"] = filtered_experience
                            print(f"Extracted {len(filtered_experience)} experience items")
                    
                    # Check if we extracted anything useful
                    if cv_data["name"] or cv_data["email"] or cv_data["phone"] or cv_data["skills"]:
                        print("Successfully extracted some data using regex fallback")
                        return cv_data
                    else:
                        print("Failed to extract any fields, response may be too incomplete")
                        # Return a structured error but with empty fields to avoid breaking the UI
                        cv_data["error"] = "Could not extract real data from CV"
                        return cv_data
                    
                except Exception as parse_error:
                    print(f"Error during JSON parsing/extraction: {str(parse_error)}")
                    # Return empty fields with error to avoid breaking the UI
                    return {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "education": [],
                        "experience": [],
                        "skills": [],
                        "error": f"JSON parsing error: {str(parse_error)}"
                    }
            except Exception as e:
                print(f"Exception in processing response: {str(e)}")
                # Return empty fields with error to avoid breaking the UI
                return {
                    "name": "",
                    "email": "",
                    "phone": "",
                    "education": [],
                    "experience": [],
                    "skills": [],
                    "error": str(e)
                }
        elif response.status_code == 404:
            # Specifically handle 404 error (model not found)
            print(f"Model '{model_name}' not found. Available models are:")
            try:
                models_response = requests.get("http://localhost:11434/api/tags")
                if models_response.status_code == 200:
                    models = models_response.json()
                    print(models)
                    return {"error": f"Model '{model_name}' not found. Please check model name."}
                else:
                    return {"error": "Model not found and couldn't retrieve available models."}
            except:
                return {"error": f"Model '{model_name}' not found. Please check if Ollama is running."}
        else:
            return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be unavailable or overloaded."}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error. The Ollama server may not be running."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Function to extract CV data using Phi via Ollama
def run_phi2_extraction(text, timeout=DEFAULT_TIMEOUT):
    prompt = f"""
    EXTRACT INFORMATION FROM THIS CV AND FORMAT AS JSON.
    
    CRITICAL INSTRUCTIONS (FOLLOW PRECISELY):
    1. YOU MUST RETURN ONLY A VALID JSON OBJECT WITH DOUBLE QUOTES
    2. DO NOT RETURN ANY PYTHON CODE, FUNCTIONS, OR CLASSES
    3. DO NOT USE CODE BLOCKS OR MARKDOWN FORMAT. NO ```
    4. DO NOT RETURN IMPORT STATEMENTS
    5. DO NOT SUGGEST CODE OR FUNCTIONS TO PROCESS THE CV
    6. YOUR ENTIRE RESPONSE SHOULD BE *JUST* THE JSON OBJECT
    7. ONLY EXTRACT REAL DATA FROM THE CV TEXT
    8. DO NOT USE PLACEHOLDERS OR EXAMPLE DATA
    9. FIELDS MISSING FROM THE CV SHOULD BE EMPTY STRINGS OR ARRAYS
    
    Expected fields:
    - name: The person's full name 
    - email: Email address from the CV
    - phone: Phone number from the CV
    - education: List of education entries
    - experience: List of work experiences
    - skills: List of skills mentioned
    
    REQUIRED FORMAT (USE DOUBLE QUOTES, NOT SINGLE QUOTES):
    {{
      "name": "Real name from CV",
      "email": "Real email from CV",
      "phone": "Real phone from CV",
      "education": ["Real education 1", "Real education 2"],
      "experience": ["Real experience 1", "Real experience 2"],
      "skills": ["Real skill 1", "Real skill 2"]
    }}
    
    CV TEXT TO EXTRACT FROM:
    
    {text}
    
    REMINDER: RETURN ONLY THE JSON OBJECT WITH REAL DATA.
    NO CODE BLOCKS, NO PYTHON CODE, NO FUNCTIONS, NO MARKDOWN.
    YOUR ENTIRE RESPONSE SHOULD BE JUST THE JSON OBJECT AND NOTHING ELSE.
    """
    
    try:
        print("Running phi extraction with Ollama...")
        model_name = "phi"
        
        # Set a temperature parameter to reduce randomness and increase parameter settings
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Slight temperature to allow creativity but not too much
                    "num_predict": 2048,  # Increase token limit for complete response
                    "top_p": 0.9,        # Reduce randomness
                    "top_k": 30          # Focus on more likely tokens
                }
            },
            timeout=timeout
        )
        
        # Use the same error-resistant JSON extraction logic
        if response.status_code == 200:
            try:
                result = response.json()
                extracted_text = result.get("response", "")
                print(f"Raw response length: {len(extracted_text)}")
                print(f"Raw response first 100 chars: {extracted_text[:100]}")
                
                # More thorough check for Python code patterns
                code_indicators = [
                    "import ", "def ", "```", "class ", 
                    "print(", "return ", "function", 
                    "# Your code", "# This function", 
                    "if __name__", "for ", "while ",
                    "try:", "except:", " = function",
                    "@param", "params", "# Test"
                ]
                
                for indicator in code_indicators:
                    if indicator in extracted_text:
                        print(f"Detected code indicator: '{indicator}' in response")
                        return {"error": "Model returned Python code instead of JSON", "raw_response": extracted_text[:200]}
                
                # Extra cleaning to handle potential code blocks
                extracted_text = extracted_text.replace("```json", "").replace("```", "")
                
                # Check if response contains our example data which would indicate the model just repeated our example
                example_data_indicators = [
                    "Extracted Name Here", 
                    "actual.email@fromcv.com", 
                    "Real phone from CV",
                    "Actual education entry",
                    "Actual experience entry",
                    "Actual skill"
                ]
                
                for indicator in example_data_indicators:
                    if indicator in extracted_text:
                        print(f"Model returned example data ({indicator}), rejecting response")
                        return {"error": "Model returned example data instead of extraction", "raw_response": extracted_text[:200]}
                
                # Try to parse the JSON response
                try:
                    # First, try to find a complete JSON object with improved regex
                    json_match = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', extracted_text)
                    if json_match:
                        json_str = json_match.group(1)
                        print(f"Found JSON pattern, length: {len(json_str)}")
                        
                        # Clean the JSON
                        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)  # Ensure property names are quoted
                        
                        # Fix trailing commas in arrays and objects
                        json_str = re.sub(r',\s*]', ']', json_str)
                        json_str = re.sub(r',\s*}', '}', json_str)
                        
                        try:
                            parsed_data = json.loads(json_str)
                            
                            # Check if the parsed data has our default data
                            if parsed_data.get("name") == "John Smith" and parsed_data.get("email") == "john@example.com":
                                print("Detected default example values in parsed JSON, rejecting")
                                return {"error": "Model returned example data instead of extraction", "raw_response": extracted_text[:200]}
                                
                            return parsed_data
                        except json.JSONDecodeError as e:
                            print(f"Still couldn't parse JSON after cleaning: {e}")
                            print(f"Cleaned JSON first 100 chars: {json_str[:100]}")
                    else:
                        print("No complete JSON pattern found, trying to extract fields directly")
                        
                    # If we reach here, try to extract individual fields
                    cv_data = {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "education": [],
                        "experience": [],
                        "skills": []
                    }
                    
                    # Extract fields with more robust patterns
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', extracted_text)
                    if name_match:
                        extracted_name = name_match.group(1)
                        # Skip if it looks like example data
                        if extracted_name != "John Smith" and extracted_name != "Extracted Name Here" and "Real name" not in extracted_name:
                            cv_data["name"] = extracted_name
                            print(f"Extracted name: {cv_data['name']}")
                    
                    email_match = re.search(r'"email"\s*:\s*"([^"]+)"', extracted_text)
                    if email_match:
                        extracted_email = email_match.group(1)
                        # Skip if it looks like example data
                        if extracted_email != "john@example.com" and extracted_email != "actual.email@fromcv.com" and "Real email" not in extracted_email:
                            cv_data["email"] = extracted_email
                            print(f"Extracted email: {cv_data['email']}")
                    
                    phone_match = re.search(r'"phone"\s*:\s*"([^"]+)"', extracted_text)
                    if phone_match:
                        extracted_phone = phone_match.group(1)
                        # Skip if it looks like example data
                        if extracted_phone != "123-456-7890" and extracted_phone != "Real phone from CV" and "Real phone" not in extracted_phone:
                            cv_data["phone"] = extracted_phone
                            print(f"Extracted phone: {cv_data['phone']}")
                    
                    # Extract skills
                    skills_match = re.search(r'"skills"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if skills_match:
                        skills_text = skills_match.group(1)
                        skills = re.findall(r'"([^"]+)"', skills_text)
                        # Filter out any that look like example data
                        filtered_skills = [s for s in skills if not s.startswith("Actual skill") and "Real skill" not in s]
                        if len(filtered_skills) > 0:
                            cv_data["skills"] = filtered_skills
                            print(f"Extracted {len(filtered_skills)} skills")
                    
                    # Extract education as simple strings
                    education_match = re.search(r'"education"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if education_match:
                        education_text = education_match.group(1)
                        education_items = re.findall(r'"([^"]+)"', education_text)
                        # Filter out any that look like example data
                        filtered_education = [e for e in education_items if not e.startswith("Actual education") and "Real education" not in e]
                        if len(filtered_education) > 0:
                            cv_data["education"] = filtered_education
                            print(f"Extracted {len(filtered_education)} education items")
                    
                    # Extract experience as simple strings
                    experience_match = re.search(r'"experience"\s*:\s*\[(.*?)\]', extracted_text, re.DOTALL)
                    if experience_match:
                        experience_text = experience_match.group(1)
                        experience_items = re.findall(r'"([^"]+)"', experience_text)
                        # Filter out any that look like example data
                        filtered_experience = [e for e in experience_items if not e.startswith("Actual experience") and "Real experience" not in e]
                        if len(filtered_experience) > 0:
                            cv_data["experience"] = filtered_experience
                            print(f"Extracted {len(filtered_experience)} experience items")
                    
                    # Check if we extracted anything useful
                    if cv_data["name"] or cv_data["email"] or cv_data["phone"] or cv_data["skills"]:
                        print("Successfully extracted some data using regex fallback")
                        return cv_data
                    else:
                        print("Failed to extract any fields, response may be too incomplete")
                        # Return a structured error but with empty fields to avoid breaking the UI
                        cv_data["error"] = "Could not extract real data from CV"
                        return cv_data
                    
                except Exception as parse_error:
                    print(f"Error during JSON parsing/extraction: {str(parse_error)}")
                    # Return empty fields with error to avoid breaking the UI
                    return {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "education": [],
                        "experience": [],
                        "skills": [],
                        "error": f"JSON parsing error: {str(parse_error)}"
                    }
            except Exception as e:
                print(f"Exception in processing response: {str(e)}")
                # Return empty fields with error to avoid breaking the UI
                return {
                    "name": "",
                    "email": "",
                    "phone": "",
                    "education": [],
                    "experience": [],
                    "skills": [],
                    "error": str(e)
                }
        elif response.status_code == 404:
            # Specifically handle 404 error (model not found)
            print(f"Model '{model_name}' not found. Available models are:")
            try:
                models_response = requests.get("http://localhost:11434/api/tags")
                if models_response.status_code == 200:
                    models = models_response.json()
                    print(models)
                    return {"error": f"Model '{model_name}' not found. Please check model name."}
                else:
                    return {"error": "Model not found and couldn't retrieve available models."}
            except:
                return {"error": f"Model '{model_name}' not found. Please check if Ollama is running."}
        else:
            return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be unavailable or overloaded."}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error. The Ollama server may not be running."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Function to select and run the appropriate LLM
def extract_with_llm(text, model_name, max_retries=2):
    # Define model-specific timeouts (larger models get more time)
    model_timeouts = {
        'llama3': 360,  # 6 minutes for the largest model
        'mistral': 300, # 5 minutes for medium-sized model
        'phi': 240      # 4 minutes for smallest model
    }
    
    # Get the appropriate timeout for this model
    timeout = model_timeouts.get(model_name, DEFAULT_TIMEOUT)
    print(f"Using {timeout} second timeout for {model_name} model")
    
    # Try the requested model first
    result = None
    error_message = ""
    
    # Try the requested model with retries
    retries = 0
    while retries <= max_retries:
        try:
            if model_name == 'llama3':
                result = run_llama3_extraction(text, timeout=timeout)
            elif model_name == 'mistral':
                result = run_mistral_extraction(text, timeout=timeout)
            elif model_name == 'phi':
                # Always use run_phi2_extraction for 'phi'
                result = run_phi2_extraction(text, timeout=timeout)
            else:
                raise ValueError(f'Invalid model name: {model_name}. Available models: llama3, mistral, phi')
            
            # Check if there was an error in the extraction
            if result and isinstance(result, dict) and "error" in result:
                error_message = result.get("error", "Unknown error")
                print(f"Extraction error with {model_name}: {error_message}")
                retries += 1
                if retries <= max_retries:
                    wait_time = 2 * retries
                    print(f"Waiting {wait_time} seconds before retry {retries}/{max_retries}...")
                    time.sleep(wait_time)
                continue  # Try again with the same model
            
            # If we get here with a result, it means success
            if result:
                return result
                
        except Exception as e:
            retries += 1
            error_message = str(e)
            print(f"Attempt {retries}/{max_retries+1} failed with error: {error_message}")
            
            if retries <= max_retries:
                # Wait longer between each retry
                wait_time = 2 * retries
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            
    # If we're here, the requested model failed after all retries
    # Try other models in order of reliability
    fallback_models = ['llama3', 'mistral', 'phi']
    if model_name in fallback_models:
        fallback_models.remove(model_name)  # Don't try the already-failed model again
    
    for fallback_model in fallback_models:
        try:
            print(f"Trying {fallback_model} as fallback after {model_name} failed with: {error_message}")
            
            if fallback_model == 'llama3':
                result = run_llama3_extraction(text, timeout=model_timeouts['llama3'])
            elif fallback_model == 'mistral':
                result = run_mistral_extraction(text, timeout=model_timeouts['mistral'])
            elif fallback_model == 'phi':
                result = run_phi2_extraction(text, timeout=model_timeouts['phi'])
            
            # Check if there was an error in the extraction
            if result and isinstance(result, dict) and "error" in result:
                print(f"Fallback to {fallback_model} also failed with: {result.get('error')}")
                continue  # Try the next model
            
            # If we get here with a result, it means success with fallback
            if result:
                print(f"Successfully extracted data using {fallback_model} as fallback")
                return result
                
        except Exception as e:
            print(f"Fallback to {fallback_model} failed with exception: {str(e)}")
    
    # If we reach here, all models have failed
    return {
        "name": "",
        "email": "",
        "phone": "",
        "education": [],
        "experience": [],
        "skills": [],
        "error": f"All models failed. Last error: {error_message}"
    } 