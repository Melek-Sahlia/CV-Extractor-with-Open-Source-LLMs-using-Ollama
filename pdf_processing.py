import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import requests
import os
import base64
import json

# Function to extract text from text-based PDFs
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to determine if a PDF page contains text
def has_text(page):
    text = page.get_text()
    return len(text.strip()) > 10  # Arbitrary threshold

# Function to extract text from image-based PDFs using Tesseract OCR
def extract_text_from_image_pdf_tesseract(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            if not has_text(page):  # If the page doesn't have text, use OCR
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                text += pytesseract.image_to_string(img)
            else:
                text += page.get_text()
    return text

# Function to extract text from image-based PDFs using a multimodal LLM via Ollama
def extract_text_from_image_pdf_llm(file_path, model_name="llava"):
    text = ""
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    
    # Create a temp directory if it doesn't exist
    temp_dir = "temp_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            if not has_text(page):  # If the page doesn't have text, use OCR
                try:
                    # Save the page as an image
                    pix = page.get_pixmap()
                    img_path = os.path.join(temp_dir, f"temp_image_{page_num}.png")
                    pix.save(img_path)
                    
                    # Convert the image to a base64 string
                    with open(img_path, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Call the multimodal LLM via Ollama
                    prompt = f"""
                    This is a page from a CV/resume. 
                    Perform OCR to extract all the text from this image.
                    Return only the extracted text, no additional comments.
                    
                    <image>
                    data:image/png;base64,{img_base64}
                    </image>
                    """
                    
                    try:
                        response = requests.post(
                            OLLAMA_API_URL,
                            json={
                                "model": model_name,
                                "prompt": prompt,
                                "stream": False
                            },
                            timeout=30  # Add timeout
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            page_text = result.get("response", "")
                            text += page_text
                        else:
                            # Fallback to Tesseract OCR if the LLM call fails
                            img = Image.open(img_path)
                            text += pytesseract.image_to_string(img)
                    except requests.exceptions.RequestException as e:
                        print(f"Error calling Ollama API: {e}")
                        # Fallback to Tesseract OCR
                        img = Image.open(img_path)
                        text += pytesseract.image_to_string(img)
                    
                    # Clean up temporary image file
                    if os.path.exists(img_path):
                        os.remove(img_path)
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    # Continue with the text we have from the page
                    text += page.get_text() or f"[Error processing page {page_num}]"
            else:
                text += page.get_text()
    
    return text

# Function to extract text from a PDF, choosing the appropriate method
def extract_text(file_path, use_mistral_ocr=True, ocr_model="llava"):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
            
        # First, try to extract text directly
        text = extract_text_from_pdf(file_path)
        
        # If we didn't get much text, it's probably an image-based PDF
        if len(text.strip()) < 100:  # Arbitrary threshold
            if use_mistral_ocr:
                try:
                    return extract_text_from_image_pdf_llm(file_path, ocr_model)
                except Exception as e:
                    print(f"Error using LLM OCR: {e}")
                    print("Falling back to Tesseract OCR")
                    return extract_text_from_image_pdf_tesseract(file_path)
            else:
                return extract_text_from_image_pdf_tesseract(file_path)
        
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return f"Error extracting text: {e}" 