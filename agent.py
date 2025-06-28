import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAICodingAgent:
    def __init__(self):
        """Initialize the coding agent with OpenAI GitHub marketplace credentials"""
        self.token = os.getenv("GITHUB_TOKEN")
        self.endpoint = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1"
        
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
    
    def load_prompt_from_json(self, json_file_path):
        """Load the refined prompt from a JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_file_path}")
    
    def create_system_prompt(self):
        """Create a comprehensive system prompt for HTML/CSS generation"""
        return """You are an expert web developer specializing in creating modern, responsive websites using HTML5 and CSS3. Your task is to convert project requirements into clean, functional code.

Guidelines:
1. Generate semantic HTML5 with proper structure and accessibility
2. Create modern, responsive CSS with clean styling
3. Use CSS Grid and Flexbox for layouts
4. Ensure mobile-first responsive design
5. Include proper meta tags, viewport settings, and SEO elements
6. Use modern CSS features (custom properties, transitions, etc.)
7. Follow web accessibility (WCAG) guidelines
8. Add meaningful comments in the code
9. Ensure cross-browser compatibility
10. If the requirements mention React, convert them to equivalent HTML/CSS structure

Color Palette Guidelines:
- Use the specified colors: #000000 (black), #ffffff (white), #808080 (gray)
- Maintain contrast ratios for accessibility
- Apply colors consistently throughout the design

Output Format:
Provide your response with separate HTML and CSS code blocks:

```html
[Complete HTML5 code here]
```

```css
[Complete CSS3 code here]
```

Make sure the HTML references the CSS file correctly and both work together seamlessly."""
    
    def extract_enhanced_prompt(self, json_data):
        """Extract the enhanced prompt from the JSON data"""
        if isinstance(json_data, dict):
            # Try to get enhanced_prompt first, then fall back to other fields
            enhanced_prompt = json_data.get('enhanced_prompt', '')
            if enhanced_prompt:
                return enhanced_prompt
            
            # If no enhanced_prompt, construct from available fields
            prompt_parts = []
            
            if 'original_prompt' in json_data:
                prompt_parts.append(f"Original Request: {json_data['original_prompt']}")
            
            if 'features' in json_data:
                features_str = ', '.join(json_data['features'])
                prompt_parts.append(f"Required Features: {features_str}")
            
            if 'export_type' in json_data:
                # Convert React requirement to HTML/CSS
                export_type = json_data['export_type']
                if export_type.lower() == 'react':
                    prompt_parts.append("Convert this to a modern HTML/CSS website (not React)")
                else:
                    prompt_parts.append(f"Export Type: {export_type}")
            
            return '\n\n'.join(prompt_parts)
        
        return str(json_data)
    
    def extract_code_blocks(self, response_text):
        """Extract HTML and CSS code blocks from the AI response"""
        html_pattern = r'```html\s*\n(.*?)\n```'
        css_pattern = r'```css\s*\n(.*?)\n```'
        
        html_match = re.search(html_pattern, response_text, re.DOTALL | re.IGNORECASE)
        css_match = re.search(css_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        html_code = html_match.group(1).strip() if html_match else ""
        css_code = css_match.group(1).strip() if css_match else ""
        
        # If no code blocks found, try to extract from the entire response
        if not html_code and not css_code:
            print("Warning: No code blocks found. Attempting to extract from full response...")
            return self.fallback_code_extraction(response_text)
        
        return html_code, css_code
    
    def fallback_code_extraction(self, response_text):
        """Fallback method to extract code if standard patterns don't work"""
        # Look for HTML doctype or html tags
        html_pattern = r'(<!DOCTYPE.*?</html>)'
        html_match = re.search(html_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        # Look for CSS selectors and rules
        css_pattern = r'(\*\s*{.*?}.*?)(?=<!DOCTYPE|$)'
        css_match = re.search(css_pattern, response_text, re.DOTALL)
        
        html_code = html_match.group(1).strip() if html_match else ""
        css_code = css_match.group(1).strip() if css_match else ""
        
        return html_code, css_code
    
    def generate_code(self, prompt_data, temperature=0.7, top_p=0.9):
        """Generate HTML and CSS code based on the prompt data"""
        
        # Extract the enhanced prompt from JSON
        user_prompt = self.extract_enhanced_prompt(prompt_data)
        
        # Add specific instruction for HTML/CSS output
        user_prompt += "\n\nPlease create this as a complete HTML/CSS website with separate HTML and CSS files. Make sure to use modern web standards and responsive design."
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.create_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                model=self.model
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Error generating code: {str(e)}")
    
    def save_files(self, html_code, css_code, output_dir="output", filename_base="generated"):
        """Save HTML and CSS code to separate files"""
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Define file paths
        html_file = output_path / f"{filename_base}.html"
        css_file = output_path / f"{filename_base}.css"
        
        # Save HTML file (and add CSS link if not present)
        if html_code:
            # Check if HTML already links to CSS, if not add it
            if f"{filename_base}.css" not in html_code and "stylesheet" not in html_code:
                # Add CSS link to head section
                css_link = f'    <link rel="stylesheet" href="{filename_base}.css">\n'
                if "<head>" in html_code:
                    html_code = html_code.replace("<head>", f"<head>\n{css_link}")
                elif "</head>" in html_code:
                    html_code = html_code.replace("</head>", f"{css_link}</head>")
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_code)
            print(f"HTML file saved: {html_file}")
        else:
            print("Warning: No HTML code to save")
        
        # Save CSS file
        if css_code:
            with open(css_file, 'w', encoding='utf-8') as f:
                f.write(css_code)
            print(f"CSS file saved: {css_file}")
        else:
            print("Warning: No CSS code to save")
        
        return str(html_file), str(css_file)
    
    def process_json_to_code(self, json_file_path, output_dir="output", filename_base=None):
        """Complete pipeline: JSON -> AI Generation -> File Output"""
        
        # Load prompt from JSON
        print(f"Loading prompt from: {json_file_path}")
        prompt_data = self.load_prompt_from_json(json_file_path)
        
        # Extract and display the prompt being used
        enhanced_prompt = self.extract_enhanced_prompt(prompt_data)
        print(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
        
        # Generate code using AI
        print("Generating code with OpenAI GPT-4.1...")
        ai_response = self.generate_code(prompt_data)
        
        # Extract HTML and CSS from response
        html_code, css_code = self.extract_code_blocks(ai_response)
        
        # Auto-generate filename if not provided
        if filename_base is None:
            filename_base = Path(json_file_path).stem
        
        # Save files
        print("Saving generated files...")
        html_path, css_path = self.save_files(html_code, css_code, output_dir, filename_base)
        
        return {
            'html_path': html_path,
            'css_path': css_path,
            'html_code': html_code,
            'css_code': css_code,
            'ai_response': ai_response,
            'prompt_used': enhanced_prompt
        }


def main():
    """Example usage of the OpenAI Coding Agent"""
    
    # Initialize the agent
    try:
        agent = OpenAICodingAgent()
        print("OpenAI Coding Agent initialized successfully!")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure GITHUB_TOKEN is set in your environment variables")
        return
    
    # Process the provided JSON file
    json_file = "enhanced_homepage_prompt.json"
    
    try:
        result = agent.process_json_to_code(
            json_file_path=json_file,
            output_dir="photographer_portfolio",
            filename_base="photographer_portfolio"
        )
        
        print("\n" + "="*60)
        print("CODE GENERATION COMPLETED!")
        print("="*60)
        print(f"HTML saved to: {result['html_path']}")
        print(f"CSS saved to: {result['css_path']}")
        print(f"HTML code length: {len(result['html_code'])} characters")
        print(f"CSS code length: {len(result['css_code'])} characters")
        
        # Display a preview of the generated code
        if result['html_code']:
            print("\n--- HTML Preview (first 200 chars) ---")
            print(result['html_code'][:200] + "..." if len(result['html_code']) > 200 else result['html_code'])
        
        if result['css_code']:
            print("\n--- CSS Preview (first 200 chars) ---")
            print(result['css_code'][:200] + "..." if len(result['css_code']) > 200 else result['css_code'])
        
    except FileNotFoundError:
        print(f"JSON file '{json_file}' not found.")
        print("Please make sure the enhanced_homepage_prompt.json file is in the current directory.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()