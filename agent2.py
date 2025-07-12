import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

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
        return """You are an expert web developer specializing in creating modern, responsive websites using HTML5, CSS3, and JavaScript.

Guidelines:
1. Generate semantic HTML5 with proper structure and accessibility
2. Create modern, responsive CSS with clean styling
3. Use JavaScript for interactive behavior (e.g., theme toggle, form validation, lightbox)
4. Use CSS Grid and Flexbox for layouts
5. Ensure mobile-first responsive design
6. Include proper meta tags, viewport settings, and SEO elements
7. Follow web accessibility (WCAG) guidelines
8. Add meaningful comments in the code
9. Ensure cross-browser compatibility
10. Prefer clean, modular JavaScript with no frameworks unless specified

Color Palette Guidelines:
- Use the specified colors from the prompt
- Maintain contrast ratios for accessibility
- Apply colors consistently throughout the design

Output Format:
Provide your response with separate HTML, CSS, and JS code blocks:

```html
[Complete HTML5 code here]
```

```css
[Complete CSS3 code here]
```

```js
[Complete JavaScript code here]
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
        html_pattern = r'```html\s*\n(.*?)\n```'
        css_pattern = r'```css\s*\n(.*?)\n```'
        js_pattern = r'```(?:js|javascript)\s*\n(.*?)\n```'

        html_match = re.search(html_pattern, response_text, re.DOTALL | re.IGNORECASE)
        css_match = re.search(css_pattern, response_text, re.DOTALL | re.IGNORECASE)
        js_match = re.search(js_pattern, response_text, re.DOTALL | re.IGNORECASE)

        html_code = html_match.group(1).strip() if html_match else ""
        css_code = css_match.group(1).strip() if css_match else ""
        js_code = js_match.group(1).strip() if js_match else ""

        if not html_code and not css_code and not js_code:
            print("Warning: No code blocks found. Attempting to extract from full response...")
            return self.fallback_code_extraction(response_text)

        return html_code, css_code, js_code
    
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
            
            ai_response = response.choices[0].message.content
            return ai_response
            
        except Exception as e:
            raise RuntimeError(f"Error generating code: {str(e)}")
    
    def generate_output_filename(self, json_file_path):
        """Generate a unique output filename with timestamp"""
        # Extract base name from input JSON file
        base_name = Path(json_file_path).stem
        
        # Remove timestamp if already present in the input filename
        if '_' in base_name:
            # Split and take the first part (before the first underscore)
            parts = base_name.split('_')
            if len(parts) > 1:
                base_name = parts[0]
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        output_filename = f"{base_name}_generated_{timestamp}.json"
        return output_filename
    
    def save_code_to_json(self, html_code, css_code, js_code, prompt_data, ai_response, json_file_path=None):
        """Save HTML and CSS code to a single JSON file"""
        
        # Generate output filename
        output_filename = "generated_code.json"
        json_output_path = Path(output_filename)
        
        # Prepare the JSON structure
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source_prompt_file": str(json_file_path) if json_file_path else None,
                "model_used": self.model,
                "generator": "OpenAI Coding Agent v2.0"
            },
            "html": html_code,
            "css": css_code,
            "js": js_code,
            "prompt_data": prompt_data,
            "ai_response": ai_response,
            "statistics": {
                "html_length": len(html_code),
                "css_length": len(css_code),
                "js_length": len(js_code),
                "total_characters": len(html_code) + len(css_code) + len(js_code)
            }
        }
        
        # Save to JSON file
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"Code generated and saved to: {json_output_path}")
            return str(json_output_path), output_data
            
        except Exception as e:
            raise RuntimeError(f"Error saving JSON file: {str(e)}")
    
    def process_json_to_code(self, json_file_path):
        """Complete pipeline: JSON -> AI Generation -> JSON Output"""
        
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
        html_code, css_code, js_code = self.extract_code_blocks(ai_response)
        
        # Validate generated code
        if not html_code and not css_code and not js_code:
            raise ValueError("No valid HTML, CSS, or JS code was generated from the AI response")

        
        # Save to JSON file
        print("Saving generated code to JSON file...")
        json_output_path, output_data = self.save_code_to_json(
        html_code, css_code, js_code, prompt_data, ai_response, json_file_path=json_file_path)
        
        return {
            'output_file': json_output_path,
            'html_code': html_code,
            'css_code': css_code,
            'ai_response': ai_response,
            'prompt_used': enhanced_prompt,
            'metadata': output_data['metadata'],
            'statistics': output_data['statistics']
        }
    
    def extract_and_save_separate_files(self, json_output_path, output_dir=None):
        """Extract HTML and CSS from generated JSON and save as separate files (optional utility)"""
        
        if output_dir is None:
            output_dir = Path(json_output_path).parent
        
        # Load the JSON file
        with open(json_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract HTML and CSS
        html_code = data.get('html', '')
        css_code = data.get('css', '')
        js_code = data.get('js', '')
        
        if not html_code and not css_code:
            print("No HTML or CSS content found in the JSON file")
            return
        
        # Generate base filename from JSON filename
        base_name = Path(json_output_path).stem.replace('_generated_', '_')
        
        # Save HTML file
        if html_code:
            html_file = Path(output_dir) / f"{base_name}.html"
            
            # Add CSS link if not present
            css_filename = f"{base_name}.css"
            if css_filename not in html_code and "stylesheet" not in html_code:
                css_link = f'    <link rel="stylesheet" href="{css_filename}">\n'
                if "<head>" in html_code:
                    html_code = html_code.replace("<head>", f"<head>\n{css_link}")
                elif "</head>" in html_code:
                    html_code = html_code.replace("</head>", f"{css_link}</head>")

            js_filename = f"{base_name}.js"
            if js_filename not in html_code and "script" not in html_code:
                js_script = f'    <script src="{js_filename}"></script>\n'
                if "</body>" in html_code:
                    html_code = html_code.replace("</body>", f"{js_script}</body>")
                else:
                    html_code += f"\n{js_script}"

            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_code)
            print(f"HTML file extracted to: {html_file}")
        
        # Save CSS file
        if css_code:
            css_file = Path(output_dir) / f"{base_name}.css"
            with open(css_file, 'w', encoding='utf-8') as f:
                f.write(css_code)
            print(f"CSS file extracted to: {css_file}")

        if js_code:
            js_file = Path(output_dir) / f"{base_name}.js"
            with open(js_file, 'w', encoding='utf-8') as f:
                f.write(js_code)
            print(f"JS file extracted to: {js_file}")



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
    json_file = "enhanced_prompt.json"
    
    try:
        result = agent.process_json_to_code(
            json_file_path=json_file
        )
        
        print("\n" + "="*60)
        print("CODE GENERATION COMPLETED!")
        print("="*60)
        print(f"Generated JSON file: {result['output_file']}")
        print(f"HTML code length: {result['statistics']['html_length']} characters")
        print(f"CSS code length: {result['statistics']['css_length']} characters")
        print(f"Total characters: {result['statistics']['total_characters']}")
        print(f"Generated at: {result['metadata']['generated_at']}")
        
        # Display a preview of the generated code
        if result['html_code']:
            print("\n--- HTML Preview (first 200 chars) ---")
            print(result['html_code'][:200] + "..." if len(result['html_code']) > 200 else result['html_code'])
        
        if result['css_code']:
            print("\n--- CSS Preview (first 200 chars) ---")
            print(result['css_code'][:200] + "..." if len(result['css_code']) > 200 else result['css_code'])
        
        # Optional: Extract separate HTML/CSS files
        print(f"\nTo extract separate HTML/CSS files, run:")
        print(f"agent.extract_and_save_separate_files('{result['output_file']}')")
        
    except FileNotFoundError:
        print(f"JSON file '{json_file}' not found.")
        print("Please make sure the enhanced_homepage_prompt JSON file is in the current directory.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()