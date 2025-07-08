"""
AI Code Organizer Agent - Enhanced Version

This script processes LLM-generated code output, detects file types, extracts filenames,
and organizes code into proper folder structures according to a predefined file structure.
It also directly integrates with the GitHub AI model API.

Features:
- Parse code blocks (HTML, CSS, JS, etc.) from LLM output
- Auto-organize files into folders
- Extract or infer filenames from comments
- Create a Git repo (optional)
- Zip output folder for download or share
- Fully offline and open source (MIT License)
- Direct integration with GitHub's GPT-4.1 model
- Modular, beginner-friendly structure

License: MIT
"""

import os
import re
import json
import zipfile
import argparse
from typing import Dict, List, Optional, Tuple, Any
import subprocess
from dotenv import load_dotenv
from openai import OpenAI


class CodeOrganizer:
    """
    Main class for organizing code from LLM output into proper folder structures.
    """
    
    def __init__(self, output_dir: str = "organized_code"):
        """
        Initialize the CodeOrganizer.
        
        Args:
            output_dir: Directory where organized code will be saved
        """
        self.output_dir = output_dir
        self.file_structure = {}
        self.extracted_code = {}
        
    def load_file_structure(self, structure_json: str) -> None:
        """
        Load the file structure from a JSON string.
        
        Args:
            structure_json: JSON string containing the file structure
        """
        try:
            self.file_structure = json.loads(structure_json)
            print(f"Loaded file structure with {len(self.file_structure)} entries")
        except json.JSONDecodeError as e:
            print(f"Error parsing file structure JSON: {e}")
            raise
    
    def load_file_structure_from_file(self, file_path: str) -> None:
        """
        Load the file structure from a file, supporting both JSON and Markdown formats.
        
        Args:
            file_path: Path to the file containing the structure
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if it's JSON or Markdown based on file extension
            if file_path.lower().endswith('.json'):
                self.load_file_structure(content)
            elif file_path.lower().endswith('.md'):
                self.load_markdown_structure(content)
            else:
                # Try to determine format based on content
                try:
                    # Try parsing as JSON
                    json.loads(content)
                    self.load_file_structure(content)
                except json.JSONDecodeError:
                    # If not JSON, assume it's markdown
                    self.load_markdown_structure(content)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise
    
    def parse_llm_output(self, llm_output: str) -> Dict[str, str]:
        """
        Parse LLM output to extract code blocks and their file types.
        
        Args:
            llm_output: The raw output from the LLM
            
        Returns:
            Dictionary mapping filenames to code content
        """
        # Dictionary to store extracted code by filename
        extracted_files = {}
        
        # Regular expression to find code blocks with filename in comments or markdown format
        patterns = [
            # Match markdown code blocks with filename in the format: ```html filename.html
            r'```(\w+)\s+([^\n]+)\n(.*?)```',
            
            # Match markdown code blocks with filename in header format: # filename.html
            r'#+\s+([^#\n]+\.[\w]+)\s*\n+```(\w+)?\n(.*?)```',
            
            # Match standard markdown code blocks and infer filename from language: ```html
            r'```(\w+)\n(.*?)```',
            
            # Match filename in comments: // filename.js or # filename.py
            r'(?:\/\/|#)\s*filename:\s*([^\n]+)\n(.*?)(?=(?:\/\/|#)\s*filename|\Z)',
            
            # Match HTML file indicators
            r'<!DOCTYPE html>[\s\S]*?<html[^>]*>([\s\S]*?)<\/html>',
            
            # Match CSS file indicators
            r'\/\*\s*filename:\s*([^\n]+\.css)\s*\*\/([\s\S]*?)(?=\/\*\s*filename|\Z)',
            
            # Match JS file indicators
            r'\/\/\s*filename:\s*([^\n]+\.js)([\s\S]*?)(?=\/\/\s*filename|\Z)',
        ]
        
        # Process markdown code blocks with filename syntax: ```html filename.html
        for match in re.finditer(patterns[0], llm_output, re.DOTALL | re.MULTILINE):
            lang_type, filename, content = match.groups()
            # Add extension if missing
            if not filename.endswith(f".{lang_type}") and "." not in filename:
                filename = f"{filename}.{lang_type}"
            extracted_files[filename.strip()] = content.strip()
        
        # Process markdown header style: # filename.html followed by ```code```
        for match in re.finditer(patterns[1], llm_output, re.DOTALL | re.MULTILINE):
            filename, lang_type, content = match.groups()
            extracted_files[filename.strip()] = content.strip()
        
        # Process standard markdown code blocks: ```html
        code_block_count = {
            "html": 0,
            "css": 0,
            "js": 0,
            "python": 0,
            "py": 0
        }
        
        for match in re.finditer(patterns[2], llm_output, re.DOTALL | re.MULTILINE):
            lang_type, content = match.groups()
            lang_type = lang_type.lower()
            
            # Skip if it's not a code language
            if lang_type in ["plaintext", "text", "txt", "markdown", "md"]:
                continue
                
            # Handle different language types
            if lang_type in ["html", "htm", "xml"]:
                filename = "index.html" if code_block_count["html"] == 0 else f"page{code_block_count['html']}.html"
                code_block_count["html"] += 1
            elif lang_type in ["css", "scss", "sass"]:
                filename = "styles.css" if code_block_count["css"] == 0 else f"styles{code_block_count['css']}.css"
                code_block_count["css"] += 1
            elif lang_type in ["js", "javascript", "jsx", "ts", "typescript"]:
                filename = "script.js" if code_block_count["js"] == 0 else f"script{code_block_count['js']}.js"
                code_block_count["js"] += 1
            elif lang_type in ["python", "py"]:
                filename = "script.py" if code_block_count["py"] == 0 else f"script{code_block_count['py']}.py"
                code_block_count["py"] += 1
            else:
                # For other languages, use a generic naming pattern
                filename = f"file.{lang_type}"
            
            extracted_files[filename] = content.strip()
        
        # Process the rest of the patterns
        for pattern in patterns[3:]:
            for match in re.finditer(pattern, llm_output, re.DOTALL | re.MULTILINE):
                if len(match.groups()) >= 2:
                    filename, content = match.groups()
                    extracted_files[filename.strip()] = content.strip()
                elif pattern.startswith('<!DOCTYPE'):
                    # Special case for HTML files without explicit filename
                    extracted_files["index.html"] = match.group(0).strip()
        
        self.extracted_code = extracted_files
        return extracted_files
    
    def infer_file_type(self, content: str) -> str:
        """
        Infer the file type from the content if not explicitly specified.
        
        Args:
            content: The code content
            
        Returns:
            Inferred file type (extension)
        """
        # Check for HTML indicators
        if re.search(r'<!DOCTYPE html>|<html>|<body>|<head>', content, re.IGNORECASE):
            return "html"
        
        # Check for CSS indicators
        if re.search(r'{[^}]*?:[^}]*?}', content) and not re.search(r'function|var|let|const', content):
            return "css"
        
        # Check for JavaScript indicators
        if re.search(r'function|var|let|const|=>|import|export', content):
            return "js"
        
        # Check for Python indicators
        if re.search(r'def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import', content):
            return "py"
        
        # Default to txt if we can't determine
        return "txt"
    
    def organize_files(self) -> None:
        """
        Organize extracted code files according to the loaded file structure.
        Creates directories and writes files as needed.
        """
        if not self.extracted_code:
            print("No extracted code. Make sure to parse LLM output first.")
            return
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track which files we've processed
        processed_files = set()
        
        # If we have a file structure, use it to organize files
        if self.file_structure:
            # First, create all directories
            for path, file_info in self.file_structure.items():
                if isinstance(file_info, dict) and not file_info.get("type", "").lower() in ["file", "document"]:
                    # This is a directory entry
                    dir_path = os.path.join(self.output_dir, path)
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Created directory: {path}")
            
            # Then, create files based on the structure
            for path, file_info in self.file_structure.items():
                if isinstance(file_info, dict) and file_info.get("type", "").lower() in ["file", "document"]:
                    # This is a file entry in the structure
                    filename = os.path.basename(path)
                    
                    # Check if we have this file in our extracted code
                    if filename in self.extracted_code:
                        dir_path = os.path.join(self.output_dir, os.path.dirname(path))
                        os.makedirs(dir_path, exist_ok=True)
                        
                        with open(os.path.join(self.output_dir, path), 'w', encoding='utf-8') as f:
                            f.write(self.extracted_code[filename])
                        
                        processed_files.add(filename)
                        print(f"Created file: {path}")
        
        # Handle any extracted files that weren't in the structure
        for filename, content in self.extracted_code.items():
            if filename not in processed_files:
                # Infer the file type if not obvious from extension
                if '.' not in filename:
                    file_type = self.infer_file_type(content)
                    filename = f"{filename}.{file_type}"
                
                # Determine the appropriate directory based on extension
                ext = filename.split('.')[-1].lower()
                if ext in ['html', 'htm']:
                    dir_path = os.path.join(self.output_dir, 'html')
                elif ext == 'css':
                    dir_path = os.path.join(self.output_dir, 'css')
                elif ext in ['js', 'jsx', 'ts', 'tsx']:
                    dir_path = os.path.join(self.output_dir, 'js')
                elif ext in ['jpg', 'jpeg', 'png', 'gif', 'svg']:
                    dir_path = os.path.join(self.output_dir, 'assets', 'images')
                else:
                    dir_path = self.output_dir
                
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Created file: {file_path}")
    
    def initialize_git_repo(self) -> bool:
        """
        Initialize a Git repository in the output directory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            subprocess.run(['git', 'init'], cwd=self.output_dir, check=True)
            subprocess.run(['git', 'add', '.'], cwd=self.output_dir, check=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.output_dir, check=True)
            print(f"Initialized Git repository in {self.output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error initializing Git repository: {e}")
            return False
        except FileNotFoundError:
            print("Git command not found. Make sure Git is installed.")
            return False
    
    def create_zip_archive(self, zip_filename: str = None) -> str:
        """
        Create a ZIP archive of the output directory.
        
        Args:
            zip_filename: Name of the ZIP file to create (default: {output_dir}.zip)
            
        Returns:
            Path to the created ZIP file
        """
        if not zip_filename:
            zip_filename = f"{self.output_dir}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.output_dir)
                    zipf.write(file_path, arcname)
        
        print(f"Created ZIP archive: {zip_filename}")
        return zip_filename
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the organized code.
        
        Returns:
            Dictionary with summary information
        """
        file_count = 0
        dir_count = 0
        file_types = {}
        
        for root, dirs, files in os.walk(self.output_dir):
            dir_count += len(dirs)
            file_count += len(files)
            
            for file in files:
                ext = file.split('.')[-1].lower() if '.' in file else 'unknown'
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "output_directory": self.output_dir,
            "total_files": file_count,
            "total_directories": dir_count,
            "file_types": file_types
        }
    
    def parse_markdown_structure(self, markdown_content: str) -> Dict[str, Dict]:
        """
        Parse a markdown file that describes the file structure.
        
        The markdown format should use headings to denote directories and
        lists to denote files within those directories.
        
        Example format:
        # Project Structure
        
        ## html
        - index.html (Main HTML file)
        
        ## css
        - styles.css (Main CSS styles)
        
        ## assets/images
        - logo.png (Logo image file)
        
        Args:
            markdown_content: String containing the markdown content
            
        Returns:
            Dictionary representing the file structure
        """
        structure = {}
        current_dir = ""
        
        # Split by lines and process
        lines = markdown_content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for directory headers (## directory or ## directory/subdirectory)
            if line.startswith('##'):
                # Extract directory name
                current_dir = line.lstrip('#').strip()
                
                # If this is a nested path (like "assets/images"), create parent dirs too
                if '/' in current_dir:
                    parts = current_dir.split('/')
                    parent_path = ""
                    
                    # Add each parent directory
                    for i, part in enumerate(parts):
                        if i == 0:
                            parent_path = part
                        else:
                            parent_path = f"{parent_path}/{part}"
                            
                        # Add directory to structure if not already there
                        if parent_path not in structure:
                            structure[parent_path] = {
                                "description": f"Directory for {part} files"
                            }
                else:
                    # Add single directory to structure
                    structure[current_dir] = {
                        "description": f"Directory for {current_dir} files"
                    }
                
            # Check for files (- filename.ext (description))
            elif line.startswith('-') and current_dir:
                # Extract filename and description
                file_info = line.lstrip('- ').strip()
                
                # Check if there's a description in parentheses
                if '(' in file_info and file_info.endswith(')'):
                    filename, description = file_info.split('(', 1)
                    filename = filename.strip()
                    description = description.rstrip(')').strip()
                else:
                    filename = file_info
                    description = f"File: {filename}"
                    
                # Add file to structure
                file_path = f"{current_dir}/{filename}"
                structure[file_path] = {
                    "type": "file",
                    "description": description
                }
        
        return structure
        
        return structure
    
    def load_markdown_structure(self, markdown_content: str) -> None:
        """
        Load the file structure from a markdown string.
        
        Args:
            markdown_content: Markdown string containing the file structure
        """
        try:
            self.file_structure = self.parse_markdown_structure(markdown_content)
            print(f"Loaded markdown file structure with {len(self.file_structure)} entries")
        except Exception as e:
            print(f"Error parsing markdown file structure: {e}")
            raise


class GitHubAI:
    """
    Class for interacting with GitHub's AI model API.
    """
    
    def __init__(self):
        """
        Initialize the GitHub AI client.
        """
        load_dotenv()  # Load environment variables from .env file
        self.token = os.environ.get("GITHUB_TOKEN")
        self.endpoint = "https://models.github.ai/inference"
        self.model = "openai/gpt-4.1"
        
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable not found. Please set it in your .env file.")
        
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
    
    def generate_code(self, prompt: str, temperature: float = 1.0) -> str:
        """
        Generate code using GitHub's AI model.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Temperature parameter for the model (0.0 to 1.0)
            
        Returns:
            Generated code as a string
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates clean, maintainable code. Please provide code with clear comments and proper organization.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                top_p=1.0,
                model=self.model
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating code: {e}")
            raise


def main():
    """
    Main function to run the code organizer from command line.
    """
    parser = argparse.ArgumentParser(description="AI Code Organizer Agent")
    parser.add_argument("--llm-output", type=str, help="Path to LLM output file")
    parser.add_argument("--structure", type=str, help="Path to file structure (JSON or Markdown)")
    parser.add_argument("--output-dir", type=str, default="organized_code", help="Output directory")
    parser.add_argument("--git", action="store_true", help="Initialize Git repository")
    parser.add_argument("--zip", action="store_true", help="Create ZIP archive")
    parser.add_argument("--generate", action="store_true", help="Generate code using GitHub AI")
    parser.add_argument("--prompt", type=str, help="Prompt for code generation")
    
    args = parser.parse_args()
    
    # If generating code with GitHub AI
    if args.generate:
        if not args.prompt:
            print("No prompt provided for code generation. Please use --prompt.")
            return
        
        try:
            github_ai = GitHubAI()
            generated_code = github_ai.generate_code(args.prompt)
            
            # Save the generated code to a file
            with open("generated_code.txt", "w", encoding="utf-8") as f:
                f.write(generated_code)
            
            print("Code generated and saved to generated_code.txt")
            
            # Use the generated code as LLM output
            llm_output = generated_code
        except Exception as e:
            print(f"Error generating code: {e}")
            return
    else:
        # Use provided LLM output file
        if args.llm_output:
            with open(args.llm_output, 'r', encoding='utf-8') as f:
                llm_output = f.read()
        else:
            print("No LLM output provided. Use --llm-output or --generate with --prompt.")
            return
    
    # Initialize the code organizer
    organizer = CodeOrganizer(output_dir=args.output_dir)
    
    # Load file structure if provided
    if args.structure:
        organizer.load_structure_from_file(args.structure)
    
    # Parse LLM output and organize files
    organizer.parse_llm_output(llm_output)
    organizer.organize_files()
    
    # Initialize Git repository if requested
    if args.git:
        organizer.initialize_git_repo()
    
    # Create ZIP archive if requested
    if args.zip:
        organizer.create_zip_archive()
    
    # Print summary
    summary = organizer.get_summary()
    print("\nCode Organization Summary:")
    print(f"Output Directory: {summary['output_directory']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Total Directories: {summary['total_directories']}")
    print("File Types:")
    for ext, count in summary['file_types'].items():
        print(f"  .{ext}: {count} files")


if __name__ == "__main__":
    main()
