"""
Demo script to show how to use the AI Code Organizer.
"""

import os
from code_organizer_enhanced import CodeOrganizer, GitHubAI

def demo_with_example_files():
    """
    Demonstrate the Code Organizer using example files.
    """
    print("=" * 50)
    print("AI Code Organizer - Demo with Example Files")
    print("=" * 50)
    
    # Create a Code Organizer instance
    organizer = CodeOrganizer(output_dir="example_output")
    
    # Load the file structure
    organizer.load_file_structure_from_file("example_structure.json")
    
    # Read the LLM output
    with open("example_llm_output.txt", "r", encoding="utf-8") as f:
        llm_output = f.read()
    
    # Parse the LLM output
    extracted_files = organizer.parse_llm_output(llm_output)
    
    # Print extracted files
    print("\nExtracted Files:")
    for filename, content in extracted_files.items():
        print(f"- {filename} ({len(content)} characters)")
    
    # Organize the files
    organizer.organize_files()
    
    # Initialize a Git repository (optional)
    # organizer.initialize_git_repo()
    
    # Create a ZIP archive (optional)
    organizer.create_zip_archive()
    
    # Print a summary
    summary = organizer.get_summary()
    print("\nCode Organization Summary:")
    print(f"Output Directory: {summary['output_directory']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Total Directories: {summary['total_directories']}")
    print("File Types:")
    for ext, count in summary['file_types'].items():
        print(f"  .{ext}: {count} files")
    
    print("\nOutput files are in the 'example_output' directory.")
    print("A ZIP archive has been created at 'example_output.zip'.")


def demo_with_github_ai():
    """
    Demonstrate the Code Organizer with GitHub AI integration.
    Note: This requires a valid GITHUB_TOKEN environment variable.
    """
    print("=" * 50)
    print("AI Code Organizer - Demo with GitHub AI")
    print("=" * 50)
    
    # Check if the GITHUB_TOKEN environment variable is set
    if not os.environ.get("GITHUB_TOKEN"):
        print("GITHUB_TOKEN environment variable not found.")
        print("Please set it in your .env file or environment variables.")
        return
    
    try:
        # Create a GitHub AI instance
        github_ai = GitHubAI()
        
        # Generate code with a prompt
        prompt = """
        Create a simple personal portfolio website with:
        1. A header with navigation
        2. A hero section with your name and title
        3. An about section
        4. A skills section
        5. A projects section
        6. A contact form
        7. A footer
        
        Use modern HTML, CSS, and a bit of JavaScript for interactivity.
        Make it responsive and visually appealing.
        """
        
        print("Generating code with GitHub AI...")
        generated_code = github_ai.generate_code(prompt, temperature=0.7)
        
        # Save the generated code to a file
        with open("generated_code.txt", "w", encoding="utf-8") as f:
            f.write(generated_code)
        
        print("Code generated and saved to generated_code.txt")
        
        # Create a Code Organizer instance
        organizer = CodeOrganizer(output_dir="ai_generated_output")
        
        # Parse the generated code
        extracted_files = organizer.parse_llm_output(generated_code)
        
        # Print extracted files
        print("\nExtracted Files:")
        for filename, content in extracted_files.items():
            print(f"- {filename} ({len(content)} characters)")
        
        # Organize the files
        organizer.organize_files()
        
        # Print a summary
        summary = organizer.get_summary()
        print("\nCode Organization Summary:")
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Total Files: {summary['total_files']}")
        print(f"Total Directories: {summary['total_directories']}")
        print("File Types:")
        for ext, count in summary['file_types'].items():
            print(f"  .{ext}: {count} files")
        
        print("\nOutput files are in the 'ai_generated_output' directory.")
    
    except Exception as e:
        print(f"Error in GitHub AI demo: {e}")


def demo_with_markdown_structure():
    """
    Demonstrate the Code Organizer using a Markdown file structure.
    """
    print("=" * 50)
    print("AI Code Organizer - Demo with Markdown Structure")
    print("=" * 50)
    
    # Create a Code Organizer instance
    organizer = CodeOrganizer(output_dir="markdown_output")
    
    # Load the Markdown file structure
    organizer.load_file_structure_from_file("example_structure.md")
    
    # Print the loaded structure
    print("\nFile Structure from Markdown:")
    for path, info in organizer.file_structure.items():
        description = info.get("description", "")
        file_type = info.get("type", "directory")
        if file_type == "file":
            print(f"  - 📄 {path}: {description}")
        else:
            print(f"  - 📁 {path}: {description}")
    
    # Read the LLM output
    with open("example_llm_output.txt", "r", encoding="utf-8") as f:
        llm_output = f.read()
    
    # Parse the LLM output
    extracted_files = organizer.parse_llm_output(llm_output)
    
    # Print extracted files
    print("\nExtracted Files:")
    for filename, content in extracted_files.items():
        print(f"  - {filename} ({len(content)} characters)")
    
    # Organize the files
    organizer.organize_files()
    
    # Create a ZIP archive
    organizer.create_zip_archive("markdown_output.zip")
    
    # Print a summary
    summary = organizer.get_summary()
    print("\nCode Organization Summary:")
    print(f"Output Directory: {summary['output_directory']}")
    print(f"Total Files: {summary['total_files']}")
    print(f"Total Directories: {summary['total_directories']}")
    print("File Types:")
    for ext, count in summary['file_types'].items():
        print(f"  .{ext}: {count} files")
    
    print("\nOutput files are in the 'markdown_output' directory.")
    print("A ZIP archive has been created at 'markdown_output.zip'.")


if __name__ == "__main__":
    # Ensure .env file is loaded for GitHub token
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the example files demo with JSON structure
    demo_with_example_files()
    
    print("\n")
    
    # Run the demo with Markdown structure
    demo_with_markdown_structure()
    
    print("\n")
    
    # Uncomment to run the GitHub AI demo
    # Note: This requires a valid GITHUB_TOKEN environment variable
    # demo_with_github_ai()
