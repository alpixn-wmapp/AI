import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppSpecification:
    """Data class to hold app specification details"""
    app_type: str
    features: List[str]
    ui_theme: str
    color_palette: str
    platforms: List[str]
    components: List[str]
    description: Optional[str] = None

class DescriptionAIAgent:
    """
    NeuraForge Description AI Agent
    
    This agent takes structured input about an app idea and generates
    a comprehensive development brief using GPT-4.1 via GitHub Marketplace
    """
    
    def __init__(self):
        """Initialize the Description AI Agent"""
        self.client = self._setup_openai_client()
        self.prompt_template = self._create_prompt_template()
        
    def _setup_openai_client(self) -> OpenAI:
        """Set up OpenAI client with GitHub Marketplace configuration"""
        try:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise ValueError("GITHUB_TOKEN environment variable not found")
                
            endpoint = "https://models.github.ai/inference"
            model = "openai/gpt-4.1"
            
            client = OpenAI(
                base_url=endpoint,
                api_key=token,
            )
            
            logger.info("OpenAI client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the enhanced prompt template for the Description AI Agent"""
        
        template = """
You are an expert AI architect and full-stack developer working on NeuraForge, an AI-powered app builder platform. 
Your task is to analyze the provided app specification and generate a comprehensive development brief.

## App Specification:
- App Type: {app_type}
- Features: {features}
- UI Theme: {ui_theme}
- Color Palette: {color_palette}
- Target Platforms: {platforms}
- Required Components: {components}
- Additional Description: {description}

## Your Task:
Generate a detailed development brief that includes:

### 1. 📝 App Summary
Provide a clear, concise summary (2-3 sentences) describing what this app does and its primary purpose.

### 2. 🧰 Recommended Tech Stack
Based on the app type, features, and platforms, recommend:
- Frontend framework/library
- Backend technology
- Database solution
- UI/CSS framework
- Additional tools and services needed

### 3. 🎨 UI Structure & Screens
List the main screens/pages needed and their purpose:
- Core navigation structure
- Key user interface components
- User flow between screens

### 4. 🔧 Backend Logic & Features
Detail the backend requirements:
- Authentication system
- Core business logic
- API endpoints needed
- Third-party integrations
- Data models and relationships

### 5. 📁 Suggested Project Structure
Provide a complete folder/file structure for the project, showing:
- Main directories
- Key files in each directory
- Configuration files
- Asset organization

### 6. 🚀 Development Priorities
Suggest the order of implementation:
- Phase 1: Core features
- Phase 2: Secondary features
- Phase 3: Enhancements

### 7. 📋 Technical Considerations
Include important technical notes:
- Scalability considerations
- Security requirements
- Performance optimizations
- Mobile responsiveness (if applicable)

Please format your response clearly with the sections above. Be specific and actionable in your recommendations.
"""
        
        return PromptTemplate(
            input_variables=[
                "app_type", "features", "ui_theme", "color_palette", 
                "platforms", "components", "description"
            ],
            template=template
        )
    
    def parse_input(self, input_data: Dict[str, Any]) -> AppSpecification:
        """Parse input data into AppSpecification object"""
        try:
            return AppSpecification(
                app_type=input_data.get("app_type", "Web App"),
                features=input_data.get("features", []),
                ui_theme=input_data.get("ui_theme", "Modern"),
                color_palette=input_data.get("color_palette", "#ffffff"),
                platforms=input_data.get("platforms", ["Web"]),
                components=input_data.get("components", []),
                description=input_data.get("description", "")
            )
        except Exception as e:
            logger.error(f"Failed to parse input data: {e}")
            raise
    
    def generate_development_brief(self, app_spec: AppSpecification) -> Dict[str, str]:
        """Generate comprehensive development brief using GPT-4.1"""
        try:
            # Format the prompt with app specification details
            formatted_prompt = self.prompt_template.format(
                app_type=app_spec.app_type,
                features=", ".join(app_spec.features),
                ui_theme=app_spec.ui_theme,
                color_palette=app_spec.color_palette,
                platforms=", ".join(app_spec.platforms),
                components=", ".join(app_spec.components),
                description=app_spec.description or "No additional description provided"
            )
            
            logger.info("Sending request to GPT-4.1...")
            
            # Make API call to GPT-4.1
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI architect and full-stack developer. Provide comprehensive, actionable development briefs for app building projects."
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.7,
                top_p=0.9,
                model="openai/gpt-4.1",
                max_tokens=4000
            )
            
            development_brief = response.choices[0].message.content
            logger.info("Development brief generated successfully")
            
            return {
                "input_specification": app_spec.__dict__,
                "development_brief": development_brief,
                "model_used": "openai/gpt-4.1",
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Failed to generate development brief: {e}")
            raise
    
    def process_from_json(self, json_file_path: str) -> Dict[str, str]:
        """Process app specification from JSON file"""
        try:
            with open(json_file_path, 'r') as file:
                input_data = json.load(file)
            
            app_spec = self.parse_input(input_data)
            return self.generate_development_brief(app_spec)
            
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def process_from_dict(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Process app specification from dictionary"""
        app_spec = self.parse_input(input_data)
        return self.generate_development_brief(app_spec)
    
    def save_output(self, result: Dict[str, str], output_file: str = "development_brief.json"):
        """Save the generated development brief to a file"""
        try:
            with open(output_file, 'w') as file:
                json.dump(result, file, indent=2)
            logger.info(f"Development brief saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            raise

def main():
    """Main function to demonstrate the Description AI Agent"""
    
    # Initialize the agent
    agent = DescriptionAIAgent()
    
    # Example 1: Using manual input (dictionary)
    manual_input = {
        "app_type": "Web App",
        "features": ["Chat", "Payments", "File Upload", "User Authentication"],
        "ui_theme": "Minimal",
        "color_palette": "#ffffff",
        "platforms": ["Web", "Mobile"],
        "components": ["Login Page", "Dashboard", "Profile Page", "Chat Interface"],
        "description": "A minimal web app for mobile and desktop that allows users to chat, upload files, and make payments"
    }
    
    print("🚀 NeuraForge Description AI Agent")
    print("=" * 50)
    
    # Choose input method
    input_method = input("\nChoose input method:\n1. Manual input (dictionary)\n2. JSON file\nEnter choice (1 or 2): ").strip()
    
    try:
        if input_method == "1":
            print("\n📥 Processing manual input...")
            result = agent.process_from_dict(manual_input)
            
        elif input_method == "2":
            json_file = input("Enter JSON file path (default: app_spec.json): ").strip()
            if not json_file:
                json_file = "app_spec.json"
            
            print(f"\n📥 Processing JSON file: {json_file}")
            result = agent.process_from_json(json_file)
            
        else:
            print("❌ Invalid choice. Using manual input as default.")
            result = agent.process_from_dict(manual_input)
        
        # Display results
        print("\n" + "=" * 80)
        print("📋 GENERATED DEVELOPMENT BRIEF")
        print("=" * 80)
        print(result["development_brief"])
        
        # Save output
        save_option = input("\n💾 Save output to file? (y/n): ").strip().lower()
        if save_option == 'y':
            output_file = input("Enter output filename (default: development_brief.json): ").strip()
            if not output_file:
                output_file = "development_brief.json"
            agent.save_output(result, output_file)
            print(f"✅ Output saved to {output_file}")
        
        print("\n🎉 Process completed successfully!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()