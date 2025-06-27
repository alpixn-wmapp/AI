import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HomepageSpecification:
    """Data class to hold homepage specification details"""
    prompt: str
    features: List[str]
    export_type: str
    ui_theme: Optional[str] = "Modern"
    color_palette: Optional[str] = "#ffffff"

class HomepagePromptSignature(dspy.Signature):
    """DSPy signature for homepage prompt enhancement"""
    
    basic_prompt = dspy.InputField(desc="The basic prompt describing the website purpose")
    features = dspy.InputField(desc="List of features to include on the homepage")
    export_type = dspy.InputField(desc="Technology stack for export (HTML/CSS/JS, React, Vue, etc.)")
    ui_theme = dspy.InputField(desc="Design theme (Modern, Minimal, Classic, etc.)")
    color_palette = dspy.InputField(desc="Color scheme for the website")
    
    enhanced_prompt = dspy.OutputField(desc="Comprehensive, detailed prompt for creating a single-page website with all homepage sections")

class HomepageStructureSignature(dspy.Signature):
    """DSPy signature for homepage structure planning"""
    
    basic_prompt = dspy.InputField(desc="The basic website description")
    features = dspy.InputField(desc="Required features")
    
    header_section = dspy.OutputField(desc="Header section requirements with navigation")
    hero_section = dspy.OutputField(desc="Hero section with compelling headline and CTA")
    features_section = dspy.OutputField(desc="Features/services section layout")
    about_section = dspy.OutputField(desc="About section content requirements")
    contact_section = dspy.OutputField(desc="Contact section specifications")
    footer_section = dspy.OutputField(desc="Footer section requirements")

class TechnicalSpecSignature(dspy.Signature):
    """DSPy signature for technical specifications"""
    
    export_type = dspy.InputField(desc="Target technology stack")
    ui_theme = dspy.InputField(desc="Design theme")
    color_palette = dspy.InputField(desc="Color scheme")
    features = dspy.InputField(desc="Required features")
    
    technical_requirements = dspy.OutputField(desc="Detailed technical implementation requirements")
    responsive_design = dspy.OutputField(desc="Mobile responsiveness specifications")
    performance_optimization = dspy.OutputField(desc="Performance and SEO requirements")

class HomepagePromptEnhancer(dspy.Module):
    """
    DSPy-powered Homepage Prompt Enhancer
    
    Uses DSPy modules to generate structured, optimized prompts for
    single-page website development with multiple reasoning steps.
    """
    
    def __init__(self):
        """Initialize the DSPy-powered Homepage Prompt Enhancer"""
        super().__init__()
        
        # Initialize DSPy language model
        self._setup_dspy_lm()
        
        # Initialize DSPy modules with Predict instead of ChainOfThought to avoid structured output issues
        self.structure_planner = dspy.Predict(HomepageStructureSignature)
        self.technical_spec = dspy.Predict(TechnicalSpecSignature)
        self.prompt_enhancer = dspy.Predict(HomepagePromptSignature)
        
        # Initialize optimizer
        self.optimizer = None
        self._setup_few_shot_examples()
        
    def _setup_dspy_lm(self):
        """Set up DSPy language model with GitHub Models API"""
        try:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise ValueError("GITHUB_TOKEN environment variable not found")
            
            # Configure DSPy to drop unsupported parameters
            import litellm
            litellm.drop_params = True
            
            # Use simple OpenAI configuration for GitHub Models
            lm = dspy.LM(
                model="gpt-4.1",  # Available model through GitHub Models
                api_base="https://models.inference.ai.azure.com",
                api_key=token,
                max_tokens=3000,
                temperature=0.7
            )
            
            dspy.configure(lm=lm)
            logger.info("DSPy language model configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure DSPy language model: {e}")
            # Fallback to basic configuration
            try:
                # Try alternative configuration without structured output
                from openai import OpenAI
                
                client = OpenAI(
                    api_key=os.environ.get("GITHUB_TOKEN"),
                    base_url="https://models.github.ai/inference"
                )
                
                # Simple wrapper that doesn't use response_format
                class SimpleOpenAIWrapper:
                    def __init__(self, client):
                        self.client = client
                        self.model = "gpt-4"
                    
                    def __call__(self, messages, **kwargs):
                        # Remove unsupported parameters
                        clean_kwargs = {k: v for k, v in kwargs.items() 
                                      if k not in ['response_format', 'tool_choice', 'tools']}
                        
                        response = self.client.chat.completions.create(
                            model="gpt-4",
                            messages=messages,
                            max_tokens=clean_kwargs.get('max_tokens', 3000),
                            temperature=clean_kwargs.get('temperature', 0.7)
                        )
                        return response.choices[0].message.content
                
                lm = SimpleOpenAIWrapper(client)
                dspy.configure(lm=lm)
                logger.info("DSPy configured with fallback wrapper")
                
            except Exception as fallback_error:
                logger.warning(f"Fallback configuration also failed: {fallback_error}")
                logger.info("Continuing without LM configuration...")
    
    def _setup_few_shot_examples(self):
        """Set up few-shot examples for DSPy optimization with proper input specification"""
        self.training_examples = [
            dspy.Example(
                basic_prompt="Create a portfolio website for a graphic designer",
                features="Portfolio Gallery, Contact Form, About Section, Services",
                export_type="React",
                ui_theme="Modern",
                color_palette="#2c3e50, #3498db, #ffffff"
            ).with_inputs("basic_prompt", "features", "export_type", "ui_theme", "color_palette"),
            
            dspy.Example(
                basic_prompt="Build a restaurant website",
                features="Menu Display, Online Ordering, Location Info, Reviews",
                export_type="HTML/CSS/JS",
                ui_theme="Elegant",
                color_palette="#8B4513, #F4A460, #FFFFFF"
            ).with_inputs("basic_prompt", "features", "export_type", "ui_theme", "color_palette")
        ]
    
    def forward(self, homepage_spec: HomepageSpecification):
        """Main forward pass through the DSPy modules"""
        try:
            # Step 1: Plan homepage structure
            structure_result = self.structure_planner(
                basic_prompt=homepage_spec.prompt,
                features=", ".join(homepage_spec.features)
            )
            
            # Step 2: Generate technical specifications
            tech_result = self.technical_spec(
                export_type=homepage_spec.export_type,
                ui_theme=homepage_spec.ui_theme,
                color_palette=homepage_spec.color_palette,
                features=", ".join(homepage_spec.features)
            )
            
            # Step 3: Create enhanced prompt combining all information
            enhanced_result = self.prompt_enhancer(
                basic_prompt=homepage_spec.prompt,
                features=", ".join(homepage_spec.features),
                export_type=homepage_spec.export_type,
                ui_theme=homepage_spec.ui_theme,
                color_palette=homepage_spec.color_palette
            )
            
            return dspy.Prediction(
                structure_plan={
                    "header": getattr(structure_result, 'header_section', 'Header with navigation'),
                    "hero": getattr(structure_result, 'hero_section', 'Hero section with CTA'),
                    "features": getattr(structure_result, 'features_section', 'Features showcase'),
                    "about": getattr(structure_result, 'about_section', 'About section'),
                    "contact": getattr(structure_result, 'contact_section', 'Contact information'),
                    "footer": getattr(structure_result, 'footer_section', 'Footer with links')
                },
                technical_specs={
                    "requirements": getattr(tech_result, 'technical_requirements', 'Standard web requirements'),
                    "responsive": getattr(tech_result, 'responsive_design', 'Mobile-first responsive design'),
                    "performance": getattr(tech_result, 'performance_optimization', 'SEO and performance optimized')
                },
                enhanced_prompt=getattr(enhanced_result, 'enhanced_prompt', self._generate_fallback_prompt(homepage_spec))
            )
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Return fallback result
            return dspy.Prediction(
                structure_plan={
                    "header": "Navigation header with logo and menu",
                    "hero": "Hero section with compelling headline and call-to-action",
                    "features": "Feature showcase section",
                    "about": "About section with company/personal information",
                    "contact": "Contact section with form and information",
                    "footer": "Footer with links and copyright"
                },
                technical_specs={
                    "requirements": f"Built with {homepage_spec.export_type}, {homepage_spec.ui_theme} theme",
                    "responsive": "Fully responsive design for all devices",
                    "performance": "Optimized for speed and SEO"
                },
                enhanced_prompt=self._generate_fallback_prompt(homepage_spec)
            )
    
    def _generate_fallback_prompt(self, homepage_spec: HomepageSpecification) -> str:
        """Generate a fallback enhanced prompt when DSPy fails"""
        features_str = ", ".join(homepage_spec.features)
        return f"""Create a {homepage_spec.ui_theme.lower()} {homepage_spec.export_type} website with the following specifications:

BASIC REQUIREMENTS:
- Purpose: {homepage_spec.prompt}
- Features: {features_str}
- Technology: {homepage_spec.export_type}
- Theme: {homepage_spec.ui_theme}
- Colors: {homepage_spec.color_palette}

STRUCTURE:
- HEADER: Navigation with logo and menu items
- HERO: Compelling headline with call-to-action button
- FEATURES: Showcase of key features/services
- ABOUT: Information about the company/person
- CONTACT: Contact form and information
- FOOTER: Links, social media, and copyright

TECHNICAL REQUIREMENTS:
- Fully responsive design
- Modern, clean aesthetic
- Fast loading and SEO optimized
- Cross-browser compatibility
- Accessibility compliant"""
    
    def optimize_with_bootstrap(self, max_bootstrapped_demos=4, max_labeled_demos=2):
        """Optimize the model using Bootstrap Few-Shot"""
        try:
            # Create a simple metric for evaluation
            def homepage_quality_metric(example, pred, trace=None):
                """Simple metric to evaluate prompt quality"""
                if not hasattr(pred, 'enhanced_prompt'):
                    return 0.0
                
                enhanced = pred.enhanced_prompt.lower()
                
                # Check for key homepage sections
                sections = ['header', 'hero', 'about', 'contact', 'footer']
                section_score = sum(1 for section in sections if section in enhanced) / len(sections)
                
                # Check for technical requirements
                tech_terms = ['responsive', 'mobile', 'seo', 'performance']
                tech_score = sum(1 for term in tech_terms if term in enhanced) / len(tech_terms)
                
                # Check for design elements
                design_terms = ['color', 'theme', 'layout', 'design']
                design_score = sum(1 for term in design_terms if term in enhanced) / len(design_terms)
                
                return (section_score + tech_score + design_score) / 3
            
            # Set up optimizer
            self.optimizer = BootstrapFewShot(
                metric=homepage_quality_metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos
            )
            
            # Compile with training examples
            optimized_enhancer = self.optimizer.compile(
                self, 
                trainset=self.training_examples
            )
            
            logger.info("DSPy model optimized successfully")
            return optimized_enhancer
            
        except Exception as e:
            logger.error(f"Failed to optimize DSPy model: {e}")
            return self
    
    def parse_input(self, input_data: Dict[str, Any]) -> HomepageSpecification:
        """Parse input data into HomepageSpecification object"""
        try:
            return HomepageSpecification(
                prompt=input_data.get("prompt", ""),
                features=input_data.get("features", []),
                export_type=input_data.get("export_type", "HTML/CSS/JS"),
                ui_theme=input_data.get("ui_theme", "Modern"),
                color_palette=input_data.get("color_palette", "#ffffff")
            )
        except Exception as e:
            logger.error(f"Failed to parse input data: {e}")
            raise
    
    def generate_enhanced_prompt(self, homepage_spec: HomepageSpecification) -> Dict[str, Any]:
        """Generate enhanced homepage prompt using DSPy"""
        try:
            logger.info("Generating enhanced prompt with DSPy...")
            
            # Run the DSPy forward pass
            result = self.forward(homepage_spec)
            
            logger.info("Enhanced prompt generated successfully")
            
            return {
                "original_prompt": homepage_spec.prompt,
                "features": homepage_spec.features,
                "export_type": homepage_spec.export_type,
                "ui_theme": homepage_spec.ui_theme,
                "color_palette": homepage_spec.color_palette,
                "structure_plan": result.structure_plan,
                "technical_specs": result.technical_specs,
                "enhanced_prompt": result.enhanced_prompt,
                "model_used": "DSPy with GPT-4"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced prompt: {e}")
            raise
    
    def process_from_json(self, json_file_path: str) -> Dict[str, Any]:
        """Process homepage specification from JSON file"""
        try:
            with open(json_file_path, 'r') as file:
                input_data = json.load(file)
            
            homepage_spec = self.parse_input(input_data)
            return self.generate_enhanced_prompt(homepage_spec)
            
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def process_from_dict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process homepage specification from dictionary"""
        homepage_spec = self.parse_input(input_data)
        return self.generate_enhanced_prompt(homepage_spec)
    
    def save_output(self, result: Dict[str, Any], output_file: str = "enhanced_homepage_prompt.json"):
        """Save the generated enhanced prompt to a file"""
        try:
            with open(output_file, 'w') as file:
                json.dump(result, file, indent=2)
            logger.info(f"Enhanced prompt saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            raise
    
    def save_prompt_only(self, result: Dict[str, Any], output_file: str = "enhanced_prompt.txt"):
        """Save only the enhanced prompt text to a file"""
        try:
            with open(output_file, 'w') as file:
                file.write(result["enhanced_prompt"])
            logger.info(f"Enhanced prompt text saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save prompt text: {e}")
            raise

def main():
    """Main function to demonstrate the DSPy-powered Homepage Prompt Enhancer"""
    
    # Initialize the enhancer
    enhancer = HomepagePromptEnhancer()
    
    # Example input
    example_input = {
        "prompt": "Create a modern portfolio website for a freelance graphic designer",
        "features": ["Portfolio Gallery", "Contact Form", "About Section", "Services List", "Testimonials"],
        "export_type": "React",
        "ui_theme": "Minimal",
        "color_palette": "#2c3e50, #3498db, #ffffff"
    }
    
    print("🚀 DSPy-Powered Homepage Prompt Enhancer")
    print("=" * 50)
    
    # Ask if user wants to optimize the model first
    optimize_choice = input("\n🎯 Optimize DSPy model with few-shot examples? (y/n): ").strip().lower()
    if optimize_choice == 'y':
        print("🔧 Optimizing DSPy model...")
        enhancer = enhancer.optimize_with_bootstrap(max_bootstrapped_demos=3, max_labeled_demos=2)
        print("✅ Model optimization completed")
    
    # Choose input method
    input_method = input("\nChoose input method:\n1. Manual input (dictionary)\n2. JSON file\n3. Interactive input\nEnter choice (1, 2, or 3): ").strip()
    
    try:
        if input_method == "1":
            print("\n📥 Processing example input...")
            result = enhancer.process_from_dict(example_input)
            
        elif input_method == "2":
            json_file = input("Enter JSON file path (default: sample_request.json): ").strip()
            if not json_file:
                json_file = "sample_request.json"
            
            print(f"\n📥 Processing JSON file: {json_file}")
            result = enhancer.process_from_json(json_file)
            
        elif input_method == "3":
            print("\n📝 Interactive Input:")
            prompt = input("Enter your basic prompt: ").strip()
            features_str = input("Enter features (comma-separated): ").strip()
            features = [f.strip() for f in features_str.split(",") if f.strip()]
            export_type = input("Enter export type (HTML/CSS/JS, React, Vue, etc.): ").strip()
            ui_theme = input("Enter UI theme (Modern, Minimal, Classic, etc.): ").strip()
            color_palette = input("Enter color palette: ").strip()
            
            interactive_input = {
                "prompt": prompt,
                "features": features,
                "export_type": export_type or "HTML/CSS/JS",
                "ui_theme": ui_theme or "Modern",
                "color_palette": color_palette or "#ffffff"
            }
            
            result = enhancer.process_from_dict(interactive_input)
            
        else:
            print("❌ Invalid choice. Using example input as default.")
            result = enhancer.process_from_dict(example_input)
        
        # Display results
        print("\n" + "=" * 80)
        print("📋 STRUCTURE PLAN")
        print("=" * 80)
        for section, content in result["structure_plan"].items():
            print(f"🔹 {section.upper()}: {content}")
        
        print("\n" + "=" * 80)
        print("⚙️ TECHNICAL SPECIFICATIONS")
        print("=" * 80)
        for spec_type, content in result["technical_specs"].items():
            print(f"🔹 {spec_type.upper()}: {content}")
        
        print("\n" + "=" * 80)
        print("🎯 ENHANCED HOMEPAGE PROMPT")
        print("=" * 80)
        print(result["enhanced_prompt"])
        
        # Save output options
        save_option = input("\n💾 Save output? (1. JSON file, 2. Text file only, 3. Both, 4. Skip): ").strip()
        
        if save_option in ["1", "3"]:
            json_output_file = input("Enter JSON output filename (default: enhanced_homepage_prompt.json): ").strip()
            if not json_output_file:
                json_output_file = "enhanced_homepage_prompt.json"
            enhancer.save_output(result, json_output_file)
            print(f"✅ JSON output saved to {json_output_file}")
        
        if save_option in ["2", "3"]:
            txt_output_file = input("Enter text output filename (default: enhanced_prompt.txt): ").strip()
            if not txt_output_file:
                txt_output_file = "enhanced_prompt.txt"
            enhancer.save_prompt_only(result, txt_output_file)
            print(f"✅ Text output saved to {txt_output_file}")
        
        print("\n🎉 Process completed successfully!")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()