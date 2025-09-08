import os
import io
import json
import logging
from PIL import Image
import google.generativeai as genai


class GeminiEnhancer:
    """
    Gemini AI Photo Enhancement System
    Provides AI-powered photo enhancement using Google's Gemini models
    """

    def __init__(self):
        print("Initializing Gemini Enhancer...")
        # Configure the API key
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

        console.log("Gemini API configured with key: " +
                    os.environ.get("GEMINI_API_KEY"))
        # The model name for image editing
        self.model_name = "gemini-2.5-flash-image-preview"
        # Instantiate the model
        self.model = genai.GenerativeModel(self.model_name)
        print(f"Gemini Enhancer ready with model: {self.model_name}")

    def enhance_image(self,
                      original_path,
                      enhanced_path,
                      enhancements,
                      enhancement_prompt="",
                      source_image_path=None):
        """
        Apply AI-powered enhancements to an image using Gemini
        
        Args:
            original_path: Path to original image
            enhanced_path: Path to save enhanced image
            enhancements: Dictionary of enhancement parameters
            enhancement_prompt: User's text prompt for AI enhancement
            source_image_path: Path to use as source (for conversational editing)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use source_image_path if provided (for conversational editing), otherwise use original
            input_path = source_image_path if source_image_path else original_path
            print(f"Enhancing image with Gemini: {input_path}")

            # Load the base image
            base_image = Image.open(input_path)
            # Convert image to RGB format if needed to avoid transparency issues
            if base_image.mode == 'RGBA':
                base_image = base_image.convert('RGB')

            # Create enhancement prompt for conversational editing
            prompt = self._create_conversational_prompt(
                enhancement_prompt, bool(source_image_path))
            print(f"Enhancement prompt: {prompt}")

            # Use the generate_content method with both prompt and image
            response = self.model.generate_content([prompt, base_image])

            # Check if the response contains parts with image data
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    # Check for image data in the part
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Extract image data from inline_data
                        image_data = part.inline_data.data
                        # Convert bytes to PIL Image
                        edited_image = Image.open(io.BytesIO(image_data))
                        # Save the edited image
                        edited_image.save(enhanced_path)
                        print(f"Enhanced image saved to: {enhanced_path}")
                        return True
                    elif hasattr(part, 'text') and part.text:
                        print("Gemini response:", part.text)

                print("Model response did not contain image data.")
                return False
            else:
                print("Model response did not contain any parts.")
                # Log any text response
                if hasattr(response, 'text') and response.text:
                    print("Gemini response:", response.text)
                return False

        except Exception as e:
            print(f"Gemini enhancement error: {e}")
            logging.error(f"Gemini enhancement error: {str(e)}")
            return False

    def _create_enhancement_prompt(self, enhancements, user_prompt=""):
        """
        Create a detailed prompt for Gemini based on enhancement parameters and user input
        """
        # Start with base prompt for photo enhancement
        base_prompt = "Create a high-quality, professional enhanced version of this photo with natural-looking improvements. "

        # Add specific enhancements based on parameters
        enhancement_descriptions = []

        if enhancements.get('eye_color'):
            enhancement_descriptions.append(
                f"change eye color to {enhancements['eye_color']}")

        if enhancements.get('eye_shape'):
            enhancement_descriptions.append(
                f"enhance eye shape to be more {enhancements['eye_shape']}")

        if enhancements.get('face_shape'):
            enhancement_descriptions.append(
                f"adjust face shape to be more {enhancements['face_shape']}")

        if enhancements.get('hair_color'):
            enhancement_descriptions.append(
                f"change hair color to {enhancements['hair_color']}")

        if enhancements.get('hair_style'):
            enhancement_descriptions.append(
                f"style hair to be {enhancements['hair_style']}")

        if enhancements.get('lip_color'):
            enhancement_descriptions.append(
                f"enhance lip color to {enhancements['lip_color']}")

        if enhancements.get('skin_tone'):
            enhancement_descriptions.append("improve skin tone and texture")

        if enhancements.get('blemish_removal'):
            enhancement_descriptions.append(
                "remove blemishes and imperfections")

        if enhancements.get('makeup_application'):
            enhancement_descriptions.append("apply natural-looking makeup")

        if enhancements.get('lighting_enhancement'):
            enhancement_descriptions.append(
                "enhance lighting and overall brightness")

        if enhancements.get('background_change'):
            enhancement_descriptions.append(
                f"change background to {enhancements.get('background_type', 'a professional setting')}"
            )

        # Combine all enhancements
        if enhancement_descriptions:
            enhancement_text = "Apply these specific enhancements: " + ", ".join(
                enhancement_descriptions) + ". "
        else:
            enhancement_text = "Apply general photo enhancement including skin smoothing, color correction, and lighting improvements. "

        # Add user's custom prompt if provided
        if user_prompt and user_prompt.strip():
            custom_prompt = f"Additionally, {user_prompt.strip()}. "
        else:
            custom_prompt = ""

        # Quality and style instructions
        quality_prompt = (
            "Maintain high image quality, natural skin tones, and realistic proportions. "
            "Ensure the enhanced photo looks professional and suitable for social media. "
            "Keep facial features recognizable and avoid over-processing.")

        # Combine all parts
        full_prompt = base_prompt + enhancement_text + custom_prompt + quality_prompt

        return full_prompt

    def _create_conversational_prompt(self, user_prompt, is_follow_up=False):
        """
        Create a conversational prompt for multi-turn editing
        """
        if is_follow_up:
            # This is a follow-up edit - focus on the specific change requested
            base_prompt = (
                "Based on this current image, make the following specific change: "
            )
            instruction = f"{user_prompt.strip()}. "
            quality_prompt = (
                "Keep all other aspects of the image exactly the same. "
                "Only modify what was specifically requested. "
                "Maintain the same quality, lighting, and composition.")
        else:
            # This is the initial edit
            base_prompt = (
                "Create a high-quality, professionally enhanced version of this photo. "
            )
            if user_prompt and user_prompt.strip():
                instruction = f"Apply this specific enhancement: {user_prompt.strip()}. "
            else:
                instruction = (
                    "Apply general photo enhancement including skin smoothing, "
                    "color correction, and lighting improvements. ")
            quality_prompt = (
                "Ensure the enhanced photo looks professional and suitable for social media. "
                "Keep facial features recognizable and avoid over-processing. "
                "Maintain natural skin tones and realistic proportions.")

        return base_prompt + instruction + quality_prompt

    def analyze_image(self, image_path):
        """
        Analyze an image to understand its content and suggest enhancements
        """
        try:
            # Load the image
            base_image = Image.open(image_path)
            if base_image.mode == 'RGBA':
                base_image = base_image.convert('RGB')

            # Create analysis prompt
            analysis_prompt = (
                "Analyze this portrait photo and suggest specific enhancements that would make it more suitable for social media. "
                "Focus on facial features, lighting, skin quality, and overall composition. "
                "Provide specific suggestions for improvements.")

            # Use the same model for analysis
            response = self.model.generate_content(
                [analysis_prompt, base_image])

            return response.text if hasattr(
                response,
                'text') and response.text else "No analysis available"

        except Exception as e:
            print(f"Image analysis error: {e}")
            return f"Analysis failed: {str(e)}"
