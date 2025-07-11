"""Analytics service using Llava-Phi3 for tea leaf analysis and waste prevention recommendations."""

import os
import base64
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analyzing detection results and providing waste prevention recommendations using Llava-Phi3."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        """
        Initialize analytics service.
        
        Args:
            ollama_host: Ollama server host URL
        """
        self.ollama_host = ollama_host
        self.model_name = "llava-phi3:3.8b"
        self.analytics_dir = "analytics"
        
        # Create analytics directory
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Check if Ollama server is available
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama server is available and model is ready."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if not any(self.model_name in name for name in model_names):
                    logger.warning(f"Llava-Phi3 model ({self.model_name}) not found. Please install it with: ollama pull {self.model_name}")
                else:
                    logger.info("Ollama server and Llava-Phi3 model are available")
            else:
                logger.warning(f"Ollama server not responding correctly: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server at {self.ollama_host}: {e}")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string for Ollama API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def _create_analysis_prompt(self, detection_data: Dict[str, Any]) -> str:
        """
        Create analysis prompt for Llava-Phi3.
        
        Args:
            detection_data: Detection results data
            
        Returns:
            Formatted prompt string
        """
        healthy_count = detection_data.get("healthy_count", 0)
        unhealthy_count = detection_data.get("unhealthy_count", 0)
        total_count = detection_data.get("total_count", 0)
        health_percentage = detection_data.get("health_percentage", 0.0)
        
        prompt = f"""
You are an expert tea leaf quality analyst and agricultural consultant. I'm showing you an image of tea leaves with detection results overlaid. If the image show no tea leaf, just reply "No tea leaf detected".

DETECTION RESULTS:
- Total leaves detected: {total_count}
- Healthy leaves: {healthy_count}
- Unhealthy leaves: {unhealthy_count}
- Health percentage: {health_percentage:.1f}%

Please analyze this image and provide detailed recommendations in the following JSON format:

{{
    "analysis": {{
        "overall_assessment": "Brief overall assessment of the tea leaf quality based on the image provided",
        "severity_level": "Low/Medium/High",
        "quality_grade": "Premium/Standard/Below standard/Reject"
    }},
    "processing_recommendations": {{
        "immediate_actions": ["List of immediate actions to take"],
        "sorting_strategy": "How to sort and separate leaves",
        "processing_method": "Recommended processing approach",
        "quality_preservation": ["Steps to preserve remaining quality"]
    }},
    "waste_prevention": {{
        "salvageable_portions": "What parts can still be used",
        "alternative_uses": ["Alternative uses for defective leaves"],
        "composting_guidelines": "How to compost unusable leaves",
        "prevention_measures": ["How to prevent similar issues in future"]
    }},
    "economic_impact": {{
        "estimated_loss_percentage": "Percentage of economic loss",
        "cost_saving_opportunities": ["Ways to minimize financial impact"],
        "value_recovery_methods": ["Methods to recover some value"]
    }},
    "recommendations": {{
        "priority_actions": ["Top 3 priority actions"],
        "timeline": "Suggested timeline for implementation",
        "monitoring_points": ["What to monitor going forward"]
    }}
}}

Focus on practical, actionable advice that can help minimize waste and maximize value recovery from the detected tea leaves. Consider both the healthy and unhealthy leaves in your analysis.
"""
        return prompt
    
    def analyze_detection_result(self, detection_data: Dict[str, Any], 
                               annotated_image_path: str, 
                               session_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze detection results using Llava-Phi3 and provide recommendations.

        Args:
            detection_data: Detection results from the detection service
            annotated_image_path: Path to the annotated image
            session_id: The ID of the session this analysis belongs to
            
        Returns:
            Analysis results with recommendations
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not annotated_image_path or not os.path.exists(annotated_image_path):
                raise FileNotFoundError(f"Annotated image not found: {annotated_image_path}")
            
            # Encode image to base64
            logger.info(f"Encoding image for analysis: {annotated_image_path}")
            image_b64 = self._encode_image_to_base64(annotated_image_path)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(detection_data)
            
            # Prepare request for Ollama API
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent analysis
                    "num_predict": 2048   # Allow longer responses
                }
            }
            
            # Send request to Ollama
            logger.info("Sending analysis request to Llava-Phi3...")
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout for vision model
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            # Parse response
            response_data = response.json()
            llama_response = response_data.get("response", "")
            
            if not llama_response:
                raise Exception("Empty response from Llava-Phi3")
            
            logger.info(f"Received response from Llava-Phi3, length: {len(llama_response)} characters")
            
            # Extract JSON from response
            analysis_result = self._parse_llama_response(llama_response)
            logger.info(f"Parsed analysis result from response for image: {annotated_image_path}")
            
            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "analysis_id": self._generate_analysis_id(),
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "model_used": self.model_name,
                "image_path": annotated_image_path,
                "detection_summary": {
                    "healthy_count": detection_data.get("healthy_count", 0),
                    "unhealthy_count": detection_data.get("unhealthy_count", 0),
                    "total_count": detection_data.get("total_count", 0),
                    "health_percentage": detection_data.get("health_percentage", 0.0)
                },
                "ai_analysis": analysis_result,
                "status": "completed"
            }
            
            # Save analysis result
            self._save_analysis_result(result)
            logger.info(f"Analysis result saved for image: {annotated_image_path}")
            
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            
            # Return fallback analysis
            return self._create_fallback_analysis(detection_data, annotated_image_path, str(e), session_id)
    
    def _parse_llama_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from Llava-Phi3.
        
        Args:
            response: Raw response string from Llava-Phi3
            
        Returns:
            Parsed analysis data
        """
        try:
            # Try to extract JSON from response
            # Look for JSON block in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, create structured response from text
                return self._parse_text_response(response)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """
        Parse text response when JSON parsing fails.
        
        Args:
            response: Raw text response
            
        Returns:
            Structured analysis data
        """
        # Create basic structure from text response
        return {
            "analysis": {
                "overall_assessment": response[:200] + "..." if len(response) > 200 else response,
                "defect_types": ["Various defects detected"],
                "severity_level": "medium",
                "quality_grade": "standard"
            },
            "processing_recommendations": {
                "immediate_actions": ["Sort leaves by quality", "Separate healthy from unhealthy"],
                "sorting_strategy": "Manual sorting recommended",
                "processing_method": "Standard processing with quality controls",
                "quality_preservation": ["Store in dry conditions", "Process quickly"]
            },
            "waste_prevention": {
                "salvageable_portions": "Healthy portions can be fully utilized",
                "alternative_uses": ["Compost unhealthy leaves", "Use for fertilizer"],
                "composting_guidelines": "Standard composting procedures",
                "prevention_measures": ["Regular quality monitoring", "Improved harvesting practices"]
            },
            "economic_impact": {
                "estimated_loss_percentage": "15-25%",
                "cost_saving_opportunities": ["Improved sorting", "Better processing"],
                "value_recovery_methods": ["Alternative products", "Composting"]
            },
            "recommendations": {
                "priority_actions": ["Sort immediately", "Process healthy leaves first", "Plan waste utilization"],
                "timeline": "Immediate action recommended",
                "monitoring_points": ["Quality trends", "Processing efficiency"]
            },
            "raw_response": response
        }
    
    def _create_fallback_analysis(self, detection_data: Dict[str, Any], 
                                image_path: str, error: str, session_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create fallback analysis when AI analysis fails.
        
        Args:
            detection_data: Detection results
            image_path: Path to image
            error: Error message
            session_id: The ID of the session this analysis belongs to
            
        Returns:
            Fallback analysis result
        """
        healthy_count = detection_data.get("healthy_count", 0)
        unhealthy_count = detection_data.get("unhealthy_count", 0)
        total_count = detection_data.get("total_count", 0)
        health_percentage = detection_data.get("health_percentage", 0.0)
        
        # Determine severity based on health percentage
        if health_percentage >= 80:
            severity = "low"
            grade = "premium"
        elif health_percentage >= 60:
            severity = "medium"
            grade = "standard"
        else:
            severity = "high"
            grade = "below_standard"
        
        return {
            "analysis_id": self._generate_analysis_id(),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0.0,
            "model_used": "fallback_analysis",
            "image_path": image_path,
            "detection_summary": {
                "healthy_count": healthy_count,
                "unhealthy_count": unhealthy_count,
                "total_count": total_count,
                "health_percentage": health_percentage
            },
            "ai_analysis": {
                "analysis": {
                    "overall_assessment": f"Automated analysis based on detection results. {health_percentage:.1f}% healthy leaves detected.",
                    "defect_types": ["Quality assessment based on detection"],
                    "severity_level": severity,
                    "quality_grade": grade
                },
                "processing_recommendations": {
                    "immediate_actions": [
                        "Sort leaves by quality immediately",
                        "Separate healthy from unhealthy leaves",
                        "Process healthy leaves first"
                    ],
                    "sorting_strategy": "Automated sorting recommended based on detection results",
                    "processing_method": "Standard processing with quality controls",
                    "quality_preservation": [
                        "Maintain proper storage conditions",
                        "Process within optimal timeframe"
                    ]
                },
                "waste_prevention": {
                    "salvageable_portions": f"{healthy_count} healthy leaves can be fully utilized",
                    "alternative_uses": [
                        "Compost unhealthy leaves for organic fertilizer",
                        "Use lower grade leaves for secondary products"
                    ],
                    "composting_guidelines": "Standard composting procedures for organic waste",
                    "prevention_measures": [
                        "Regular quality monitoring",
                        "Improved harvesting practices",
                        "Better storage conditions"
                    ]
                },
                "economic_impact": {
                    "estimated_loss_percentage": f"{100 - health_percentage:.1f}%",
                    "cost_saving_opportunities": [
                        "Improved sorting efficiency",
                        "Better processing methods"
                    ],
                    "value_recovery_methods": [
                        "Alternative product development",
                        "Organic fertilizer production"
                    ]
                },
                "recommendations": {
                    "priority_actions": [
                        "Sort leaves immediately",
                        "Process healthy leaves first",
                        "Plan waste utilization strategy"
                    ],
                    "timeline": "Immediate action recommended for fresh leaves",
                    "monitoring_points": [
                        "Quality trends over time",
                        "Processing efficiency metrics"
                    ]
                }
            },
            "status": "completed_fallback",
            "error": error
        }
    
    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID."""
        import uuid
        timestamp = int(datetime.now().timestamp())
        unique_id = uuid.uuid4().hex[:8]
        return f"analysis_{unique_id}_{timestamp}"
    
    def _save_analysis_result(self, analysis_result: Dict[str, Any]):
        """
        Save analysis result to file.
        
        Args:
            analysis_result: Analysis result to save
        """
        try:
            analysis_id = analysis_result.get("analysis_id")
            
            # Don't save if analysis_id is missing or contains "unknown"
            if not analysis_id or "unknown" in analysis_id:
                logger.warning(f"Skipping save for invalid analysis_id: {analysis_id}")
                return
            
            filename = f"{analysis_id}.json"
            filepath = os.path.join(self.analytics_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis result saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
    
    def analyze_batch_results(self, batch_results: List[Dict[str, Any]], session_id: int) -> Dict[str, Any]:
        """
        Analyze a batch of detection results and provide aggregate recommendations.
        
        Args:
            batch_results: List of individual detection results
            session_id: The ID of the session this batch belongs to
            
        Returns:
            Aggregate analysis with batch-level recommendations
        """
        start_time = datetime.now()
        
        try:
            # Calculate aggregate statistics
            total_images = len(batch_results)
            total_healthy = sum(r.get("healthy_count", 0) for r in batch_results)
            total_unhealthy = sum(r.get("unhealthy_count", 0) for r in batch_results)
            total_leaves = total_healthy + total_unhealthy
            
            overall_health_percentage = (total_healthy / total_leaves * 100) if total_leaves > 0 else 0
            
            # If only one result, do NOT run batch analysis, just run individual analysis and return it
            if total_images == 1:
                single_result = batch_results[0]
                analysis = self.analyze_detection_result(
                    single_result,
                    single_result["annotated_image_path"],
                    session_id=session_id
                )
                # Only return the individual analysis, no batch_analysis_id
                return {
                    "individual_analyses": [analysis],
                    "status": "single_analysis_only"
                }

            # Analyze individual results that have AI analysis
            analyzed_results = []
            for result in batch_results:
                if result.get("annotated_image_path") and os.path.exists(result["annotated_image_path"]):
                    try:
                        analysis = self.analyze_detection_result(
                            result, 
                            result["annotated_image_path"], 
                            session_id=session_id
                        )
                        analyzed_results.append(analysis)
                    except Exception as e:
                        logger.warning(f"Failed to analyze image {result.get('image_name', 'unknown')}: {e}")

            # Create batch summary
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate proper batch analysis ID
            batch_analysis_id = self._generate_analysis_id()
            
            batch_analysis = {
                "batch_analysis_id": batch_analysis_id,
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "processing_time": processing_time,
                "batch_summary": {
                    "total_images": total_images,
                    "analyzed_images": len(analyzed_results),
                    "total_healthy_leaves": total_healthy,
                    "total_unhealthy_leaves": total_unhealthy,
                    "total_leaves": total_leaves,
                    "overall_health_percentage": overall_health_percentage
                },
                "aggregate_recommendations": self._generate_batch_recommendations(
                    overall_health_percentage, total_leaves, analyzed_results
                ),
                "individual_analyses": analyzed_results,
                "status": "completed"
            }
            
            # Save batch analysis with proper ID
            self._save_analysis_result(batch_analysis)
            
            return batch_analysis
            
        except Exception as e:
            logger.error(f"Error during batch analysis: {e}")
            # Generate proper ID even for failed analysis
            failed_analysis_id = self._generate_analysis_id()
            return {
                "batch_analysis_id": failed_analysis_id,
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_batch_recommendations(self, health_percentage: float, 
                                      total_leaves: int, 
                                      individual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate batch-level recommendations based on aggregate data.
        
        Args:
            health_percentage: Overall health percentage
            total_leaves: Total number of leaves
            individual_analyses: List of individual analysis results
            
        Returns:
            Batch-level recommendations
        """
        # Determine overall quality level
        if health_percentage >= 85:
            quality_level = "excellent"
            priority = "standard_processing"
        elif health_percentage >= 70:
            quality_level = "good"
            priority = "quality_monitoring"
        elif health_percentage >= 50:
            quality_level = "moderate"
            priority = "enhanced_sorting"
        else:
            quality_level = "poor"
            priority = "damage_control"
        
        return {
            "overall_assessment": {
                "quality_level": quality_level,
                "health_percentage": health_percentage,
                "priority_level": priority,
                "recommended_action": self._get_priority_action(priority)
            },
            "batch_processing_strategy": {
                "sorting_approach": "Automated pre-sorting followed by manual quality control",
                "processing_sequence": [
                    "Sort by quality grade",
                    "Process premium leaves first",
                    "Handle defective leaves separately"
                ],
                "quality_controls": [
                    "Implement quality checkpoints",
                    "Monitor processing parameters",
                    "Document quality metrics"
                ]
            },
            "waste_minimization": {
                "estimated_recoverable_value": f"{health_percentage:.1f}% of total value",
                "waste_reduction_strategies": [
                    "Immediate processing of healthy leaves",
                    "Alternative uses for lower grade leaves",
                    "Composting program for unusable material"
                ],
                "cost_optimization": [
                    "Prioritize high-value leaves",
                    "Batch similar quality grades",
                    "Minimize handling time"
                ]
            },
            "monitoring_recommendations": {
                "key_metrics": [
                    "Health percentage trends",
                    "Processing efficiency",
                    "Waste reduction rates"
                ],
                "alert_thresholds": {
                    "health_percentage_below": 60,
                    "processing_time_above": 300,
                    "waste_percentage_above": 30
                }
            }
        }
    
    def _get_priority_action(self, priority: str) -> str:
        """Get priority action description based on priority level."""
        actions = {
            "standard_processing": "Continue with standard processing procedures",
            "quality_monitoring": "Implement enhanced quality monitoring",
            "enhanced_sorting": "Deploy enhanced sorting and quality controls",
            "damage_control": "Immediate intervention required to minimize losses"
        }
        return actions.get(priority, "Review and assess situation")
    
    def get_analysis_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get history of analysis results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results
        """
        try:
            analysis_files = []
            
            if os.path.exists(self.analytics_dir):
                for filename in os.listdir(self.analytics_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.analytics_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                data['filename'] = filename
                                analysis_files.append(data)
                        except Exception as e:
                            logger.warning(f"Failed to read analysis file {filename}: {e}")
            
            # Sort by timestamp and limit results
            analysis_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return analysis_files[:limit]
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
