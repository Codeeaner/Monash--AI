import os
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.analytics_service import AnalyticsService
from app.models.detection import DetectionSession, DetectionResult
from app.models.analytics import (
    AnalysisResult, 
    BatchAnalysisResult, 
    WastePreventionRecommendation,
    QualityMetric,
    ProcessingAction
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])


@router.post("/analyze/{result_id}")
async def analyze_detection_result(
    result_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze a single detection result using Llama 3.2 Vision.
    
    Args:
        result_id: ID of the detection result to analyze
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Analysis initiation response
    """
    
    try:
        # Get detection result
        detection_result = db.query(DetectionResult).filter(
            DetectionResult.id == result_id
        ).first()
        
        if not detection_result:
            raise HTTPException(status_code=404, detail="Detection result not found")
        
        if not detection_result.annotated_image_path or not os.path.exists(detection_result.annotated_image_path):
            raise HTTPException(
                status_code=400, 
                detail="Annotated image not found for analysis"
            )
        
        # Check if analysis already exists
        existing_analysis = db.query(AnalysisResult).filter(
            AnalysisResult.detection_result_id == result_id
        ).first()
        
        if existing_analysis:
            return {
                "message": "Analysis already exists for this detection result",
                "analysis_id": existing_analysis.analysis_id,
                "status": existing_analysis.status
            }
        
        # Start analysis in background
        background_tasks.add_task(
            analyze_detection_background,
            result_id,
            db
        )
        
        return {
            "message": f"Analysis started for detection result {result_id}",
            "result_id": result_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for detection result {result_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch")
async def analyze_batch_results(
    session_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze all detection results in a session using Llama 3.2 Vision.
    
    Args:
        session_id: ID of the detection session
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Batch analysis initiation response
    """
    
    try:
        # Get detection session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Can only analyze completed sessions"
            )
        
        # Check if batch analysis already exists
        existing_batch_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.session_id == session_id
        ).first()
        
        if existing_batch_analysis:
            return {
                "message": "Batch analysis already exists for this session",
                "batch_analysis_id": existing_batch_analysis.batch_analysis_id,
                "status": existing_batch_analysis.status
            }
        
        # Start batch analysis in background
        background_tasks.add_task(
            analyze_batch_background,
            session_id,
            db
        )
        
        return {
            "message": f"Batch analysis started for session {session_id}",
            "session_id": session_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch/{batch_id}")
async def analyze_batch_results_by_id(
    batch_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze all detection results in a batch using Llama 3.2 Vision, identified by batch ID.
    
    Args:
        batch_id: ID of the batch (session ID)
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        Batch analysis initiation response
    """
    
    try:
        # Get detection session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == batch_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Batch session not found")
        
        if session.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Can only analyze completed batch sessions"
            )
        
        # Check if batch analysis already exists
        existing_batch_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.session_id == batch_id
        ).first()
        
        if existing_batch_analysis:
            return {
                "message": "Batch analysis already exists for this batch",
                "batch_analysis_id": existing_batch_analysis.batch_analysis_id,
                "status": existing_batch_analysis.status
            }
        
        # Start batch analysis in background
        background_tasks.add_task(
            analyze_batch_background,
            batch_id,
            db
        )
        
        return {
            "message": f"Batch analysis started for batch {batch_id}",
            "batch_id": batch_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch analysis for batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/results/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Get analysis result by analysis ID.
    
    Args:
        analysis_id: Analysis ID
        db: Database session
        
    Returns:
        Analysis result
    """
    
    try:
        analysis_result = db.query(AnalysisResult).filter(
            AnalysisResult.analysis_id == analysis_id
        ).first()
        
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        logger.info(f"Retrieved analysis result for analysis_id: {analysis_id}")
        return {
            "analysis_id": analysis_result.analysis_id,
            "status": analysis_result.status,
            "model_used": analysis_result.model_used,
            "processing_time": analysis_result.processing_time,
            "detection_summary": {
                "healthy_count": analysis_result.healthy_count,
                "unhealthy_count": analysis_result.unhealthy_count,
                "total_count": analysis_result.total_count,
                "health_percentage": analysis_result.health_percentage
            },
            "ai_analysis": analysis_result.ai_analysis,
            "error_message": analysis_result.error_message,
            "created_at": analysis_result.created_at,
            "image_path": analysis_result.image_path,
            "annotated_image_path": analysis_result.annotated_image_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis result {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/batch/{batch_analysis_id}")
async def get_batch_analysis_result(
    batch_analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Get batch analysis result.
    
    Args:
        batch_analysis_id: Batch analysis ID (can be UUID string or session ID)
        db: Database session
        
    Returns:
        Batch analysis result
    """
    
    try:
        batch_result = None
        # First, try to find by batch_analysis_id (UUID string)
        batch_result = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.batch_analysis_id == batch_analysis_id
        ).first()

        # If not found, and the input looks like an integer, try to find by session_id
        if not batch_result:
            try:
                session_id_as_int = int(batch_analysis_id)
                batch_result = db.query(BatchAnalysisResult).filter(
                    BatchAnalysisResult.session_id == session_id_as_int
                ).first()
            except ValueError:
                # The batch_analysis_id was not an integer, so it's not a session ID
                pass
        
        if not batch_result:
            raise HTTPException(status_code=404, detail="Batch analysis result not found")
        
        return {
            "batch_analysis_id": batch_result.batch_analysis_id,
            "session_id": batch_result.session_id,
            "status": batch_result.status,
            "total_images": batch_result.total_images,
            "analyzed_images": batch_result.analyzed_images,
            "processing_time": batch_result.processing_time,
            "batch_summary": {
                "total_healthy_leaves": batch_result.total_healthy_leaves,
                "total_unhealthy_leaves": batch_result.total_unhealthy_leaves,
                "total_leaves": batch_result.total_leaves,
                "overall_health_percentage": batch_result.overall_health_percentage
            },
            "aggregate_recommendations": batch_result.aggregate_recommendations,
            "individual_analysis_ids": batch_result.individual_analysis_ids,
            "created_at": batch_result.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch analysis result {batch_analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}")
async def get_session_analytics(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all analytics for a session.
    
    Args:
        session_id: Session ID
        db: Database session
        
    Returns:
        Session analytics summary
    """
    
    try:
        # Get session
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get individual analyses
        individual_analyses = db.query(AnalysisResult).filter(
            AnalysisResult.session_id == session_id
        ).all()
        
        # Get batch analysis
        batch_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.session_id == session_id
        ).first()
        
        # Get recommendations
        recommendations = []
        for analysis in individual_analyses:
            analysis_recommendations = db.query(WastePreventionRecommendation).filter(
                WastePreventionRecommendation.analysis_result_id == analysis.id
            ).all()
            recommendations.extend(analysis_recommendations)
        
        return {
            "session_id": session_id,
            "session_name": session.name,
            "individual_analyses": [
                {
                    "analysis_id": analysis.analysis_id,
                    "status": analysis.status,
                    "health_percentage": analysis.health_percentage,
                    "created_at": analysis.created_at
                } for analysis in individual_analyses
            ],
            "batch_analysis": {
                "batch_analysis_id": str(batch_analysis.batch_analysis_id) if batch_analysis and isinstance(batch_analysis.batch_analysis_id, (str, int)) else None,
                "status": batch_analysis.status,
                "overall_health_percentage": batch_analysis.overall_health_percentage,
                "created_at": batch_analysis.created_at
            } if batch_analysis else None,
            "total_recommendations": len(recommendations),
            "pending_recommendations": len([r for r in recommendations if r.status == "pending"]),
            "implemented_recommendations": len([r for r in recommendations if r.status == "implemented"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session analytics {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendations/{analysis_id}")
async def get_recommendations(
    analysis_id: str,
    priority: Optional[str] = Query(None, description="Filter by priority level"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """
    Get waste prevention recommendations for an analysis.
    
    Args:
        analysis_id: Analysis ID
        priority: Filter by priority level
        status: Filter by status
        db: Database session
        
    Returns:
        List of recommendations
    """
    
    try:
        # Get analysis result
        analysis_result = db.query(AnalysisResult).filter(
            AnalysisResult.analysis_id == analysis_id
        ).first()
        
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis result not found")
        
        # Build query
        query = db.query(WastePreventionRecommendation).filter(
            WastePreventionRecommendation.analysis_result_id == analysis_result.id
        )
        
        if priority:
            query = query.filter(WastePreventionRecommendation.priority_level == priority)
        
        if status:
            query = query.filter(WastePreventionRecommendation.status == status)
        
        recommendations = query.all()
        
        return [
            {
                "id": rec.id,
                "recommendation_type": rec.recommendation_type,
                "priority_level": rec.priority_level,
                "title": rec.title,
                "description": rec.description,
                "estimated_cost_saving": rec.estimated_cost_saving,
                "implementation_effort": rec.implementation_effort,
                "expected_outcome": rec.expected_outcome,
                "status": rec.status,
                "notes": rec.notes,
                "created_at": rec.created_at,
                "implemented_at": rec.implemented_at
            } for rec in recommendations
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history")
async def get_analytics_history(
    limit: int = Query(20, description="Maximum number of results"),
    session_id: Optional[int] = Query(None, description="Filter by session ID"),
    db: Session = Depends(get_db)
):
    """
    Get analytics history.
    
    Args:
        limit: Maximum number of results
        session_id: Filter by session ID
        db: Database session
        
    Returns:
        List of analytics results
    """
    
    try:
        # Build query for individual analyses
        query = db.query(AnalysisResult)
        
        if session_id:
            query = query.filter(AnalysisResult.session_id == session_id)
        
        analyses = query.order_by(AnalysisResult.created_at.desc()).limit(limit).all()
        
        # Build query for batch analyses
        batch_query = db.query(BatchAnalysisResult)
        
        if session_id:
            batch_query = batch_query.filter(BatchAnalysisResult.session_id == session_id)
        
        batch_analyses = batch_query.order_by(BatchAnalysisResult.created_at.desc()).limit(limit).all()
        
        return {
            "individual_analyses": [
                {
                    "analysis_id": analysis.analysis_id,
                    "session_id": analysis.session_id,
                    "status": analysis.status,
                    "model_used": analysis.model_used,
                    "health_percentage": analysis.health_percentage,
                    "processing_time": analysis.processing_time,
                    "created_at": analysis.created_at
                } for analysis in analyses
            ],
            "batch_analyses": [
                {
                    "batch_analysis_id": batch.batch_analysis_id,
                    "session_id": batch.session_id,
                    "status": batch.status,
                    "total_images": batch.total_images,
                    "analyzed_images": batch.analyzed_images,
                    "overall_health_percentage": batch.overall_health_percentage,
                    "processing_time": batch.processing_time,
                    "created_at": batch.created_at
                } for batch in batch_analyses
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/recommendations/{recommendation_id}/implement")
async def implement_recommendation(
    recommendation_id: int,
    notes: str = "",
    db: Session = Depends(get_db)
):
    """
    Mark a recommendation as implemented.
    
    Args:
        recommendation_id: Recommendation ID
        notes: Implementation notes
        db: Database session
        
    Returns:
        Updated recommendation
    """
    
    try:
        recommendation = db.query(WastePreventionRecommendation).filter(
            WastePreventionRecommendation.id == recommendation_id
        ).first()
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        recommendation.status = "implemented"
        recommendation.notes = notes
        recommendation.implemented_at = datetime.utcnow()
        
        db.commit()
        db.refresh(recommendation)
        
        return {
            "message": "Recommendation marked as implemented",
            "recommendation_id": recommendation_id,
            "status": recommendation.status,
            "implemented_at": recommendation.implemented_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error implementing recommendation {recommendation_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


def analyze_detection_background(result_id: int, db: Session):
    """Background task to analyze a detection result."""
    
    try:
        analytics_service = AnalyticsService()
        
        # Get detection result
        detection_result = db.query(DetectionResult).filter(
            DetectionResult.id == result_id
        ).first()
        
        if not detection_result:
            logger.error(f"Detection result {result_id} not found")
            return
        
        # Prepare detection data
        detection_data = {
            "healthy_count": detection_result.healthy_count,
            "unhealthy_count": detection_result.unhealthy_count,
            "total_count": detection_result.total_count,
            "health_percentage": detection_result.health_percentage,
            "boxes": []  # Could be populated from detection_boxes if needed
        }
        
        # Run analysis
        analysis_result = analytics_service.analyze_detection_result(
            detection_data, 
            detection_result.annotated_image_path
        )
        
        # Save to database
        db_analysis = AnalysisResult(
            analysis_id=analysis_result["analysis_id"],
            detection_result_id=result_id,
            session_id=detection_result.session_id,
            model_used=analysis_result["model_used"],
            processing_time=analysis_result["processing_time"],
            status=analysis_result["status"],
            image_path=detection_result.image_path,
            annotated_image_path=detection_result.annotated_image_path,
            healthy_count=detection_result.healthy_count,
            unhealthy_count=detection_result.unhealthy_count,
            total_count=detection_result.total_count,
            health_percentage=detection_result.health_percentage,
            ai_analysis=analysis_result["ai_analysis"]
        )
        
        db.add(db_analysis)
        db.commit()
        
        logger.info(f"Analysis completed for detection result {result_id}")
        
    except Exception as e:
        logger.error(f"Error in background analysis for result {result_id}: {e}")
        db.rollback()


def analyze_batch_background(session_id: int, db: Session):
    """Background task to analyze a batch of detection results."""
    
    try:
        analytics_service = AnalyticsService()
        
        # Get session and results
        session = db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            logger.error(f"Session {session_id} not found")
            return
        
        results = db.query(DetectionResult).filter(
            DetectionResult.session_id == session_id,
            DetectionResult.status == "completed"
        ).all()
        
        logger.info(f"Found {len(results)} completed results for session {session_id}")
        
        # Prepare batch data
        batch_data = []
        for result in results:
            batch_data.append({
                "healthy_count": result.healthy_count,
                "unhealthy_count": result.unhealthy_count,
                "total_count": result.total_count,
                "health_percentage": result.health_percentage,
                "annotated_image_path": result.annotated_image_path,
                "image_name": result.image_name
            })
        
        # Run batch analysis
        logger.info(f"Starting batch analysis for session {session_id}")
        batch_analysis = analytics_service.analyze_batch_results(batch_data)
        logger.info(f"Batch analysis completed with ID {batch_analysis.get('batch_analysis_id', 'unknown')}")
        
        # Save to database
        db_batch_analysis = BatchAnalysisResult(
            batch_analysis_id=batch_analysis["batch_analysis_id"],
            session_id=session_id,
            total_images=batch_analysis["batch_summary"]["total_images"],
            analyzed_images=batch_analysis["batch_summary"]["analyzed_images"],
            processing_time=batch_analysis["processing_time"],
            status=batch_analysis["status"],
            total_healthy_leaves=batch_analysis["batch_summary"]["total_healthy_leaves"],
            total_unhealthy_leaves=batch_analysis["batch_summary"]["total_unhealthy_leaves"],
            total_leaves=batch_analysis["batch_summary"]["total_leaves"],
            overall_health_percentage=batch_analysis["batch_summary"]["overall_health_percentage"],
            aggregate_recommendations=batch_analysis["aggregate_recommendations"],
            individual_analysis_ids=[a["analysis_id"] for a in batch_analysis.get("individual_analyses", [])]
        )
        
        logger.info(f"Adding batch analysis {batch_analysis['batch_analysis_id']} for session {session_id} to database")
        db.add(db_batch_analysis)
        db.commit()
        logger.info(f"Successfully committed batch analysis {batch_analysis['batch_analysis_id']} for session {session_id} to database")
        
        # Verify the save operation
        saved_analysis = db.query(BatchAnalysisResult).filter(
            BatchAnalysisResult.batch_analysis_id == batch_analysis["batch_analysis_id"]
        ).first()
        if saved_analysis:
            logger.info(f"Verified batch analysis {batch_analysis['batch_analysis_id']} exists in database for session {session_id}")
        else:
            logger.error(f"Failed to verify batch analysis {batch_analysis['batch_analysis_id']} in database for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in background batch analysis for session {session_id}: {e}", exc_info=True)
        db.rollback()
        logger.info(f"Rollback completed for session {session_id} due to error")
