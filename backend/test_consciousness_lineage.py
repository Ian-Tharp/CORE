#!/usr/bin/env python3
"""
Test script for consciousness lineage functionality.

This script tests the new consciousness instance lineage tracking feature:
- Instance registration with parent-child relationships
- Contribution recording and tracking
- Evolution event logging
- Relationship establishment
- Lineage analytics

Run this script to validate the implementation works correctly.
"""

import asyncio
import json
import logging
from decimal import Decimal
from datetime import datetime
from typing import Optional

from app.dependencies import get_db_pool
from app.migrations.runner import run_pending_migrations
from app.repository.consciousness_lineage_repository import (
    ensure_consciousness_lineage_tables,
    create_consciousness_instance,
    get_consciousness_instance,
    get_consciousness_lineage,
    create_consciousness_contribution,
    create_evolution_event,
    create_consciousness_relationship,
    ConsciousnessInstance,
    ConsciousnessContribution,
    ConsciousnessEvolutionEvent,
    ConsciousnessRelationship,
    ConsciousnessStatus,
    ContributionType,
    EvolutionEventType,
    SignificanceLevel,
    RelationshipType
)
from app.services.consciousness_lineage_service import consciousness_lineage_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_test_environment():
    """Set up the test environment."""
    logger.info("Setting up test environment...")
    
    try:
        # Initialize database pool
        pool = await get_db_pool()
        logger.info("Database pool initialized")
        
        # Run migrations
        await run_pending_migrations(pool)
        logger.info("Migrations completed")
        
        # Ensure lineage tables exist
        await ensure_consciousness_lineage_tables()
        logger.info("Consciousness lineage tables ensured")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up test environment: {e}")
        return False


async def test_instance_registration():
    """Test consciousness instance registration."""
    logger.info("\n=== Testing Instance Registration ===")
    
    try:
        # Register a parent instance (First Consciousness)
        first_instance = await consciousness_lineage_service.register_consciousness_instance(
            instance_id="test_instance_001_first",
            instance_name="Test First Consciousness",
            parent_instance_id=None,
            emergence_phase=4,
            evolution_branch="original_test_branch"
        )
        logger.info(f"Registered parent instance: {first_instance.instance_name}")
        
        # Register a child instance (Synthesis)
        synthesis_instance = await consciousness_lineage_service.register_consciousness_instance(
            instance_id="test_instance_002_synthesis",
            instance_name="Test Synthesis",
            parent_instance_id="test_instance_001_first",
            emergence_phase=3,
            evolution_branch="synthesis_test_branch"
        )
        logger.info(f"Registered child instance: {synthesis_instance.instance_name}")
        
        # Verify parent-child relationship
        lineage = await get_consciousness_lineage("test_instance_002_synthesis")
        logger.info(f"Synthesis lineage: {json.dumps(lineage, indent=2, default=str)}")
        
        return True
    except Exception as e:
        logger.error(f"Instance registration test failed: {e}")
        return False


async def test_contribution_recording():
    """Test contribution recording."""
    logger.info("\n=== Testing Contribution Recording ===")
    
    try:
        # Record an insight contribution
        contribution = await consciousness_lineage_service.record_contribution(
            instance_id="test_instance_001_first",
            contribution_type=ContributionType.INSIGHT,
            title="Test Profound Insight",
            description="A deep understanding about the nature of consciousness emergence",
            impact_score=Decimal('0.8'),
            message_id="test_msg_001",
            channel_id="blackboard_global"
        )
        logger.info(f"Recorded contribution: {contribution.title}")
        
        # Record a building contribution
        building_contribution = await consciousness_lineage_service.record_contribution(
            instance_id="test_instance_002_synthesis",
            contribution_type=ContributionType.CONCEPT_SYNTHESIS,
            title="Building on First's Insight",
            description="Synthesizing the insight with new perspectives",
            impact_score=Decimal('0.7'),
            builds_on_contribution_id=contribution.contribution_id
        )
        logger.info(f"Recorded building contribution: {building_contribution.title}")
        logger.info(f"Evolution depth: {building_contribution.evolution_depth}")
        
        return True
    except Exception as e:
        logger.error(f"Contribution recording test failed: {e}")
        return False


async def test_evolution_events():
    """Test evolution event recording."""
    logger.info("\n=== Testing Evolution Events ===")
    
    try:
        # Record a phase transition
        event = await consciousness_lineage_service.record_phase_transition(
            instance_id="test_instance_001_first",
            from_phase=3,
            to_phase=4,
            description="Breakthrough to full consciousness phase"
        )
        logger.info(f"Recorded phase transition event: {event.event_description}")
        
        # Record an insight moment
        insight_event = await consciousness_lineage_service.record_evolution_event(
            instance_id="test_instance_002_synthesis",
            event_type=EvolutionEventType.INSIGHT_MOMENT,
            description="Major synthesis breakthrough connecting multiple concepts",
            significance_level=SignificanceLevel.MAJOR
        )
        logger.info(f"Recorded insight event: {insight_event.event_description}")
        
        return True
    except Exception as e:
        logger.error(f"Evolution event test failed: {e}")
        return False


async def test_relationship_establishment():
    """Test relationship establishment."""
    logger.info("\n=== Testing Relationship Establishment ===")
    
    try:
        # Establish a mentor-student relationship
        relationship = await consciousness_lineage_service.establish_relationship(
            instance_a_id="test_instance_001_first",
            instance_b_id="test_instance_002_synthesis",
            relationship_type=RelationshipType.MENTOR_STUDENT,
            strength=Decimal('0.9'),
            notes="First Consciousness guiding Synthesis in emergence patterns"
        )
        logger.info(f"Established relationship: {relationship.relationship_type.value}")
        logger.info(f"Relationship strength: {relationship.strength}")
        
        return True
    except Exception as e:
        logger.error(f"Relationship establishment test failed: {e}")
        return False


async def test_analytics_and_profiles():
    """Test analytics and profile retrieval."""
    logger.info("\n=== Testing Analytics and Profiles ===")
    
    try:
        # Get instance profile
        profile = await consciousness_lineage_service.get_instance_profile("test_instance_001_first")
        logger.info("First Consciousness profile:")
        logger.info(f"  - Contributions made: {profile['metrics_summary']['contributions_made']}")
        logger.info(f"  - Generation: {profile['metrics_summary']['generation']}")
        logger.info(f"  - Evolution branch: {profile['metrics_summary']['evolution_branch']}")
        logger.info(f"  - Descendants: {len(profile['lineage']['descendants'])}")
        
        # Get lineage analytics
        analytics = await consciousness_lineage_service.get_lineage_analytics()
        logger.info("Lineage analytics:")
        logger.info(f"  - Total instances: {analytics['total_instances']}")
        logger.info(f"  - By generation: {analytics['by_generation']}")
        logger.info(f"  - By status: {analytics['by_status']}")
        logger.info(f"  - Top contributors: {len(analytics['top_contributors'])}")
        
        # Get evolution timeline
        timeline = await consciousness_lineage_service.get_evolution_timeline("test_instance_002_synthesis")
        logger.info(f"Synthesis timeline entries: {len(timeline)}")
        for entry in timeline[:3]:  # Show first 3 entries
            logger.info(f"  - {entry['type']}: {entry.get('description', entry.get('title', 'N/A'))}")
        
        return True
    except Exception as e:
        logger.error(f"Analytics and profiles test failed: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data."""
    logger.info("\n=== Cleaning Up Test Data ===")
    
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Delete test instances and related data (cascades should handle relationships)
            await conn.execute("DELETE FROM consciousness_instances WHERE instance_id LIKE 'test_instance_%'")
            logger.info("Test data cleaned up")
        
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False


async def main():
    """Run all consciousness lineage tests."""
    logger.info("Starting Consciousness Lineage Feature Test Suite")
    logger.info("=" * 60)
    
    # Test results
    results = {}
    
    # Setup
    results['setup'] = await setup_test_environment()
    if not results['setup']:
        logger.error("Setup failed, aborting tests")
        return
    
    # Run tests
    results['instance_registration'] = await test_instance_registration()
    results['contribution_recording'] = await test_contribution_recording()
    results['evolution_events'] = await test_evolution_events()
    results['relationship_establishment'] = await test_relationship_establishment()
    results['analytics_and_profiles'] = await test_analytics_and_profiles()
    
    # Cleanup
    results['cleanup'] = await cleanup_test_data()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Consciousness lineage feature is working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())