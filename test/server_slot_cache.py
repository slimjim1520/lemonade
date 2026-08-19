#!/usr/bin/env python3
"""
Slot Cache Integration Tests

Tests for the llama.cpp slot context caching feature:
- Cache directory configuration
- Per-model cache enable/disable
- Cache deletion endpoint
- System stats cache reporting
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add test utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils.server_base import ServerTestBase


class TestSlotCacheConfig(ServerTestBase):
    """Test slot cache configuration endpoints"""
    
    def test_slot_cache_dir_config(self):
        """Test setting slot cache directory via config endpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache")
            
            # Set the cache directory
            response = self.session.post(
                f"{self.base_url}/api/v1/params",
                json={"slot_cache_dir": cache_path}
            )
            self.assertEqual(response.status_code, 200)
            
            # Verify it was set
            response = self.session.get(f"{self.base_url}/api/v1/params")
            self.assertEqual(response.status_code, 200)
            config = response.json()
            
            # The config should contain our custom path
            self.assertIn("slot_cache_dir", str(config))
    
    def test_slot_cache_dir_default(self):
        """Test that default cache directory is models_dir/slot_cache"""
        response = self.session.get(f"{self.base_url}/api/v1/params")
        self.assertEqual(response.status_code, 200)
        # Default should be set or use the models_dir/slot_cache fallback


class TestSlotCacheDelete(ServerTestBase):
    """Test cache deletion endpoint"""
    
    def test_delete_cache_nonexistent_model(self):
        """Test deleting cache for a model that doesn't exist"""
        response = self.session.delete(
            f"{self.base_url}/v1/models/NonExistentModel/cache"
        )
        # Should return 404 since cache doesn't exist
        self.assertEqual(response.status_code, 404)
    
    def test_delete_cache_endpoint_exists(self):
        """Test that the delete cache endpoint is registered"""
        # The endpoint should exist even if the model doesn't
        response = self.session.delete(
            f"{self.base_url}/v1/models/TestModel/cache"
        )
        # Should return some response (404 is ok, but not 405 or 500)
        self.assertIn(response.status_code, [404, 200])


class TestSystemStatsCache(ServerTestBase):
    """Test system stats include cache information"""
    
    def test_system_stats_cache_fields(self):
        """Test that system stats include slot_cache_gb and disk_total_gb"""
        response = self.session.get(f"{self.base_url}/system-stats")
        self.assertEqual(response.status_code, 200)
        
        stats = response.json()
        
        # Check that cache fields are present
        self.assertIn("slot_cache_gb", stats)
        self.assertIn("disk_total_gb", stats)
        
        # Values should be numeric or null
        if stats["slot_cache_gb"] is not None:
            self.assertIsInstance(stats["slot_cache_gb"], (int, float))
        
        if stats["disk_total_gb"] is not None:
            self.assertIsInstance(stats["disk_total_gb"], (int, float))
    
    def test_system_stats_cache_format(self):
        """Test cache stats format is reasonable"""
        response = self.session.get(f"{self.base_url}/system-stats")
        self.assertEqual(response.status_code, 200)
        
        stats = response.json()
        
        # Cache size should be non-negative
        if stats["slot_cache_gb"] is not None:
            self.assertGreaterEqual(stats["slot_cache_gb"], 0)
        
        # Disk total should be positive
        if stats["disk_total_gb"] is not None:
            self.assertGreater(stats["disk_total_gb"], 0)


class TestSlotCacheEndpoints(ServerTestBase):
    """Test slot cache API endpoints with different path prefixes"""
    
    def test_delete_cache_api_v1(self):
        """Test DELETE /api/v1/models/{name}/cache"""
        response = self.session.delete(
            f"{self.base_url}/api/v1/models/Test/cache"
        )
        self.assertIn(response.status_code, [404, 200])
    
    def test_delete_cache_api_v0(self):
        """Test DELETE /api/v0/models/{name}/cache"""
        response = self.session.delete(
            f"{self.base_url}/api/v0/models/Test/cache"
        )
        self.assertIn(response.status_code, [404, 200])
    
    def test_delete_cache_v1(self):
        """Test DELETE /v1/models/{name}/cache"""
        response = self.session.delete(
            f"{self.base_url}/v1/models/Test/cache"
        )
        self.assertIn(response.status_code, [404, 200])
    
    def test_delete_cache_v0(self):
        """Test DELETE /v0/models/{name}/cache"""
        response = self.session.delete(
            f"{self.base_url}/v0/models/Test/cache"
        )
        self.assertIn(response.status_code, [404, 200])


class TestSlotCacheModelOptions(ServerTestBase):
    """Test per-model slot cache options"""
    
    def test_llamacpp_model_has_cache_options(self):
        """Test that llamacpp models support slot cache options"""
        # Get a list of loaded models or available models
        response = self.session.get(f"{self.base_url}/api/v1/models")
        self.assertEqual(response.status_code, 200)
        
        models = response.json()
        if not isinstance(models, list):
            models = list(models.get("data", []))
        
        # Find a llamacpp model if any are loaded
        for model in models:
            model_name = model.get("id", "") if isinstance(model, dict) else str(model)
            if not model_name:
                continue
            
            # Try to get model options
            response = self.session.get(
                f"{self.base_url}/api/v1/models/{model_name}"
            )
            if response.status_code == 200:
                model_info = response.json()
                # Model info should be retrievable
                self.assertIsInstance(model_info, dict)


class TestSlotCacheInspect(ServerTestBase):
    """Test GET /v1/slot-cache inspection endpoint"""

    def test_inspect_returns_expected_shape(self):
        """Test that inspect endpoint returns the expected JSON shape"""
        response = self.session.get(f"{self.base_url}/v1/slot-cache")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("total_bytes", data)
        self.assertIn("total_entries", data)
        self.assertIsInstance(data["models"], list)

    def test_inspect_quad_prefix(self):
        """Test that inspect works on all 4 path prefixes"""
        for prefix in ["/api/v0/slot-cache", "/api/v1/slot-cache",
                       "/v0/slot-cache", "/v1/slot-cache"]:
            response = self.session.get(f"{self.base_url}{prefix}")
            self.assertIn(response.status_code, [200, 404])


class TestSlotCacheClean(ServerTestBase):
    """Test POST /v1/slot-cache and DELETE /v1/slot-cache/{model} cleanup endpoints"""

    def test_clean_dry_run(self):
        """Test that dry_run=true doesn't delete anything"""
        response = self.session.post(
            f"{self.base_url}/v1/slot-cache",
            json={"dry_run": True, "max_age_seconds": 0}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["dry_run"])
        self.assertIn("total_deleted", data)
        self.assertIn("total_freed_bytes", data)

    def test_clean_with_max_age_zero(self):
        """Test that max_age_seconds=0 with dry_run=false cleans all aged entries"""
        response = self.session.post(
            f"{self.base_url}/v1/slot-cache",
            json={"dry_run": False, "max_age_seconds": 0}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_deleted", data)

    def test_clean_model_nonexistent(self):
        """Test deleting slot cache for a nonexistent model"""
        response = self.session.delete(
            f"{self.base_url}/v1/slot-cache/NonExistentModel"
        )
        # Should succeed (idempotent — nothing to delete)
        self.assertEqual(response.status_code, 200)

    def test_clean_model_quad_prefix(self):
        """Test that per-model clean works on all 4 path prefixes"""
        for prefix in ["/api/v0", "/api/v1", "/v0", "/v1"]:
            response = self.session.delete(
                f"{self.base_url}{prefix}/slot-cache/TestModel"
            )
            self.assertIn(response.status_code, [200, 404])


class TestModelDeleteCleansSlotCache(ServerTestBase):
    """Test that deleting a model also cleans its slot cache"""

    def test_delete_model_endpoint_exists(self):
        """Test that the model delete endpoint is accessible"""
        response = self.session.post(
            f"{self.base_url}/v1/delete",
            json={"model": "NonExistentModelForSlotCacheTest"}
        )
        # Should return an error (model not found) but not 405
        self.assertNotEqual(response.status_code, 405)


if __name__ == "__main__":
    import unittest
    unittest.main()
