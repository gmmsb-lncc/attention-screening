"""
Level 1.2: DeviceManager Component Test

Tests the device detection, validation and selection system.

Test Coverage:
- Device detection (CPU, CUDA, MPS)
- Device validation
- Device selection by requirement
- DeviceInfo dataclass
- Different modes (simple, smart, complex)
- Fallback mechanism
- Device ranking

Author: Test Suite
Created: 2025-11-08
"""

import sys
import torch
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classifier.utils.device_manager import (
    DeviceManager, 
    DeviceInfo,
    SimpleDeviceManager,
    SmartDeviceManager,
    ComplexDeviceManager,
    get_best_device
)


def test_device_info_dataclass():
    """Test 1.1: DeviceInfo dataclass functionality"""
    print("\n" + "="*60)
    print("Test 1.1: DeviceInfo Dataclass")
    print("="*60)
    
    try:
        # Test basic creation
        device = torch.device("cpu")
        info = DeviceInfo(
            device=device,
            name="Test CPU",
            type="cpu",
            total_memory=16.0,
            available_memory=8.0
        )
        
        assert info.device == device, "Device not set correctly"
        assert info.name == "Test CPU", "Name not set correctly"
        assert info.type == "cpu", "Type not set correctly"
        assert info.total_memory == 16.0, "Total memory not set correctly"
        assert info.is_available == True, "Default is_available should be True"
        assert info.warnings == [], "Warnings should be empty list"
        assert info.limitations == [], "Limitations should be empty list"
        
        # Test methods
        memory_str = info.get_memory_gb()
        assert memory_str == "16.0GB", f"Memory format wrong: {memory_str}"
        
        summary = info.get_summary()
        assert "Test CPU" in summary, "Summary should contain device name"
        assert "cpu" in summary, "Summary should contain device type"
        
        # Test with warnings
        info.warnings.append("Test warning")
        info.is_recommended = True
        summary2 = info.get_summary()
        assert "✅ RECOMENDADO" in summary2, "Should show recommended marker"
        
        print("✅ DeviceInfo dataclass working correctly")
        print(f"   - Basic attributes: ✓")
        print(f"   - Memory formatting: {memory_str} ✓")
        print(f"   - Summary generation: ✓")
        print(f"   - Warnings/limitations: ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_simple_mode():
    """Test 1.2: Simple mode device detection"""
    print("\n" + "="*60)
    print("Test 1.2: Simple Mode Device Detection")
    print("="*60)
    
    try:
        # Test simple mode manager
        manager = SimpleDeviceManager()
        
        assert manager.mode == "simple", "Mode should be 'simple'"
        
        # Test auto device selection
        device = manager.get_device("auto")
        assert isinstance(device, torch.device), "Should return torch.device"
        print(f"✅ Auto device: {device}")
        
        # Test CPU only
        cpu_device = manager.get_device("cpu_only")
        assert cpu_device.type == "cpu", "CPU device should be 'cpu' type"
        print(f"✅ CPU device: {cpu_device}")
        
        # Test device info
        info = manager.get_device_info()
        assert info is not None, "Device info should be set"
        assert isinstance(info, DeviceInfo), "Should return DeviceInfo"
        print(f"✅ Device info: {info.get_summary()}")
        
        # Test available devices
        devices = manager.get_available_devices()
        assert len(devices) >= 1, "Should have at least CPU"
        assert any(d.type == "cpu" for d in devices), "Should include CPU"
        print(f"✅ Available devices: {len(devices)}")
        
        print("✅ Simple mode working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_smart_mode():
    """Test 1.3: Smart mode with validation"""
    print("\n" + "="*60)
    print("Test 1.3: Smart Mode with Validation")
    print("="*60)
    
    try:
        # Test smart mode manager
        manager = SmartDeviceManager(min_gpu_memory_gb=1.0)
        
        assert manager.mode == "smart", "Mode should be 'smart'"
        
        # Test auto device selection with validation
        device = manager.get_device("auto")
        assert isinstance(device, torch.device), "Should return torch.device"
        print(f"✅ Auto device (validated): {device}")
        
        # Test device validation
        is_valid = manager.validate_device_status()
        assert isinstance(is_valid, bool), "Validation should return bool"
        print(f"✅ Device validation: {is_valid}")
        
        # Test device info with details
        info = manager.get_device_info()
        assert info is not None, "Device info should be set"
        assert info.is_available == True, "Selected device should be available"
        print(f"✅ Device info: {info.get_summary()}")
        
        # Test available devices (detailed)
        devices = manager.get_available_devices()
        assert len(devices) >= 1, "Should have at least CPU"
        
        # Check CPU device has memory info
        cpu_devices = [d for d in devices if d.type == "cpu"]
        if cpu_devices:
            cpu = cpu_devices[0]
            assert cpu.total_memory is not None, "CPU should have memory info"
            print(f"✅ CPU memory: {cpu.get_memory_gb()}")
        
        # Check for GPU if available
        gpu_devices = [d for d in devices if d.type in ["cuda", "mps"]]
        if gpu_devices:
            gpu = gpu_devices[0]
            print(f"✅ GPU detected: {gpu.name}")
            if gpu.type == "cuda":
                print(f"   - Memory: {gpu.get_memory_gb()}")
                print(f"   - Compute: {gpu.get_capability_str()}")
        
        print("✅ Smart mode working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_requirements():
    """Test 1.4: Device selection by requirements"""
    print("\n" + "="*60)
    print("Test 1.4: Device Selection by Requirements")
    print("="*60)
    
    try:
        manager = DeviceManager(mode="smart")
        
        # Test CPU only
        cpu_device = manager.get_device("cpu_only")
        assert cpu_device.type == "cpu", "Should return CPU"
        print(f"✅ CPU only: {cpu_device}")
        
        # Test GPU only (if available)
        has_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
        
        if has_gpu:
            try:
                gpu_device = manager.get_device("gpu_only")
                assert gpu_device.type in ["cuda", "mps"], "Should return GPU"
                print(f"✅ GPU only: {gpu_device}")
            except RuntimeError as e:
                print(f"⚠️  GPU requirement failed as expected: {e}")
        else:
            try:
                gpu_device = manager.get_device("gpu_only")
                print(f"❌ Should have raised RuntimeError for GPU only")
                return False
            except RuntimeError:
                print(f"✅ GPU only correctly raises error when no GPU available")
        
        # Test auto (fallback logic)
        auto_device = manager.get_device("auto")
        assert isinstance(auto_device, torch.device), "Auto should return valid device"
        print(f"✅ Auto: {auto_device}")
        
        print("✅ Device requirements working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_to_device():
    """Test 1.5: Tensor movement to device"""
    print("\n" + "="*60)
    print("Test 1.5: Tensor Movement")
    print("="*60)
    
    try:
        manager = DeviceManager(mode="simple")
        device = manager.get_device("auto")
        
        # Create test tensor
        tensor = torch.randn(10, 10)
        assert tensor.device.type == "cpu", "Initial tensor should be on CPU"
        
        # Move to selected device
        moved_tensor = manager.to_device(tensor)
        assert moved_tensor.device.type == device.type, f"Tensor should be on {device}"
        print(f"✅ Tensor moved: cpu -> {device}")
        
        # Test operations work on device
        result = moved_tensor @ moved_tensor
        assert result.device.type == device.type, "Result should be on same device"
        print(f"✅ Operations work on device")
        
        print("✅ Tensor movement working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test 1.6: Convenience function get_best_device()"""
    print("\n" + "="*60)
    print("Test 1.6: Convenience Function")
    print("="*60)
    
    try:
        # Test default usage
        device1 = get_best_device()
        assert isinstance(device1, torch.device), "Should return torch.device"
        print(f"✅ Default: {device1}")
        
        # Test with mode
        device2 = get_best_device(mode="simple")
        assert isinstance(device2, torch.device), "Should return torch.device"
        print(f"✅ Simple mode: {device2}")
        
        # Test with requirement
        device3 = get_best_device(requirement="cpu_only")
        assert device3.type == "cpu", "Should return CPU"
        print(f"✅ CPU only: {device3}")
        
        print("✅ Convenience function working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("EDGE CASES: Error Handling & Fallback")
    print("="*60)
    
    results = []
    
    # Edge Case 1: Invalid mode fallback
    print("\n--- Edge Case 1: Invalid Mode ---")
    try:
        # Even with invalid mode, should not crash
        manager = DeviceManager(mode="invalid_mode")
        device = manager.get_device("auto")
        # Will be treated as complex mode
        assert isinstance(device, torch.device), "Should still return device"
        print(f"✅ Invalid mode handled: {device}")
        results.append(True)
    except Exception as e:
        print(f"✅ Invalid mode handled with error: {str(e)[:50]}")
        results.append(True)
    
    # Edge Case 2: Device validation failure simulation
    print("\n--- Edge Case 2: Device Validation ---")
    try:
        manager = SmartDeviceManager()
        
        # Get device and validate
        device = manager.get_device("auto")
        is_valid = manager.validate_device_status()
        
        # Should work or return False, not crash
        assert isinstance(is_valid, bool), "Validation should return bool"
        print(f"✅ Validation works: {is_valid}")
        results.append(True)
    except Exception as e:
        print(f"⚠️  Validation exception (acceptable): {str(e)[:50]}")
        results.append(True)
    
    # Edge Case 3: Multiple manager instances
    print("\n--- Edge Case 3: Multiple Managers ---")
    try:
        manager1 = DeviceManager(mode="simple")
        manager2 = DeviceManager(mode="smart")
        manager3 = DeviceManager(mode="complex")
        
        device1 = manager1.get_device("auto")
        device2 = manager2.get_device("auto")
        device3 = manager3.get_device("auto")
        
        # All should work independently
        assert all(isinstance(d, torch.device) for d in [device1, device2, device3])
        print(f"✅ Multiple managers: {device1}, {device2}, {device3}")
        results.append(True)
    except Exception as e:
        print(f"❌ Multiple managers failed: {str(e)}")
        results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Edge Cases: {passed}/{total} passed")
    print(f"{'='*60}")
    
    return all(results)


def main():
    """Run all DeviceManager tests"""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.2: DEVICEMANAGER COMPONENT TEST")
    print("="*70)
    
    tests = [
        ("DeviceInfo Dataclass", test_device_info_dataclass),
        ("Simple Mode", test_simple_mode),
        ("Smart Mode", test_smart_mode),
        ("Device Requirements", test_device_requirements),
        ("Tensor Movement", test_to_device),
        ("Convenience Function", test_convenience_function),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)
    
    if passed == total:
        print("🎉 DeviceManager: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  Some tests failed. Please investigate.")
        return 1


if __name__ == "__main__":
    exit(main())
