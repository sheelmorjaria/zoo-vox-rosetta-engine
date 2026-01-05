#!/usr/bin/env python3
"""
Test Real-Time System Dependencies
=================================

This script verifies that all required dependencies for the real-time
animal communication system are properly installed and configured.

Author: Sheel Morjaria
License: CC BY-ND 4.0 International
"""

import sys
import importlib
from typing import Dict, List, Tuple

def test_package_import(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """Test if a package can be imported optionally with version check."""
    try:
        module = importlib.import_module(package_name)

        if min_version is not None:
            version = getattr(module, '__version__', 'unknown')
            if version != 'unknown':
                # Simple version comparison
                version_parts = version.split('.')
                min_parts = min_version.split('.')

                for i, (v_part, min_part) in enumerate(zip(version_parts, min_parts)):
                    v_int = int(v_part) if v_part.isdigit() else 0
                    min_int = int(min_part) if min_part.isdigit() else 0

                    if v_int > min_int:
                        return True, f"✓ {package_name} v{version} (>= {min_version})"
                    elif v_int < min_int:
                        return False, f"✗ {package_name} v{version} (requires >= {min_version})"

                return True, f"✓ {package_name} v{version} (>= {min_version})"

        return True, f"✓ {package_name} imported successfully"

    except ImportError as e:
        return False, f"✗ {package_name} not installed: {str(e)}"
    except Exception as e:
        return False, f"✗ {package_name} import failed: {str(e)}"

def test_audio_interface() -> Dict[str, bool]:
    """Test audio interface capabilities."""
    results = {
        'microphone_input': False,
        'speaker_output': False,
        'low_latency': False
    }

    try:
        import sounddevice as sd

        # Test device query
        devices = sd.query_devices()
        if len(devices) > 0:
            print(f"Found {len(devices)} audio devices")

            # Check for input/output devices
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            output_devices = [d for d in devices if d['max_output_channels'] > 0]

            if len(input_devices) > 0:
                results['microphone_input'] = True
                print(f"✓ Microphone input available: {len(input_devices)} input devices")

            if len(output_devices) > 0:
                results['speaker_output'] = True
                print(f"✓ Speaker output available: {len(output_devices)} output devices")

            # Test low-latency capability
            try:
                # Test default device latency
                default_device = sd.default.device
                if default_device:
                    latency = sd.query_devices(default_device[0])['default_low_input_latency']
                    if latency and float(latency) < 0.1:  # < 100ms
                        results['low_latency'] = True
                        print(f"✓ Low-latency audio available: {latency}s latency")
            except:
                pass

    except ImportError:
        print("✗ SoundDevice not installed - cannot test audio interface")
    except Exception as e:
        print(f"✗ Audio interface test failed: {str(e)}")

    return results

def main():
    """Run all dependency tests."""
    print("=" * 80)
    print("REAL-TIME SYSTEM DEPENDENCY CHECK")
    print("=" * 80)

    # Core dependencies
    core_deps = [
        ('numpy', '2.0.0'),
        ('scipy', '1.10.0'),
        ('pandas', '2.0.0'),
        ('librosa', '0.10.0'),
        ('soundfile', '0.12.0'),
        ('torch', '2.0.0'),
        ('torchaudio', '2.0.0'),
    ]

    print("\nCORE DEPENDENCIES:")
    print("-" * 50)
    core_results = []
    for package, version in core_deps:
        success, message = test_package_import(package, version)
        core_results.append(success)
        print(f"  {message}")

    # Audio interface dependencies
    print("\nAUDIO INTERFACE DEPENDENCIES:")
    print("-" * 50)

    audio_deps = [
        ('sounddevice', '0.5.0'),
        ('pyaudio', '0.2.14'),
    ]

    audio_results = []
    for package, version in audio_deps:
        success, message = test_package_import(package, version)
        audio_results.append(success)
        print(f"  {message}")

    # Real-time processing dependencies
    print("\nREAL-TIME PROCESSING DEPENDENCIES:")
    print("-" * 50)

    realtime_deps = [
        ('psutil', '5.9.0'),
        ('tqdm', '4.65.0'),
    ]

    realtime_results = []
    for package, version in realtime_deps:
        success, message = test_package_import(package, version)
        realtime_results.append(success)
        print(f"  {message}")

    # Test audio interface
    print("\nAUDIO INTERFACE CAPABILITIES:")
    print("-" * 50)
    audio_capabilities = test_audio_interface()

    # Summary
    print("\n" + "=" * 80)
    print("DEPENDENCY SUMMARY")
    print("=" * 80)

    all_core_ok = all(core_results)
    all_audio_ok = all(audio_results)
    all_realtime_ok = all(realtime_results)
    all_audio_caps_ok = all(audio_capabilities.values())

    print(f"Core Dependencies:         {'✅ READY' if all_core_ok else '❌ NEEDS ATTENTION'}")
    print(f"Audio Interface:          {'✅ READY' if all_audio_ok else '❌ INSTALL sounddevice/pyaudio'}")
    print(f"Real-time Processing:     {'✅ READY' if all_realtime_ok else '❌ NEEDS ATTENTION'}")
    print(f"Audio Capabilities:       {'✅ READY' if all_audio_caps_ok else '❌ CONFIGURE AUDIO HARDWARE'}")

    overall_status = all_core_ok and all_realtime_ok

    print(f"\nOVERALL STATUS: {'✅ REAL-TIME SYSTEM READY' if overall_status else '❌ NEEDS SETUP'}")

    if not overall_status:
        print("\nTO ENABLE REAL-TIME FUNCTIONALITY:")
        print("1. Install system PortAudio library:")
        print("   Linux: sudo apt-get install portaudio19-dev")
        print("   macOS: brew install portaudio")
        print("   Windows: Download from https://www.portaudio.com/")
        print("\n2. Install missing Python packages:")
        print("   pip install -r requirements_realtime.txt")
        print("\n3. Configure audio hardware (microphone and speakers)")
        print("\n4. Test with: python3 test_realtime_dependencies.py")

    print("\n" + "=" * 80)
    return 0 if overall_status else 1

if __name__ == "__main__":
    sys.exit(main())