"""
Test script to verify the Call Center AI Assistant system setup.
Run this after installing dependencies to ensure everything works.
"""
import sys
import os


def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    required_packages = [
        ('langgraph', 'langgraph'),
        ('langchain', 'langchain'),
        ('langchain_openai', 'langchain-openai'),
        ('streamlit', 'streamlit'),
        ('openai', 'openai'),
        ('dotenv', 'python-dotenv'),
        ('pydantic', 'pydantic'),
    ]
    
    failed = []
    for module, package in required_packages:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!\n")
    return True


def test_config():
    """Test configuration and environment setup."""
    print("🔍 Testing configuration...")
    
    try:
        from utils.config import Config
        
        # Check if .env exists
        if not os.path.exists('.env'):
            print("  ⚠️  .env file not found")
            print("     Run: cp .env.example .env")
            print("     Then add your OpenAI API key\n")
            return False
        
        # Check API key
        if not Config.OPENAI_API_KEY:
            print("  ❌ OPENAI_API_KEY not set in .env")
            return False
        
        if Config.OPENAI_API_KEY == "your_openai_api_key_here":
            print("  ⚠️  OPENAI_API_KEY is still the default value")
            print("     Please update .env with your actual API key\n")
            return False
        
        print(f"  ✅ API Key configured")
        print(f"  ✅ GPT Model: {Config.GPT_MODEL}")
        print(f"  ✅ Whisper Model: {Config.WHISPER_MODEL}")
        print("✅ Configuration valid!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}\n")
        return False


def test_agents():
    """Test that all agents can be instantiated."""
    print("🔍 Testing agents...")
    
    try:
        from agents import (
            CallIntakeAgent,
            TranscriptionAgent,
            SummarizationAgent,
            QualityScoringAgent,
            CallCenterWorkflow
        )
        
        # Try to instantiate each agent
        agents = [
            ('Call Intake Agent', CallIntakeAgent),
            ('Transcription Agent', TranscriptionAgent),
            ('Summarization Agent', SummarizationAgent),
            ('Quality Scoring Agent', QualityScoringAgent),
            ('Workflow', CallCenterWorkflow),
        ]
        
        for name, agent_class in agents:
            try:
                agent = agent_class()
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                return False
        
        print("✅ All agents instantiated successfully!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Agent import/instantiation error: {e}\n")
        return False


def test_sample_data():
    """Test that sample data exists."""
    print("🔍 Testing sample data...")
    
    sample_files = [
        'sample_data/sample_call_transcript.txt',
        'sample_data/sample_call_poor_quality.txt',
    ]
    
    all_exist = True
    for file in sample_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} not found")
            all_exist = False
    
    if all_exist:
        print("✅ Sample data available!\n")
    else:
        print("⚠️  Some sample files missing\n")
    
    return all_exist


def test_workflow():
    """Test a simple workflow with sample data."""
    print("🔍 Testing workflow with sample data...")
    
    try:
        from agents.workflow import CallCenterWorkflow
        
        # Check if sample data exists
        sample_file = 'sample_data/sample_call_transcript.txt'
        if not os.path.exists(sample_file):
            print("  ⚠️  Sample file not found, skipping workflow test\n")
            return True
        
        # Read sample data
        with open(sample_file, 'r') as f:
            sample_text = f.read()
        
        # Create workflow
        workflow = CallCenterWorkflow()
        
        print("  🔄 Running workflow on sample data...")
        result = workflow.process("text", sample_text)
        
        # Check results
        if result.get('error'):
            print(f"  ❌ Workflow error: {result['error']}")
            return False
        
        if result.get('call_data'):
            print("  ✅ Call data extracted")
        
        if result.get('summary'):
            print("  ✅ Summary generated")
        
        if result.get('quality_score'):
            print("  ✅ Quality score calculated")
        
        print("✅ Workflow test successful!\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  Call Center AI Assistant - Setup Verification")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Agents", test_agents()))
    results.append(("Sample Data", test_sample_data()))
    results.append(("Workflow", test_workflow()))
    
    # Summary
    print("="*60)
    print("  Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print("="*60 + "\n")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("🎉 All tests passed! Your system is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Upload a call transcript or audio file")
        print("   3. View the analysis results\n")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("   Check SETUP_GUIDE.md for troubleshooting help.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
