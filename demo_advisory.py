#!/usr/bin/env python3
"""
Demo script for CodeContext AI™ Advisory System
Shows functionality without requiring trained models or ML dependencies.
"""

import sys
from pathlib import Path
import tempfile
from unittest.mock import Mock

# Mock the inference engine to demonstrate advisory system
class MockInferenceEngine:
    """Mock inference engine for demonstration."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Mock: Loading advisory model from {model_path}")
    
    def generate(self, prompt, **kwargs):
        """Generate mock advisory recommendations."""
        
        # Analyze prompt to return appropriate recommendations
        if "refactor" in prompt.lower():
            return """1. Function complexity exceeds recommended threshold
→ Break down nested conditional logic into separate validation functions  
→ Improves code readability and reduces cognitive load for maintenance

2. Magic numbers detected in calculation logic
→ Extract numeric constants to named variables with descriptive names
→ Enhances code clarity and makes values easier to modify

3. Function length violates single responsibility principle
→ Split large function into focused, single-purpose methods
→ Increases testability and code reusability"""

        elif "performance" in prompt.lower():
            return """1. Nested loop structure creating O(n²) complexity
→ Consider using hash map or set for faster lookups
→ Dramatically improves performance for larger datasets

2. Database queries inside iteration loop
→ Batch queries or use joins to reduce database roundtrips
→ Eliminates N+1 query problem and reduces latency"""

        elif "security" in prompt.lower():
            return """1. User input used without validation or sanitization
→ Implement input validation with type checking and bounds verification
→ Prevents injection attacks and malformed data processing

2. Potential SQL injection vulnerability in query construction
→ Use parameterized queries or prepared statements
→ Eliminates SQL injection attack vectors"""

        elif "architecture" in prompt.lower():
            return """1. High coupling between modules indicated by import count
→ Introduce dependency injection and interface abstractions
→ Reduces tight coupling and improves testability

2. Class violates single responsibility principle with multiple concerns
→ Split class into focused components with clear boundaries
→ Improves maintainability and follows SOLID principles"""

        else:
            return """1. Code structure could benefit from improved organization
→ Consider refactoring for better separation of concerns
→ Enhances maintainability and code clarity"""


# Mock the torch dependency
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()

# Now import the advisory system
sys.path.insert(0, str(Path(__file__).parent))
from codecontext_ai.advisory import AdvisoryEngine, AdvisoryType, AdvisoryReport

# Monkey patch the InferenceEngine
import codecontext_ai.advisory
codecontext_ai.advisory.InferenceEngine = MockInferenceEngine


def demo_advisory_analysis():
    """Demonstrate advisory system with different analysis types."""
    
    print("=" * 60)
    print("CodeContext AI™ Advisory System Demo")
    print("=" * 60)
    
    # Create sample code files for analysis
    sample_codes = {
        "refactor_example.py": '''
def process_user_data(users):
    results = []
    for user in users:
        if user.get('age', 0) > 18:
            if user.get('status') == 'active':
                if user.get('subscription') == 'premium':
                    if user.get('last_login_days', 999) < 30:
                        result = {
                            'id': user['id'],
                            'name': user['name'],
                            'tier': 'premium_active',
                            'score': user['age'] * 1.5 + 100
                        }
                        results.append(result)
    return results
''',
        
        "performance_example.js": '''
function updateUserProfiles(userIds) {
    const profiles = [];
    for (let i = 0; i < userIds.length; i++) {
        for (let j = 0; j < userIds.length; j++) {
            if (i !== j) {
                const profile = database.query(`SELECT * FROM profiles WHERE user_id = ${userIds[i]}`);
                profiles.push(profile);
            }
        }
    }
    return profiles;
}
''',
        
        "security_example.py": '''
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    user = db.execute(query).fetchone()
    
    if user:
        session['user_id'] = user[0]
        return {"status": "success", "user": user}
    return {"status": "failed"}

def process_upload(filename):
    return f"/uploads/{filename}"
''',
        
        "architecture_example.py": '''
import requests
import json
import sqlite3
import smtplib
import logging
from datetime import datetime
from pathlib import Path

class UserManager:
    def __init__(self):
        self.db = sqlite3.connect('users.db')
        self.logger = logging.getLogger()
        
    def create_user(self, data):
        # Validate and create user
        user_id = self.db.execute("INSERT INTO users (...) VALUES (...)")
        
        # Send welcome email
        smtp = smtplib.SMTP('localhost')
        smtp.send_message(...)
        
        # Log activity
        self.logger.info(f'User created: {user_id}')
        
        # Update analytics
        requests.post('https://analytics.com/track', json={...})
        
        return user_id
'''
    }
    
    # Initialize advisory engine with mock model
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as mock_model:
        mock_model_path = mock_model.name
    
    engine = AdvisoryEngine(mock_model_path)
    
    # Demonstrate different advisory types
    advisory_types = [
        (AdvisoryType.REFACTOR, "refactor_example.py"),
        (AdvisoryType.PERFORMANCE, "performance_example.js"), 
        (AdvisoryType.SECURITY, "security_example.py"),
        (AdvisoryType.ARCHITECTURE, "architecture_example.py")
    ]
    
    for advisory_type, filename in advisory_types:
        print(f"\n🔍 {advisory_type.value.upper()} ANALYSIS")
        print("-" * 50)
        
        # Create temporary file with sample code
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as temp_file:
            temp_file.write(sample_codes[filename])
            temp_file.flush()
            
            try:
                # Analyze the file
                report = engine.analyze_file(temp_file.name, advisory_type)
                
                print(f"📁 File: {Path(temp_file.name).name}")
                print(f"💻 Language: {report.language}")
                print(f"📋 Summary: {report.summary}")
                print()
                
                # Display recommendations
                for i, rec in enumerate(report.recommendations, 1):
                    priority_emoji = "🔴" if rec.priority <= 2 else "🟡" if rec.priority <= 3 else "🟢"
                    complexity_emoji = "🔥" if rec.complexity == "high" else "⚡" if rec.complexity == "medium" else "✨"
                    
                    print(f"{priority_emoji} {i}. {rec.category.upper()} - Priority {rec.priority}")
                    print(f"   📍 Location: {rec.location}")
                    print(f"   ❗ Issue: {rec.issue}")
                    print(f"   💡 Solution: {rec.solution}")
                    print(f"   🎯 Impact: {rec.impact}")
                    print(f"   {complexity_emoji} Complexity: {rec.complexity}")
                    print()
                
                # Show next steps
                if report.next_steps:
                    print("🚀 Recommended Next Steps:")
                    for step in report.next_steps:
                        print(f"   → {step}")
                
            finally:
                # Clean up temporary file
                Path(temp_file.name).unlink()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\n📚 To use with real models:")
    print("1. Train advisory model: make train MODEL=advisory")
    print("2. Run analysis: python -m codecontext_ai.guidance_cli analyze myfile.py")
    print("3. Scan directory: python -m codecontext_ai.guidance_cli scan ./src --type security")


def demo_cli_usage():
    """Show CLI usage examples."""
    
    print("\n" + "=" * 60)
    print("CLI Usage Examples")
    print("=" * 60)
    
    cli_examples = [
        ("Single File Analysis", "codecontext-guidance analyze auth.py --type security --format detailed"),
        ("Directory Scan", "codecontext-guidance scan ./src --pattern '*.py' --type refactor --threshold 2"),
        ("Quick Analysis", "codecontext-guidance quick payment.py"),
        ("Batch Analysis", "codecontext-guidance batch analysis_config.json"),
        ("JSON Output", "codecontext-guidance analyze api.js --type performance --format json > results.json")
    ]
    
    for title, command in cli_examples:
        print(f"\n📝 {title}:")
        print(f"   {command}")
    
    print(f"\n📄 Batch Configuration Example (analysis_config.json):")
    print("""{
  "model": "./models/codecontext-advisory-qwen3-8b.gguf",
  "analyses": [
    {"file": "src/auth.py", "type": "security"},
    {"file": "src/payment.py", "type": "performance"},
    {"file": "src/models.py", "type": "architecture"},
    {"file": "src/utils.py", "type": "refactor"}
  ]
}""")


if __name__ == '__main__':
    demo_advisory_analysis()
    demo_cli_usage()