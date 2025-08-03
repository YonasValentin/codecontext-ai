#!/usr/bin/env python3
"""
Example usage of CodeContext AI™ advisory guidance system.
Demonstrates how to get code improvement recommendations.
"""

from codecontext_ai.advisory import AdvisoryEngine, AdvisoryType
from pathlib import Path


def example_refactor_analysis():
    """Example: Get refactoring recommendations for a Python file."""
    
    # Sample code to analyze
    sample_code = '''
def process_user_data(users):
    results = []
    for user in users:
        if user['age'] > 18:
            if user['status'] == 'active':
                if user['subscription'] == 'premium':
                    if user['last_login'] > 30:
                        result = {
                            'id': user['id'],
                            'name': user['name'],
                            'tier': 'premium_active'
                        }
                        results.append(result)
    return results
'''
    
    # Write sample to temp file
    temp_file = Path("temp_analysis.py")
    temp_file.write_text(sample_code)
    
    try:
        # Initialize advisory engine
        engine = AdvisoryEngine("./models/codecontext-advisory-qwen3-8b.gguf")
        
        # Analyze for refactoring opportunities
        report = engine.analyze_file(str(temp_file), AdvisoryType.REFACTOR)
        
        print("=== Refactoring Analysis ===")
        print(f"File: {report.file_path}")
        print(f"Language: {report.language}")  
        print(f"Summary: {report.summary}")
        print()
        
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec.category.upper()} - Priority {rec.priority}")
            print(f"   Location: {rec.location}")
            print(f"   Issue: {rec.issue}")
            print(f"   Solution: {rec.solution}")
            print(f"   Impact: {rec.impact}")
            print(f"   Complexity: {rec.complexity}")
            print()
        
        print("Next Steps:")
        for step in report.next_steps:
            print(f"→ {step}")
            
    finally:
        temp_file.unlink()


def example_performance_analysis():
    """Example: Performance analysis."""
    
    sample_code = '''
def get_user_orders(user_ids):
    orders = []
    for user_id in user_ids:
        user_orders = db.query("SELECT * FROM orders WHERE user_id = ?", user_id)
        for order in user_orders:
            order_items = db.query("SELECT * FROM order_items WHERE order_id = ?", order.id)
            order['items'] = order_items
            orders.append(order)
    return orders
'''
    
    temp_file = Path("temp_performance.py")
    temp_file.write_text(sample_code)
    
    try:
        engine = AdvisoryEngine("./models/codecontext-advisory-qwen3-8b.gguf")
        report = engine.analyze_file(str(temp_file), AdvisoryType.PERFORMANCE)
        
        print("=== Performance Analysis ===")
        print(f"Summary: {report.summary}")
        print()
        
        for rec in report.recommendations:
            print(f"• {rec.issue}")
            print(f"  → {rec.solution}")
            print(f"  → {rec.impact}")
            print()
            
    finally:
        temp_file.unlink()


def example_security_analysis():
    """Example: Security analysis."""
    
    sample_code = '''
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    user = db.execute(query).fetchone()
    
    if user:
        session['user_id'] = user.id
        return True
    return False

def upload_file(request):
    file = request.files['upload']
    filename = file.filename
    file.save(f"/uploads/{filename}")
    return f"File {filename} uploaded successfully"
'''
    
    temp_file = Path("temp_security.py")
    temp_file.write_text(sample_code)
    
    try:
        engine = AdvisoryEngine("./models/codecontext-advisory-qwen3-8b.gguf")
        report = engine.analyze_file(str(temp_file), AdvisoryType.SECURITY)
        
        print("=== Security Analysis ===")
        print(f"Summary: {report.summary}")
        print()
        
        for rec in report.recommendations:
            print(f"⚠️  {rec.issue}")
            print(f"   → {rec.solution}")
            print(f"   → {rec.impact}")
            print()
            
    finally:
        temp_file.unlink()


def example_architecture_analysis():
    """Example: Architecture analysis."""
    
    sample_code = '''
class UserManager:
    def __init__(self):
        self.db = Database()
        self.email_service = EmailService()
        self.payment_processor = PaymentProcessor()
        self.logger = Logger()
        
    def create_user(self, data):
        # Validate input
        if not data.get('email'):
            raise ValueError('Email required')
        
        # Create user
        user = self.db.create_user(data)
        
        # Send welcome email  
        self.email_service.send_welcome(user.email)
        
        # Process payment
        if data.get('subscription'):
            self.payment_processor.charge(user.id, data['subscription'])
        
        # Log activity
        self.logger.info(f'User created: {user.id}')
        
        return user
        
    def update_user(self, user_id, data):
        user = self.db.get_user(user_id)
        user.update(data)
        self.db.save(user)
        self.logger.info(f'User updated: {user_id}')
        return user
        
    def delete_user(self, user_id):
        user = self.db.get_user(user_id)
        self.db.delete(user)
        self.logger.info(f'User deleted: {user_id}')
'''
    
    temp_file = Path("temp_architecture.py")
    temp_file.write_text(sample_code)
    
    try:
        engine = AdvisoryEngine("./models/codecontext-advisory-qwen3-8b.gguf")
        report = engine.analyze_file(str(temp_file), AdvisoryType.ARCHITECTURE)
        
        print("=== Architecture Analysis ===")
        print(f"Summary: {report.summary}")
        print()
        
        for rec in report.recommendations:
            print(f"🏗️  {rec.issue}")
            print(f"   → {rec.solution}")
            print(f"   → {rec.impact}")
            print()
            
    finally:
        temp_file.unlink()


def example_cli_usage():
    """Example: Using the CLI interface."""
    
    print("=== CLI Usage Examples ===")
    print()
    print("1. Analyze single file:")
    print("   codecontext-guidance analyze myfile.py --type refactor")
    print()
    print("2. Scan directory:")
    print("   codecontext-guidance scan ./src --pattern '*.py' --type performance")
    print()
    print("3. Quick analysis:")
    print("   codecontext-guidance quick myfile.py")
    print()
    print("4. Batch analysis:")
    print("   codecontext-guidance batch analysis_config.json")
    print()
    print("Example batch config (analysis_config.json):")
    print('''{
  "model": "./models/codecontext-advisory-qwen3-8b.gguf",
  "analyses": [
    {"file": "src/auth.py", "type": "security"},
    {"file": "src/payment.py", "type": "performance"},
    {"file": "src/models.py", "type": "architecture"}
  ]
}''')


if __name__ == '__main__':
    print("CodeContext AI™ Advisory System Examples")
    print("=" * 50)
    
    # Check if model exists
    model_path = Path("./models/codecontext-advisory-qwen3-8b.gguf")
    if not model_path.exists():
        print("Model not found. Please train advisory model first:")
        print("make train MODEL=advisory")
        print()
        example_cli_usage()
    else:
        example_refactor_analysis()
        print("\n" + "=" * 50 + "\n")
        
        example_performance_analysis()
        print("\n" + "=" * 50 + "\n")
        
        example_security_analysis()
        print("\n" + "=" * 50 + "\n")
        
        example_architecture_analysis()
        print("\n" + "=" * 50 + "\n")
        
        example_cli_usage()