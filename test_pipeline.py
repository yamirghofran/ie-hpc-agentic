#!/usr/bin/env python3
"""
Test script for the Agentic ML Pipeline
"""

import os
import pandas as pd
import numpy as np
from agentic_ml_pipeline import AgenticMLPipeline

def create_test_data():
    """Create a small test dataset if titanic3.xls doesn't exist"""
    print("Creating test dataset...")
    
    # Create a small synthetic dataset similar to Titanic
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'name': [f"Person_{i}, Mr." if i % 2 == 0 else f"Person_{i}, Mrs." for i in range(n_samples)],
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'age': np.random.normal(30, 15, n_samples),
        'sibsp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
        'parch': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]),
        'ticket': [f"TICKET_{i//3}" for i in range(n_samples)],  # Some shared tickets
        'fare': np.random.exponential(30, n_samples),
        'cabin': [f"C{i}" if i % 4 == 0 else None for i in range(n_samples)],
        'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'boat': [f"Boat_{i}" if np.random.random() > 0.7 else None for i in range(n_samples)],
        'body': [i if np.random.random() > 0.9 else None for i in range(n_samples)],
        'home.dest': [f"City_{i//10}" if np.random.random() > 0.4 else None for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[df.sample(frac=0.2).index, 'age'] = None
    df.loc[df.sample(frac=0.1).index, 'fare'] = None
    df.loc[df.sample(frac=0.05).index, 'embarked'] = None
    
    # Save as Excel file
    df.to_excel("titanic3.xls", index=False)
    print(f"Created test dataset with {len(df)} samples")
    return df

def main():
    print("🧪 Testing Agentic ML Pipeline")
    print("=" * 40)
    
    # Check if dataset exists, create test data if not
    if not os.path.exists("titanic3.xls"):
        print("⚠️  Dataset 'titanic3.xls' not found!")
        print("Creating a small test dataset...")
        create_test_data()
    
    print("📊 Dataset found! Starting pipeline test...")
    
    try:
        # Create and run the pipeline
        pipeline = AgenticMLPipeline(data_path="titanic3.xls", seed=42)
        
        print("\n🚀 Running agentic ML pipeline...")
        result = pipeline.run_pipeline()
        
        if result.get("error"):
            print(f"❌ Pipeline failed: {result['error']}")
        else:
            print("\n✅ Pipeline test completed successfully!")
            
            if "metrics" in result and result["metrics"]:
                print(f"\n📊 Model Performance:")
                val_metrics = result["metrics"].get("validation", {})
                test_metrics = result["metrics"].get("test", {})
                
                if val_metrics:
                    print(f"   Validation - Accuracy: {val_metrics.get('accuracy', 0):.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
                if test_metrics:
                    print(f"   Test - Accuracy: {test_metrics.get('accuracy', 0):.4f}, AUC: {test_metrics.get('auc', 0):.4f}")
            
            print(f"\n📝 Logs saved to: logs/agent_actions.txt")
            print(f"📝 Total agent actions logged: {len(result.get('logs', []))}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()