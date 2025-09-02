# test_integration.py
from EarningsStats import run_professional_analysis

# Replace with your actual CSV filename
CSV_FILE = "earnings_data.csv"  # Change this to your actual file name

try:
    print("Starting integration test...")
    results = run_professional_analysis(CSV_FILE, mode='daily')
    
    if 'trading_signals' in results:
        print("✓ Integration successful!")
        print(f"Found {len(results['trading_signals'])} trading signals")
        
        if results['trading_signals']:
            signal = results['trading_signals'][0]
            print(f"Best strategy: {signal['strategy_id']}")
            print(f"Buy time: {signal['timing']['optimal_buy_time']}")
            print(f"Expected return: {signal['expected_performance']['annual_return_percent']}%")
        
        # Save results to file
        import json
        with open('integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Results saved to integration_test_results.json")
        
    else:
        print("✗ Integration failed")
        print(f"Error: {results.get('error', 'Unknown error')}")
        
except Exception as e:
    print(f"✗ Test failed with exception: {e}")
    import traceback
    traceback.print_exc()