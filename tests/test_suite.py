import unittest
import HtmlTestRunner
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.insert(0, project_root)

def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=current_dir, pattern='test*.py')

    runner = HtmlTestRunner.HTMLTestRunner(
        output=os.path.join(project_root, 'reports'), 
        report_title='NumCompute 33-Case Integrated Test Report',
        combine_reports=True,
        report_name="Integrated_Test_Report"
    )

    print(f"Starting tests from: {current_dir}")
    runner.run(suite)

if __name__ == "__main__":
    run_all_tests()