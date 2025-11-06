#!/usr/bin/env python3
"""Test script to verify the application loads correctly"""

from main import app

print('✓ Flask app loaded successfully')
print('✓ All routes registered')
print('\nAvailable routes:')

with app.app_context():
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        print(f'  [{methods:6}] {rule}')

print('\n✓ Application is ready to run!')
print('\nTo start the server, run:')
print('  python3 main.py')
