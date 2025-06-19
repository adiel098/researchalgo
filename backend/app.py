from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import json
import logging
import random
import string
from typing import Dict, List, Set, Tuple, Optional, Any

# Import our simplified Santa Claus algorithm implementation
from .santa_claus_problem import Instance, AllocationBuilder, divide, santa_claus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a custom handler that will store logs
class LogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

    def get_logs(self):
        return self.logs

    def clear_logs(self):
        self.logs = []

# Create the handler and add it to the logger
log_handler = LogHandler()
log_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(log_handler)

# Create Flask app
app = Flask(__name__, static_folder='../static')
# הגדר CORS עם תמיכה מלאה
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Enable CORS for all routes
@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(app.static_folder, 'static/js'), filename)

@app.route('/static/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'static/css'), filename)
# הגדרת נתיב API
@app.route('/api')
def api_index():
    return jsonify({
        'message': 'Santa Claus Algorithm API',
        'endpoints': {
            '/api/run-algorithm': 'POST - Run the Santa Claus algorithm',
            '/api/generate-random': 'GET - Generate random input data'
        }
    })

# הגדרת נתיב כללי שיטפל בכל הבקשות מ-localhost
@app.route('/localhost:<port>/api/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_localhost(port, endpoint):
    # העבר את הבקשה לנתיב המתאים ב-API
    if endpoint == 'run-algorithm' and request.method == 'POST':
        return run_algorithm()
    elif endpoint == 'generate-random' and request.method == 'GET':
        return generate_random_data()
    else:
        return {'error': f'Endpoint not found: {endpoint}'}, 404

# הגדרת נתיבים להגשת קבצים סטטיים
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # אם הנתיב מתחיל ב-api או localhost, אל תנסה להגיש קובץ סטטי
    if path.startswith('api') or path.startswith('localhost'):
        return {'error': 'Not Found'}, 404
        
    # נסה להגיש קובץ ספציפי
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # אחרת הגש את דף הבית
    else:
        return send_from_directory(app.static_folder, 'index.html')

# הנתיבים הישנים הוחלפו בנתיב כללי למעלה

@app.route('/api/run-algorithm', methods=['POST'])
def run_algorithm():
    try:
        # Clear previous logs
        log_handler.clear_logs()
        
        # Get data from request
        data = request.json
        logger.info(f"Received input data: {data}")
        
        # Extract kids and presents from the input
        kids = data.get('kids', [])
        presents = data.get('presents', [])
        valuations = data.get('valuations', {})
        
        if not kids or not presents:
            return jsonify({
                'error': 'Missing required data: kids and presents are required'
            }), 400
        
        # Create an instance of the Santa Claus problem
        instance = create_instance(kids, presents, valuations)
        
        # Create an AllocationBuilder for the algorithm
        alloc_builder = AllocationBuilder(instance)
        
        # Run the algorithm
        logger.info("Running Santa Claus algorithm...")
        allocation = divide(santa_claus, instance)
        
        # Calculate the minimum happiness value
        min_happiness = float('inf')
        for kid, items in allocation.items():
            kid_happiness = sum(instance.agent_item_value(kid, item) for item in items)
            min_happiness = min(min_happiness, kid_happiness)
        
        if min_happiness == float('inf'):
            min_happiness = 0
        
        # Format the results
        result = {
            'optimal_value': min_happiness,
            'allocation': allocation,
            'input': {
                'kids': kids,
                'presents': presents,
                'valuations': valuations
            },
            'logs': log_handler.get_logs()
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error running algorithm: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'logs': log_handler.get_logs()
        }), 500

def create_instance(kids, presents, valuations):
    """
    Create an Instance object for the Santa Claus algorithm.
    
    Args:
        kids: List of kid names
        presents: List of present names
        valuations: Dictionary mapping kid names to their valuations for each present
        
    Returns:
        An Instance object
    """
    # Format the valuations as required by fairpyx
    formatted_valuations = {}
    
    for kid in kids:
        kid_valuations = {}
        for present in presents:
            # Use the valuation if provided, otherwise default to 0
            kid_valuations[present] = valuations.get(kid, {}).get(present, 0)
        formatted_valuations[kid] = kid_valuations
    
    # Create and return the Instance
    return Instance(
        valuations=formatted_valuations,
        agent_capacities={kid: 1 for kid in kids},  # Default capacity of 1 item per kid
        item_capacities={present: 1 for present in presents}  # Default capacity of 1 kid per item
    )

# הנתיבים הישנים הוחלפו בנתיב כללי למעלה

@app.route('/api/generate-random', methods=['GET'])
def generate_random_data():
    try:
        # Get parameters from query string with limits
        num_kids = min(int(request.args.get('num_kids', 3)), 10)  # Limit to 10 kids
        num_presents = min(int(request.args.get('num_presents', 5)), 20)  # Limit to 20 presents
        
        # Generate random kids and presents
        kids = [f"Kid_{i+1}" for i in range(num_kids)]
        presents = [f"Present_{i+1}" for i in range(num_presents)]
        
        # Generate random valuations for the restricted assignment case
        # In this case, each present has a fixed value for all kids who can receive it
        valuations = {}
        present_values = {}
        can_receive = {}
        
        # Assign a fixed value to each present
        for present in presents:
            present_values[present] = random.randint(1, 10)
        
        # For each present, randomly decide which kids can receive it (at least one kid)
        for present in presents:
            # Ensure at least one kid can receive each present
            num_kids_can_receive = random.randint(1, num_kids)
            can_receive[present] = set(random.sample(kids, num_kids_can_receive))
        
        # For each kid, ensure they can receive at least one present
        for kid in kids:
            can_receive_any = False
            for present in presents:
                if kid in can_receive[present]:
                    can_receive_any = True
                    break
            
            # If a kid can't receive any present, randomly assign one
            if not can_receive_any and presents:
                random_present = random.choice(presents)
                can_receive[random_present].add(kid)
        
        # Create valuations based on the restricted assignment
        for kid in kids:
            valuations[kid] = {}
            for present in presents:
                if kid in can_receive[present]:
                    valuations[kid][present] = present_values[present]
                else:
                    valuations[kid][present] = 0
        
        return jsonify({
            'kids': kids,
            'presents': presents,
            'valuations': valuations
        })
    
    except Exception as e:
        logger.error(f"Error generating random data: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
