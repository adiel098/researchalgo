from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import json
import random
import logging
from logging.handlers import MemoryHandler
from typing import Dict, List, Set, Tuple, Optional, Any

# Import the full algorithm implementation from the local module
from santa_claus_problem import divide, Instance, AllocationBuilder
# Import the new modular algorithm implementation
from santa_claus.core import AllocationBuilder as NewAllocationBuilder
from santa_claus.algorithm import santa_claus_algorithm

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set algorithm specific logger to show more details
algorithm_logger = logging.getLogger('santa_claus_problem')
algorithm_logger.setLevel(logging.DEBUG)

# Set the new modular package loggers
santa_claus_logger = logging.getLogger('santa_claus')
santa_claus_logger.setLevel(logging.DEBUG)

# Configure sub-modules loggers
for module in ['core', 'algorithm', 'lp_solver', 'clustering']:
    submodule_logger = logging.getLogger(f'santa_claus.{module}')
    submodule_logger.setLevel(logging.DEBUG)

# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

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

# Add the handler to the algorithm logger as well to capture all detailed logs
algorithm_logger.addHandler(log_handler)

# Add the handler to the new modular package loggers
santa_claus_logger.addHandler(log_handler)

# Add handler to each submodule logger
for module in ['core', 'algorithm', 'lp_solver', 'clustering']:
    submodule_logger = logging.getLogger(f'santa_claus.{module}')
    submodule_logger.addHandler(log_handler)

# Attach handler to the root logger as well so that *any* log emitted
# anywhere in the process is captured and returned to the frontend.
logging.getLogger().addHandler(log_handler)

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
        
        # First try to use the new algorithm
        try:
            # Create a new instance of the Santa Claus problem with the improved algorithm
            instance = create_instance(kids, presents, valuations)
            
            # Create a temporary allocation builder for the legacy instance
            alloc_builder = AllocationBuilder(instance)
            
            # Create the new allocation builder for the improved algorithm
            new_alloc = NewAllocationBuilder(instance)
            
            # Run the improved algorithm
            logger.info("Running improved Santa Claus algorithm...")
            optimal_target_value = None
            
            # Import the function to find the optimal target value
            from santa_claus.lp_solver import find_optimal_target_value
            
            try:
                # Run the improved algorithm
                santa_claus_algorithm(new_alloc, alpha=3.0)
                
                # Get the optimal target value (T*)
                optimal_target_value = find_optimal_target_value(new_alloc)
                
                # Convert the allocation from the improved algorithm to a dictionary
                allocation = {}
                for agent in new_alloc.instance.agents:
                    allocation[agent] = list(new_alloc.allocation.get(agent, []))
            except Exception as alg_err:
                logger.error(f"Error in improved algorithm: {str(alg_err)}", exc_info=True)
                # Fall back to the legacy algorithm
                logger.info("Falling back to legacy algorithm...")
                allocation = divide(lambda alloc: None, instance)  # Just divide evenly as fallback
                optimal_target_value = 0
        except Exception as setup_err:
            logger.error(f"Error setting up algorithm: {str(setup_err)}", exc_info=True)
            # Fall back to a simple even distribution
            allocation = {kid: [] for kid in kids}
            optimal_target_value = 0
        
        # Ensure every present is allocated (greedy post-processing)
        assigned_items = {i for bundle in allocation.values() for i in bundle}
        unassigned_items = set(presents) - assigned_items
        if unassigned_items:
            logger.info(f"Assigning {len(unassigned_items)} unallocated presents greedily")
        for item in unassigned_items:
            best_kid = max(kids, key=lambda k: valuations.get(k, {}).get(item, 0))
            allocation.setdefault(best_kid, []).append(item)

        # Calculate the actual achieved minimum happiness value
        min_happiness = float('inf')
        for kid, items in allocation.items():
            if items:
                kid_happiness = sum(instance.agent_item_value(kid, item) for item in items)
                min_happiness = min(min_happiness, kid_happiness)
            else:
                min_happiness = 0
        
        if min_happiness == float('inf'):
            min_happiness = 0
        
        # Format the results
        result = {
            'target_value': optimal_target_value,  # The theoretical T* value
            'achieved_value': min_happiness,      # The actual achieved min happiness
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
    kids_list = kids
    presents_list = presents
    valuations_dict = formatted_valuations
    
    # Create agent capacities (unlimited - each kid can get any number of presents)
    agent_capacities = {kid: len(presents_list) for kid in kids_list}
    
    # Create item capacities (each item can be given exactly once)
    item_capacities = {present: 1 for present in presents_list}  # Default capacity of 1 kid per item
    
    # Create and return the Instance
    return Instance(
        agents=kids_list,  # List of kid names
        items=presents_list,  # List of present names
        valuations=valuations_dict,
        agent_capacities=agent_capacities,
        item_capacities=item_capacities
    )

# הנתיבים הישנים הוחלפו בנתיב כללי למעלה

@app.route('/api/generate-random', methods=['GET'])
def generate_random_data():
    try:
        # Get what to randomize (valuations or all)
        randomize_type = request.args.get('randomize_type', 'all')
        
        # אם המשתמש בחר 'all', נבחר מספר רנדומלי של ילדים ומתנות
        if randomize_type == 'all':
            # בחר מספר רנדומלי של ילדים בין 2-5
            num_kids = random.randint(2, 5)
            # בחר מספר רנדומלי של מתנות בין מספר הילדים ל-10
            # ודא שמספר המתנות גדול או שווה למספר הילדים
            num_presents = random.randint(num_kids, 10)
            logger.info(f"Randomly selected {num_kids} kids and {num_presents} presents for 'all' randomization")
        else:
            # קבל פרמטרים מ-query string עם מגבלות
            num_kids = min(int(request.args.get('num_kids', 3)), 10)  # הגבל ל-10 ילדים
            # ודא שמספר המתנות גדול או שווה למספר הילדים
            req_presents = int(request.args.get('num_presents', max(5, num_kids)))
            num_presents = min(max(req_presents, num_kids), 20)  # הגבל ל-20 מתנות וודא מינימום כמו מספר הילדים
        
        # Get current data if provided (for partial randomization)
        current_data = request.args.get('current_data')
        current_kids = []
        current_presents = []
        current_valuations = {}
        
        if current_data:
            try:
                current_data_json = json.loads(current_data)
                current_kids = current_data_json.get('kids', [])
                current_presents = current_data_json.get('presents', [])
                current_valuations = current_data_json.get('valuations', {})
            except:
                logger.warning("Failed to parse current data, using default values")
        
        # Build the base lists starting with whatever the user already had on the screen
        kids = current_kids.copy()
        presents = current_presents.copy()

        # Add extra random kids/presents until the requested target size is reached
        while len(kids) < num_kids:
            new_name = f"Kid_{len(kids)+1}"
            if new_name not in kids:
                kids.append(new_name)
        while len(presents) < num_presents:
            new_name = f"Present_{len(presents)+1}"
            if new_name not in presents:
                presents.append(new_name)
        
        # Initialize valuations if we're keeping current ones
        if randomize_type != 'valuations' and randomize_type != 'all' and current_valuations:
            valuations = current_valuations
            # Make sure all kids and presents are covered in the valuations
            for kid in kids:
                if kid not in valuations:
                    valuations[kid] = {}
                for present in presents:
                    if present not in valuations[kid]:
                        valuations[kid][present] = 0
        else:
            # Generate new random valuations
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
    app.run(debug=True, port=5006)
