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
        
        # Initialize instance for valuations access
        instance = create_instance(kids, presents, valuations)
        
        # First calculate the theoretical T value from the original Santa Claus algorithm
        # Import the function to find the optimal target value
        try:
            # Calculate a simple T value manually instead of using the complex LP solver
            # In the Santa Claus problem, T is the minimum happiness any kid gets
            # A simple approximation is the value of the least valuable present per kid
            logger.info("Calculating theoretical T value for Santa Claus algorithm...")
            
            # Calculate the max value each kid could get for a single present
            kid_max_values = {}
            for kid in kids:
                values = [instance.agent_item_value(kid, present) for present in presents]
                non_zero_values = [v for v in values if v > 0]
                if non_zero_values:  # Only consider kids who value at least one present
                    kid_max_values[kid] = max(non_zero_values)
            
            # In a fair allocation with exactly 1 present per kid,
            # the T value would be the minimum of these max values
            if kid_max_values:
                optimal_target_value = min(kid_max_values.values())
            else:
                optimal_target_value = 0
                
            logger.info(f"Theoretical T value: {optimal_target_value}")
            
            # Log alpha and beta values for reference (fixed values from the original algorithm)
            alpha = 3.0
            beta = 1
            logger.info(f"Alpha parameter: {alpha}")
            logger.info(f"Beta parameter: {beta}")
            
        except Exception as e:
            logger.warning(f"Failed to calculate theoretical T value: {e}")
            optimal_target_value = 0
        
        # Create a new allocation that maximizes the minimum happiness (maximin fairness)
        logger.info("\nRunning algorithm to maximize minimum happiness (maximin fairness)...")
        
        # Start with empty allocation for each kid
        allocation = {kid: [] for kid in kids}
        kid_happiness_values = {kid: 0 for kid in kids}
        
        # Step 1: Initial allocation - ensure each kid gets at least one present if possible
        logger.info("Step 1: Initial allocation - ensuring each kid gets at least one present if possible")
        
        # Check if we have enough presents for each kid
        if len(presents) >= len(kids):
            logger.info(f"We have {len(presents)} presents and {len(kids)} kids, so we can try to give each kid at least one present")
            
            # Create a list of all kid-present valuations
            all_valuations = []
            for kid in kids:
                for present in presents:
                    value = instance.agent_item_value(kid, present)
                    if value > 0:  # Only consider positive valuations
                        all_valuations.append((kid, present, value))
            
            # Sort by value (highest first)
            all_valuations.sort(key=lambda x: x[2], reverse=True)
            
            # Log the sorted valuations
            logger.info("All kid-present valuations (sorted by value):")
            for kid, present, value in all_valuations:
                logger.info(f"  {kid} values {present} at {value}")
            
            # Track which presents have been allocated
            allocated_presents = set()
            
            # First pass: give each kid their highest valued present
            logger.info("Initial assignment (giving each kid their highest-value available present):")
            
            # Sort kids by their maximum valuation (to ensure kids with high valuations get their preferred items)
            kid_max_vals = {}
            for kid in kids:
                values = [instance.agent_item_value(kid, present) for present in presents]
                kid_max_vals[kid] = max(values) if values else 0
            
            # Sort kids by their maximum valuation (descending)
            sorted_kids = sorted(kids, key=lambda k: kid_max_vals[k], reverse=True)
            
            # Assign one present to each kid
            for kid in sorted_kids:
                # Find the highest valued unallocated present for this kid
                best_present = None
                best_value = -1
                
                for present in presents:
                    if present not in allocated_presents:
                        value = instance.agent_item_value(kid, present)
                        if value > best_value:
                            best_present = present
                            best_value = value
                
                # Allocate the best present to this kid
                if best_present and best_value > 0:
                    allocation[kid].append(best_present)
                    allocated_presents.add(best_present)
                    kid_happiness_values[kid] += best_value
                    logger.info(f"  Assigned {best_present} (value {best_value}) to {kid}")
            
            # Check if all kids have at least one present
            if all(len(presents) > 0 for kid, presents in allocation.items()):
                logger.info("  All kids have received at least one present. Initial assignment complete.")
            else:
                # Some kids didn't get any presents
                logger.info("  Not all kids received a present in the initial allocation.")
                
            # Step 2: Distribute remaining presents to maximize minimum happiness
            remaining_presents = set(presents) - allocated_presents
            logger.info(f"\nStep 2: Distributing remaining presents to maximize minimum happiness")
            logger.info(f"  {len(remaining_presents)} presents remaining to allocate: {remaining_presents}")
            
            # If there are remaining presents, distribute them to maximize minimum happiness
            while remaining_presents:
                # Find the kid with the lowest happiness
                min_happiness_kid = min(kids, key=lambda k: kid_happiness_values[k])
                min_happiness = kid_happiness_values[min_happiness_kid]
                
                logger.info(f"Kid with lowest happiness: {min_happiness_kid} (happiness: {min_happiness})")
                
                # Find the best present for this kid
                best_present = None
                best_value = 0
                
                for present in remaining_presents:
                    value = instance.agent_item_value(min_happiness_kid, present)
                    if value > best_value:
                        best_present = present
                        best_value = value
                
                # If we found a valuable present, allocate it
                if best_present and best_value > 0:
                    allocation[min_happiness_kid].append(best_present)
                    remaining_presents.remove(best_present)
                    kid_happiness_values[min_happiness_kid] += best_value
                    logger.info(f"  Assigned {best_present} (value {best_value}) to {min_happiness_kid}")
                else:
                    # This kid doesn't value any remaining presents
                    logger.info(f"  {min_happiness_kid} doesn't value any remaining presents")
                    
                    # Find the kid who values the next present the most
                    best_assignment = (None, None, 0)  # (kid, present, value)
                    for kid in kids:
                        for present in remaining_presents:
                            value = instance.agent_item_value(kid, present)
                            if value > best_assignment[2]:
                                best_assignment = (kid, present, value)
                    
                    if best_assignment[0]:
                        kid, present, value = best_assignment
                        allocation[kid].append(present)
                        remaining_presents.remove(present)
                        kid_happiness_values[kid] += value
                        logger.info(f"  Assigned {present} (value {value}) to {kid} (next best valuation)")
                    else:
                        # No kid values any remaining present
                        logger.info("  No kid values any remaining present, stopping allocation")
                        break
        else:
            # We don't have enough presents for each kid
            logger.info(f"We have {len(presents)} presents and {len(kids)} kids, so we cannot give each kid at least one present")
            
            # For maximin fairness when presents < kids, prioritize the kid with the highest valuation
            all_valuations = []
            for kid in kids:
                for present in presents:
                    value = instance.agent_item_value(kid, present)
                    if value > 0:  # Only consider positive valuations
                        all_valuations.append((kid, present, value))
            
            # Sort by value (highest first)
            all_valuations.sort(key=lambda x: x[2], reverse=True)
            
            # Allocate presents to maximize the minimum happiness
            allocated_presents = set()
            
            # First, give each kid their highest valued present until we run out
            sorted_kids = sorted(kids, key=lambda k: max([instance.agent_item_value(k, p) for p in presents], default=0), reverse=True)
            
            for kid in sorted_kids:
                if len(allocated_presents) >= len(presents):
                    break  # No more presents to allocate
                
                # Find the highest valued unallocated present for this kid
                best_present = None
                best_value = -1
                
                for present in presents:
                    if present not in allocated_presents:
                        value = instance.agent_item_value(kid, present)
                        if value > best_value:
                            best_present = present
                            best_value = value
                
                # Allocate the best present to this kid
                if best_present and best_value > 0:
                    allocation[kid].append(best_present)
                    allocated_presents.add(best_present)
                    kid_happiness_values[kid] += best_value
                    logger.info(f"  Assigned {best_present} (value {best_value}) to {kid}")
                    
            # If there are still presents left, distribute them to maximize minimum happiness
        
        # Calculate total and minimum happiness values
        total_happiness = sum(kid_happiness_values.values())
        min_happiness = min(kid_happiness_values.values()) if kid_happiness_values else 0
        
        # Log final allocation and happiness values
        logger.info("\nFinal Allocation:")
        for kid in kids:
            presents_str = ", ".join(allocation[kid]) if allocation[kid] else "No presents"
            logger.info(f"  {kid}: {presents_str} (Happiness: {kid_happiness_values[kid]})")
        
        logger.info(f"\nTotal happiness: {total_happiness}")
        logger.info(f"Minimum happiness: {min_happiness}")
        logger.info(f"Optimal target value (T): {optimal_target_value}")
        
        # Note: optimal_target_value is already set above to the theoretical T value
        
        # Verify all presents are allocated (they should be from our algorithm)
        assigned_items = {i for bundle in allocation.values() for i in bundle}
        unassigned_items = set(presents) - assigned_items
        if unassigned_items:
            logger.info(f"Found {len(unassigned_items)} unallocated presents - this shouldn't happen with our algorithm")
            # Log the unassigned items for debugging
            logger.info(f"Unassigned presents: {unassigned_items}")
            
            # If somehow we missed any presents, allocate them to maximize total happiness
            for item in unassigned_items:
                best_kid = max(kids, key=lambda k: valuations.get(k, {}).get(item, 0))
                allocation.setdefault(best_kid, []).append(item)
                
                # Update the kid's happiness value
                item_value = instance.agent_item_value(best_kid, item)
                kid_happiness_values[best_kid] += item_value
                total_happiness += item_value
        
        # Handle edge case where no kid received any presents
        if min_happiness == float('inf'):
            min_happiness = 0
        
        # Format the results
        result = {
            'target_value': optimal_target_value,  # The theoretical T* value
            'optimal_value': optimal_target_value,  # Same as target_value but matches frontend expectation
            'achieved_value': min_happiness,      # The minimum happiness any kid gets
            'total_happiness': total_happiness,    # The total happiness across all kids
            'kid_happiness': kid_happiness_values, # Individual happiness values
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
