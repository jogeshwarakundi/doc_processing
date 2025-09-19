#Function to merge several JSON fragments into a single JSON object.
import json

# Sample Inputs:
# {"key1":"value1"}
# {"key2":"value2", "key3":{"subkey1":"subvalue1"}}

# Expected Output:
# {"key1":"value1", "key2":"value2", "key3":{"subkey1":"subvalue1"}}

def merge_json_fragments(fragments):
    merged = {}
    for fragment in fragments:
        try:
            data = json.loads(fragment)
            if isinstance(data, dict):
                merged.update(data)
            else:
                print(f"Warning: Fragment is not a JSON object: {fragment}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON fragment: {fragment}. Error: {e}")
    return json.dumps(merged, indent=2)