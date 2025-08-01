import json
import os
import ast
import re

def restructure_data(data):
    """
    Restructures the data from a single 'output' field into 'description' 
    and an optional 'nutrients' dictionary.
    Also filters the nutrient data.
    """
    # Remove original instruction and input fields
    data.pop('instruction', None)
    data.pop('input', None)

    output_str = data.pop('output', '')
    
    # Extract the dictionary part from the output string
    dict_start = output_str.find('{')
    dict_end = output_str.rfind('}')
    
    if dict_start != -1 and dict_end != -1:
        dict_str = output_str[dict_start:dict_end+1]
        description = output_str[:dict_start].strip()
        
        try:
            nutrients_dict = ast.literal_eval(dict_str)
            
            # Filter out entries where the value is 0 or just a unit
            filtered_dict = {}
            for key, value in nutrients_dict.items():
                # Search for any non-zero digit
                if re.search(r'[1-9]', str(value)):
                    filtered_dict[key] = value
            
            data['description'] = description
            if filtered_dict:
                data['nutrients'] = filtered_dict
            
        except (ValueError, SyntaxError):
            # If parsing fails, treat the whole output as description
            data['description'] = output_str
            
    else:
        # No dictionary found, the whole output is the description
        data['description'] = output_str.strip()
            
    return data

def main():
    input_file = 'combined_train.jsonl'
    output_file = input_file + '.tmp'

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    data = json.loads(line)
                    restructured_data = restructure_data(data)
                    outfile.write(json.dumps(restructured_data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line that is not valid JSON: {line.strip()}")

        # Replace the original file with the cleaned one
        os.replace(output_file, input_file)
        print(f"Successfully restructured {input_file}")

    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == '__main__':
    main()