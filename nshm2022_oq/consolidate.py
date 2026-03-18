import xml.etree.ElementTree as ET
from collections import OrderedDict

def consolidate_logic_tree(input_file, output_file):
    # Register namespace to avoid 'ns0' prefixes
    ET.register_namespace('', "http://openquake.org/xmlns/nrml/0.5")
    tree = ET.parse(input_file)
    root = tree.getroot()
    ns = {'nrml': "http://openquake.org/xmlns/nrml/0.5"}

    for branch_set in root.findall('.//nrml:logicTreeBranchSet', ns):
        # Dictionary to store {model_text: [first_branch_element, total_weight]}
        consolidated = OrderedDict()
        
        branches = branch_set.findall('nrml:logicTreeBranch', ns)
        
        for branch in branches:
            model_node = branch.find('nrml:uncertaintyModel', ns)
            weight_node = branch.find('nrml:uncertaintyWeight', ns)
            
            # Use the model string as the unique key
            model_text = model_node.text.strip()
            weight_val = float(weight_node.text.strip())
            
            if model_text not in consolidated:
                # Keep the first branch instance we find
                consolidated[model_text] = [branch, weight_val]
            else:
                # Add weight to the existing entry and remove this duplicate branch
                consolidated[model_text][1] += weight_val
                branch_set.remove(branch)
        
        # Update the weights in the kept branches
        for model_text, (branch_element, total_weight) in consolidated.items():
            weight_node = branch_element.find('nrml:uncertaintyWeight', ns)
            # Formatting to 10 decimal places to maintain precision
            weight_node.text = f"{total_weight:.10f}"
            
            # Optional: Update branchID to show it is a consolidated branch
            if "Combined" not in branch_element.get('branchID'):
                branch_element.set('branchID', branch_element.get('branchID') + "_combined")

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Success! Consolidated logic tree saved to: {output_file}")

if __name__ == "__main__":
    # Change these filenames as needed
    consolidate_logic_tree('./source_model.xml', './consolidated.xml')
