import re

# Read the current ref_path.cpp file
with open('ref_path.cpp', 'r') as f:
    content = f.read()

# Extract just the numbers (remove the array declaration and braces)
match = re.search(r'float ref_path_array\[4804\]\s*=\s*\{(.*?)\};', content, re.DOTALL)
if not match:
    print("Could not find ref_path_array in file")
    exit(1)

numbers_str = match.group(1)
# Split by comma and clean up whitespace
numbers = [x.strip() for x in numbers_str.split(',') if x.strip()]

print(f"Found {len(numbers)} numbers")

# Reshape into 2D array (1201 rows × 4 columns)
output = "// filepath: /home/tanmay-desai/MPPI_FPGA_DATAFLOW/MPPI_FLOAT_KINEMATIC/MPPI_Float/ref_path.cpp\n"
output += "#include \"ref_path.hpp\"\n\n"
output += "pos_nn ref_path_array[1201][4] = {\n"

for i in range(0, len(numbers), 4):
    row = numbers[i:i+4]
    if len(row) == 4:
        output += "    {" + ", ".join(row) + "},\n"

output += "};\n"

# Write to new file
with open('ref_path_new.cpp', 'w') as f:
    f.write(output)

print("✅ Done! Created ref_path_new.cpp with 2D array format")
print(f"   Array shape: [1201][4]")