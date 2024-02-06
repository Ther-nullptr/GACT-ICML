import re

log_text = '''

'''

# use regular expression to extract the numbers
pattern = r"\w+': (\d+\.\d+)"
matches = re.findall(pattern, log_text)

# output the numbers
for match in matches:
    # only 3 decimal places
    print(f"{float(match):.3f}", end=",")
    
print(f"\n{float(max(matches)):.3f}")
