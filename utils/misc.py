import re
# extract the month numbers
def extract_number(mystring):
    numbers = re.findall("^\d+", mystring)
    return int(numbers[0])