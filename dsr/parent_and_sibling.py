import numpy as np


#array_of_sequence is the sequence of operator and operand befor the i-th location : [0:i-1]
def parent_sibling(array_of_sequence):
    uniary=["sin","cos","tan"] #just for the example
    binary=["+","-","%","/","*"]
    operand=["x1","x2","x3","x4"]
    if array_of_sequence[-1] in uniary:
        parent = array_of_sequence[-1]
        sibling = "0"
    elif  array_of_sequence[-1] in binary:
        parent = array_of_sequence[-1]
        sibling = "0"
    elif array_of_sequence[-1] in operand:
        sum_of_operand = 0
        for i in range(len(array_of_sequence)):
            if array_of_sequence[len(array_of_sequence)-i-1] in operand: #read from backward
                sum_of_operand += 1
            elif array_of_sequence[len(array_of_sequence)-i-1] in binary:
                sum_of_operand -= 1
            if sum_of_operand == 0:
                parent = array_of_sequence[len(array_of_sequence)-i-1]
                sibling= array_of_sequence[len(array_of_sequence)-i]
                break
    return [parent, sibling]


array_of_sequence = ["*","+","*","sin","cos","x1","x2","/","tan","x3","x4"]
print(parent_sibling(array_of_sequence))

    
