def func1(i):
    return i+1

def func2(j):
    return j+j

# global n
# n = 0
# count = 9

def func3(z, n, count):
    # global n
    f1 = func1(z)
    f2 = func2(f1)
    if n == count:
        return f2
    else:
        # n = n+1
        return func3(f2, n+1, count)


print(func3(1,0,9))
    
    

# input1 = 1
# for i in range(10):
#     input1 = func3(input1)
# print(input1)
