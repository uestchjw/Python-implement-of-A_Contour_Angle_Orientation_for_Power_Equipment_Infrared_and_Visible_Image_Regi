
# 这种默认参数，如果不传给他值，那就是默认的
# def hjw(a:int,b=3):
#     c = a+b**2
#     return c
# d = hjw(8)
# print(d)



# import argparse

# parse = argparse.ArgumentParser()
# parse.add_argument('hjw')
# args = parse.parse_args()

# print(args.hjw)

a = int('-')
print(a)



from math import factorial
expr = '(5m+3)^4'
def expand(expr):
    n = int(expr[expr.find('^')+1:])
    var_place = [i for i,x in enumerate(expr) if x>='a' and x<='z'][0]
    var = expr[var_place]
    a = int(expr[1:var_place])
    b = int(expr[var_place+1:expr.find('^')-1])
    
    if n==0:
        return '1'
    elif n==1:
        return expr[1:expr.find('^')-1]
    else:
        if a==0:
            return str(b**n)
        elif b==0:
            return str(a**n)+var+'^'+str(n)
        else:
            res = ''
            for k in range(n,-1,-1):
                coef = int(factorial(n)/factorial(k)/factorial(n-k) * (a**k) * (b**(n-k)))
                if k!=n :                    
                    coef = str(coef) if coef<0 else '+'+str(coef)
                    if k==0:
                        res += coef
                    elif k==1:
                        res += coef+var
                    else:
                        res += coef+var+'^'+str(k)
                else:
                    res += str(coef)+var+'^'+str(k)
            return res


b = expand(expr)
print(b)
# "-8k^3-36k^2-54k-27"
# "625m^4+1500m^3+1350m^2+540m+81"