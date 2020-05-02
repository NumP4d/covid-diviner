import predictor

print('Hello World!')
#generate some test values
x=[]
for i in range(10):
    x.append(i)
print(x)

y = predictor.predict_next_value(x)
print('Let me guess! Hmmmm maybe it will be:')
print(y)
