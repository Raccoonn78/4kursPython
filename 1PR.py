from cmath import pi, sqrt
import math as M

    ############################################
def digit_check():
    while True:
        try:
            number = int(input('Enter the number:\n'))
            return number
        except ValueError:
            print('Please, enter the number\n')
            continue
print("Введите первую сторону треугольника= ",)
triagle1 = digit_check()

   
print("Введите вторую сторону треугольника= ",)
triagle2 = digit_check()

print("Введите третью сторону треугольника= ",)
triagle3 = digit_check()




p = triagle1+triagle2+triagle3
p2=p/2
triagle= M.sqrt(p2*(p2-triagle1)*(p2-triagle2)*(p2-triagle3))
print("Площадь треугольника = ", triagle)

######################################################################################################################################
print("Введите перывую сторону прямоугольника= ",)
square1= digit_check()
print("Введите вторую сторону прямоугольника= ",)
square2= digit_check()


print('Площадь треугольника= ',square1*square2 )

square=square1*square2

###################################################################################################################################
print("Введите радиус круга= ")
R= digit_check()


Sr= M.pi* R*R
print('Площадь курга= ',Sr )
##########################################################################################################################################
dict_sample = {
  "Треугольник": triagle, 
  "Прямоугольник": square, 
  "Круг": Sr 
} 

print(dict_sample)

################################################################################################################################
print("Введите первое число= ")
numFirst=digit_check()

print("Введите второе число= ")
numSecond=digit_check()



print("Введите нужную вам операцию +,-,/,//,** : ")

action = (input("Введите нужную вам операцию +,-,/,//,** : "))
if action=='+':
    end = numFirst+numSecond
elif action=='-':
    end = numFirst+numSecond
elif action=='/':
    end = numFirst/numSecond
elif action=='//':
    end = numFirst//numSecond
elif action=='**':
    end = numFirst**numSecond


print('Выполнение операции= ',end )

#####(#################################################################################################################
print("Введите первую сторону треугольника: ")
triagle14 = digit_check()

print("Введите вторую сторону треугольника: ")
triagle24 = digit_check()
print("Введите третью сторону треугольника: ")
triagle34 =digit_check() 



p4 = triagle14+triagle24+triagle34
p24=p4/2
triagle4= M.sqrt(p24* (p24-triagle14)*(p24-triagle24)*(p24-triagle34))
print("Площадь треугольника = ", triagle4)
