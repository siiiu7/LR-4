'''
Лабораторная работа №4
С клавиатуры вводится два числа K и N. Квадратная матрица А(N,N), состоящая из 4-х равных по размерам подматриц, B,C,D,E заполняется случайным образом целыми числами в интервале [-10,10].
Для отладки использовать не случайное заполнение, а целенаправленное. Вид матрицы А:
В	С	
D	Е

Вариант 24
    Формируется матрица F следующим образом: скопировать в нее А и  если в Е количество чисел, больших К в четных столбцах меньше, чем произведение чисел в нечетных строках,
    то поменять местами В и С симметрично, иначе С и В поменять местами несимметрично.
    При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
    то вычисляется выражение: A-1*AT – K * FТ, иначе вычисляется выражение (A +GТ-F-1)*K, где G-нижняя треугольная матрица, полученная из А.
    Выводятся по мере формирования А, F и все матричные операции последовательно.
'''

import numpy as np
from math import prod
from copy import deepcopy

import matplotlib.pyplot as plt

K, N = (int(item) for item in input('Press values K and N: ').split()) #Ввод начальных условий
mid = N//2 # Для удобства расчитываем размерность подматриц

A = np.random.randint (-10, 10, (N, N)) #Генерируем матрицу A случайными числами в диапазоне [-10, 10]
print(f'Матрица A:\n{A}')

c1 = 0
c2 = 1
for row in A[mid:,mid+1::2]: #Вычисляем данные по условию - В Е количество чисел, больших К в четных столбцах
    for item in row:
        if item > K: c1 += 1  
for item in A[mid::2, mid:]: #Вычисляем данные по условию - В Е произведение чисел в нечетных строках
    c2 *= prod(item)    
print(f'\nВ Е количество чисел, больших К в четных столбцах = {c1}')
print(  f'В Е произведение чисел в нечетных строках         = {c2}')

F = deepcopy(A) #создаем матрицу F на основе матрицы A
for i in range(mid): #по условию меняются местами подматрицы B и C, поэтому делаем проход по столбцам
    if c1<c2: #меняем подматрицы симметрично
        F[:mid,i], F[:mid,-1-i] = deepcopy(F[:mid,-1-i]), deepcopy(F[:mid,i]) 
    else: #меняем подматрицы несимметрично
        F[:mid,i], F[:mid, mid+i] = deepcopy(F[:mid, mid+i]), deepcopy(F[:mid,i])
print(f'\nМатрица F:\n{F}')

determinant = np.linalg.det(A) #Вычисляем определитель матрицы A
sum_diag = np.trace(F) #Вычисляем сумму диагональных элементов матрицы F
print(f'\nОпределитель матрицы A                 = {determinant:.2f}')
print(  f'Сумма диагональных элементов матрицы F = {sum_diag}')

np.set_printoptions(precision = 2, suppress=True) #Устанавливаем параметры для вывода матрицы на печать: 2 знака после запятой

if determinant > sum_diag: 
    #A-1*AT – K * FТ
    print(f'Вычисляем выражение: A-1*AT – K * FТ')
    A_inv = np.linalg.inv(A)
    print(f'\nРезультат A-1 :\n{A_inv}')
    A_t = np.transpose(A)
    print(f'\nРезультат At :\n{A_t}')
    res_1 = np.dot(A_inv, A_t)
    print(f'\nРезультат A-1 * At :\n{res_1}')
    F_t = np.transpose(F)
    print(f'\nРезультат Ft :\n{F_t}')
    print(f'\nРезультат K*Ft :\n{K*F_t}')
    print(f'\nРезультат A-1*At - K*Ft :\n{res_1 - K*F_t}')
else:
    #A +GТ-F-1)*K
    print(f'Вычисляем выражение: (A +GТ-F-1)*K')
    G = np.tril(A)
    print(f'\nМатрица G:\n{G}')
    G_t = np.transpose(G)
    print(f'\nРезультат Gt :\n{G_t}')
    F_inv = np.linalg.inv(F)
    print(f'\nРезультат F-1 :\n{F_inv}')
    res_1 = A + G_t - F_inv
    print(f'\nРезультат A+Gt-F-1 :\n{res_1}')
    print(f'\nРезультат (A+Gt-F-1)*K :\n{res_1*K}')

#Для матрицы F выводим три графика
fig = plt.figure()

ax_1 = fig.add_subplot(1, 3, 1, projection='3d')
ax_2 = fig.add_subplot(1, 3, 2)
ax_3 = fig.add_subplot(1, 3, 3, projection='3d')

ax_1.set(title = 'ax_1')
ax_2.set(title = 'ax_2')
ax_3.set(title = 'ax_3')

X = np.arange(0, N, 1)
Y = np.arange(0, N, 1)
X, Y = np.meshgrid(X, Y)

ax_1.scatter(X, Y, F)
ax_2.imshow(F)
ax_3.plot_surface(X, Y, F)

plt.show()
