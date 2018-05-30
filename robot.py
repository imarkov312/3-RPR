import math
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import matplotlib.patches as ps 
import matplotlib.lines as ls
import matplotlib.path as ph

l = 1
k = l / math.sqrt(3)

gamma1 = math.pi * (0.5 + 1 * 2/3) 
gamma2 = math.pi * (0.5 + 2 * 2/3) 
gamma3 = math.pi * (0.5 + 3 * 2/3)

x1 = 0.0
y1 = 0.0
x2 = 3.4
y2 = 0.33
x3 = 2.0
y3 = 3.0

def print_line(x1,y1,x2,y2):	
	plt.plot([x1, x2],[y1, y2])

# inverse problem
def draw(centr_x, centr_y, phi):
	ax = plt.subplot()
	def triangle_vertex(i):
		return centr_x + k * math.cos(math.pi * (0.5 + i * 2 / 3.0) + phi), centr_y + k * math.sin(math.pi * (0.5 + i * 2 / 3.0) + phi)

	def print_triangle():
		vert_x1, vert_y1 = triangle_vertex(1)
		vert_x2, vert_y2 = triangle_vertex(2)	
		vert_x3, vert_y3 = triangle_vertex(3)		
		print_line(vert_x1, vert_y1, vert_x2, vert_y2)
		print_line(vert_x3, vert_y3, vert_x2, vert_y2)
		print_line(vert_x1, vert_y1, vert_x3, vert_y3)
	
	def draw_kinematic_chains(x, y, i):
		vert_x, vert_y = triangle_vertex(i)
		print_line(x, y, vert_x, vert_y)
		print("L", i, "=", ((vert_x - x)**2 + (vert_y - y)**2)**(0.5))
		plt.plot(x, y, 'go')
		
	
	plt.plot(centr_x, centr_y, 'go')
	print_triangle()		
	draw_kinematic_chains(x1, y1, 1)
	draw_kinematic_chains(x2, y2, 2)
	draw_kinematic_chains(x3, y3, 3)
	
	plt.axis('equal')
	plt.show()

def print_solution(results, method):
	print(method)
	for i in range(len(results)):
		print(i+1, " solution: ", "x = ", results[i][0], "y = ", results[i][1], "phi = ", results[i][2])

def normalize(phi):
	if (abs(phi) < 0.0001) or (abs(phi - 2*math.pi) < 0.01):
		phi = 0.0
		return phi
	inc = -2*math.pi
	if phi < 0:
		inc *= -1

	while not(0 <= phi <= 2 * math.pi):
		phi += inc
	if (abs(phi) < 0.0001) or (abs(phi - 2*math.pi) < 0.01):
		phi = 0.0
	return phi

def new_solution(results, res):
	if len(results) == 0:
		results.append(res)
		return True
	else:
		for i in range(len(results)):
			if abs(results[i][0] - res[0]) < 0.01 and abs(results[i][1] - res[1]) < 0.01 and abs(results[i][2] - res[2]) < 0.01:
				return False
		results.append(res)
	return True


# direct problem 
def system_newton(n):
	
	def func_sys(t):
		x = t[0]
		y = t[1]
		phi = t[2]
		return np.array([(x + k * math.cos(gamma1 + phi) - x1)**2 + (y + k * math.sin(gamma1 + phi) - y1)**2 - L1**2,
			   (x + k * math.cos(gamma2 + phi) - x2)**2 + (y + k * math.sin(gamma2 + phi) - y2)**2 - L2**2,
			   (x + k * math.cos(gamma3 + phi) - x3)**2 + (y + k * math.sin(gamma3 + phi) - y3)**2 - L3**2])

	def x_derivative1(x, y, phi):
		return 2*(x + k*math.cos(gamma1+phi) - x1)

	def y_derivative1(x, y, phi):
		return 2*(y + k*math.sin(gamma1+phi) - y1)

	def phi_derivative1(x, y, phi):
		return 2*k*math.cos(gamma1+phi) * (y + k*math.sin(gamma1+phi) - y1) - 2*k*math.sin(gamma1+phi) * (x + k*math.cos(gamma1+phi) - x1)

	def x_derivative2(x, y, phi):
		return 2*(x + k*math.cos(gamma2+phi) - x2)

	def y_derivative2(x, y, phi):
		return 2*(y + k*math.sin(gamma2+phi) - y2)

	def phi_derivative2(x, y, phi):
		return 2*k*math.cos(gamma2+phi) * (y + k*math.sin(gamma2+phi) - y2) - 2*k*math.sin(gamma2+phi) * (x + k*math.cos(gamma2+phi) - x2)

	def x_derivative3(x, y, phi):
		return 2*(x + k*math.cos(gamma3+phi) - x3)

	def y_derivative3(x, y, phi):
		return 2*(y + k*math.sin(gamma3+phi) - y3)

	def phi_derivative3(x, y, phi):
		return 2*k*math.cos(gamma3+phi) * (y + k*math.sin(gamma3+phi) - y3) - 2*k*math.sin(gamma3+phi) * (x + k*math.cos(gamma3+phi) - x3)

	def Jacobian(point):
		x = point[0]
		y = point[1]
		phi = point[2]
		return np.array([[x_derivative1(x, y, phi), y_derivative1(x, y, phi), phi_derivative1(x, y, phi)],
						[x_derivative2(x, y, phi), y_derivative2(x, y, phi), phi_derivative2(x, y, phi)],
						[x_derivative3(x, y, phi), y_derivative3(x, y, phi), phi_derivative3(x, y, phi)]])

	def newton(guess):
		eps = 0.000000001
		ind = 0
		while abs(func_sys(guess)[0])>eps and abs(func_sys(guess)[1])>eps and abs(func_sys(guess)[2])>eps:			
			if ind > 100:
				return[-1, -1, -1]
			try:			
				inv_jac = np.linalg.inv(Jacobian(guess))
			except np.linalg.LinAlgError:
				return [-1, -1, -1]
			guess = guess - np.matmul(inv_jac, func_sys(guess))
			ind += 1
		return guess
	
	stepx = 3.5 / n
	stepy = 3.0 / n
	guess_x = []
	guess_y = []
	for i in range(n):
		guess_x.append(i*stepx)
		guess_y.append(i*stepy)
	results = []
	for i in range(n):
		for j in range(n):
			#print(i, j)
			res = newton(np.array([guess_x[i], guess_y[j], math.pi/4]))
			if res[0] == -1:
				continue
			res[2] = normalize(res[2])
			new_solution(results, res)
	return results
			

L1 = 1.545483614921123
L2 = 1.6740913226063425
L3 = 1.1182620521713125
 
results = system_newton(30)

if(len(results) == 0):
	print("Incorrect values")
else:
	for i in range(len(results)):
		draw(results[i][0], results[i][1], results[i][2])
	print_solution(results, "Newton System")	


#draw(1.5, 1.5, 0.2)
