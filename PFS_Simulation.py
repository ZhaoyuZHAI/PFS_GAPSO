# -*- coding: utf-8 -*-

import copy
from math import modf
from operator import itemgetter
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
from sympy import Point, Circle, Line
import time


UAVs_msg = [[-100, -190, 20, 23, 90, 300, [2, 2, 3], 6],
            [150, -50, 0, 25, 100, 400, [2, 0, 1], 5],
            [900, 700, 70, 25, 100, 300, [1, 3, 2], 4],
            [-800, 800, 270, 30, 150, 250, [1, 2, 1], 3],
            [-900, -600, 320, 30, 150, 300, [1, 2, 0], 2],
            [30, 850, -30, 25, 130, 250, [1, 1, 3], 1]
           
            ]

Insted_INF=99999

border = [[-1000, 1000], [-1000, 1000]]
UAV_num = len(UAVs_msg)
print('------------------------------------------------')
print('UAV list:')
print('x, y, phi, velocity, minimum turning radius, detect_scope, resource, priority')
for i in range(UAV_num):
    print(UAVs_msg[i])
print('UAV number:', UAV_num)
print('------------------------------------------------')

Targets_msg = [[-350, -200, [3, 5, 4], 3],
               [-600, 500, [3, 1, 2], 2],
               [0, 100, [0, 0, 1], 1]
               ]
target_num = len(Targets_msg)
print('Target list:')
print('x, y, resource, priority')
for i in range(target_num):
    print(Targets_msg[i])
print('target number:', target_num)
print('------------------------------------------------')

Targets_condition = np.ones(target_num)  

run_time = 1000 
time_interval = 0.1  
deviation = 0.01  

print('Overall simulation time:', run_time)
print('Sample time:', time_interval)
print('Deviation:', deviation)
print('------------------------------------------------')

def sign1(x):
    if x >= 0:
        return 1
    else:
        return -1


def sign2(x):
    if x > 0:
        return 1
    else:
        return -1


def sat(x):
    if x > 1:
        x = 1
    elif x < -1:
        x = -1
    return x


class UAV_Msg:
    def __init__(self, msg):

        self.site = np.array(msg[0:2])  
        self.phi = np.radians(msg[2])  
        self.v = msg[3]  
        self.r_min = msg[4]  
        self.detect_scope = msg[5]  
        self.resource = msg[6]  
        self.priority = msg[7]
        
        self.path = []  
        self.planning_route = [] 
        self.condition = 1
        UAV_num_Init = UAV_num        

    def move(self):
        self.path.append(self.site)
        if self.condition == 1:
            self.site = self.site + self.v * time_interval * np.array([np.cos(self.phi), np.sin(self.phi)])
            self.border_process()
        else:
            self.site = self.planning_route.pop(0)
            if len(self.planning_route) == 0:              
                self.condition = 1

    def search_target(self):       
        detect_target = []
        for i in range(target_num):
            if Targets_condition[i] == 0:
                continue
            target_site = np.array(Targets_msg[i][0:2])
            dis = np.sqrt(sum(np.square(target_site - self.site)))
            if dis <= self.detect_scope:
                detect_target.append(i)
                print('Target', i, 'is found')
                print('------------------------------------------------')
        return detect_target

    def border_process(self):
        site = self.site
        phi = self.phi
        R = self.r_min
        phi = phi % (2 * np.pi)  

        theta_reduced = [0, -np.pi / 2, np.pi, np.pi / 2]
        type = -1
        theta = phi
        l1 = 0
        if site[1] >= border[1][1] and 0 < phi < np.pi:
            type = 0
            l1 = border[0][1] - site[0]
        elif site[0] >= border[0][1] and (0 < phi < np.pi / 2 or 3 * np.pi / 2 < phi <= np.pi * 2):
            type = 1
            l1 = site[1] - border[1][0]
        elif site[1] <= border[1][0] and np.pi <= phi <= np.pi * 2:
            type = 2
            l1 = site[0] - border[0][0]
        elif site[0] <= border[0][0] and np.pi / 2 <= phi <= np.pi * 3 / 2:
            type = 3
            l1 = border[1][1] - site[1]

        if type != -1:
            theta = (theta - theta_reduced[type]) % (2 * np.pi)
            
            c1 = np.array([site[0] + R * np.sin(phi), site[1] - R * np.cos(phi)])
            radian1, direction1 = self.__cal_border_radian(c1, site, phi, theta, R, l1, type)

            c2 = [site[0] - R * np.sin(phi), site[1] + R * np.cos(phi)]
            radian2, direction2 = self.__cal_border_radian(c2, site, phi, theta, R, l1, type)

            msg = [c1, radian1, direction1]
            if radian1 > radian2:
                msg = [c2, radian2, direction2]
            self.border_path_plan(msg[0], msg[1], msg[2])

    def __cal_border_radian(self, c, site, phi, theta, R, l1, type):
        vec1 = site - c
        vec2 = [np.cos(phi), np.sin(phi)]
        direction = np.sign(vec1[0] * vec2[1] - vec1[1] * vec2[0])

        if direction == 1:
            theta = np.pi - theta
            l1 = border[type % 2][1] - border[type % 2][0] - l1
        temp1 = 2 * theta * (sign1(l1 - 2 * R * np.sin(theta)) + 1) / 2
        temp2 = theta + np.pi - np.arcsin(sat(((l1 - R * np.sin(theta)) / R)))
        temp3 = (sign2(2 * R * np.sin(theta) - l1) + 1) / 2
        radian = temp1 + temp2 * temp3
        return radian, direction

    def border_path_plan(self, center, radian, direction):

        R = self.r_min
        len_interval = self.v * time_interval
        radian_interval = len_interval / R
        theta0 = np.arctan2(self.site[1] - center[1], self.site[0] - center[0])
        theta_add = 0
        
        while abs(abs(theta_add) - radian) > deviation and abs(theta_add) < radian:  
            theta_add += direction * radian_interval  
            theta = theta0 + theta_add
            point = [center[0] + R * np.cos(theta), center[1] + R * np.sin(theta)]
            self.planning_route.append(point)

        self.condition = 2  
        self.phi = self.phi + direction * radian  


class GAPSO:
    def __init__(self):
        self.popSize = 10
        self.crossoverRate = 0.8
        self.mutationRate = 0.2
        self.population = []
        self.value = []
        self.fvalue = [] 
        self.rank = [] 
        self.corwed = []
        
        print('GAPSO initialization')
        print('------------------------------------------------')

    def InitPop(self):
        self.population = []
        self.fvalue = []
        self.corwed = []
        self.rank = []

        for i in range(self.popSize):
            gene = np.random.randint(0, 2, self.gene_len)
            fvalue = self.CalFit(gene)
            self.population.append(gene)
            self.fvalue.append(fvalue)
        self.NSGAII()
        self.Corwed()

    def NSGAII(self):
        S = [[] for i in range(self.popSize)]
        n = np.zeros(self.popSize)
        rank = np.zeros(self.popSize)
        F = []
        H = []
        for i in range(self.popSize):
            for j in range(self.popSize):
                result = self.control(self.fvalue[i], self.fvalue[j])
                if result == 1:
                    S[i].append(j)
                elif result == -1:
                    n[i] = n[i] + 1
            if n[i] == 0:
                rank[i] = 1  
                F.append(i)
        p = 1
        while len(F) != 0:
            H = []
            for i in F:
                for j in S[i]:
                    n[j] = n[j] - 1
                    if n[j] == 0:
                        rank[j] = p + 1
                        H.append(j)
            p = p + 1
            F = H
        self.rank = rank

    def Corwed(self):
        self.corwed = np.zeros(self.popSize)
        value_index = [[value.copy(), i] for value, i in zip(self.fvalue, range(self.popSize))]
        for m in range(len(self.fvalue[0])):
            sbi_fvalue = sorted(value_index, key=lambda x: x[0][m])
            self.corwed[sbi_fvalue[0][1]] = Insted_INF
            self.corwed[sbi_fvalue[-1][1]] = Insted_INF

            for i in range(1, self.popSize - 1):
                self.corwed[sbi_fvalue[i][1]] += np.abs(sbi_fvalue[i - 1][0][m] - sbi_fvalue[i + 1][0][m])

    def control(self, fvalue1, fvalue2):
        result = np.alltrue(np.array(fvalue1) > np.array(fvalue2))
        if np.alltrue(np.array(fvalue1) == np.array(fvalue2)):
            return 0
        elif np.alltrue(np.array(fvalue1) <= np.array(fvalue2)):
            return 1
        elif np.alltrue(np.array(fvalue1) >= np.array(fvalue2)):
            return -1

    def CalFit(self, gene):

        resourse = [gene[i] * np.array(UAV_groups[self.candidate[i]].resource) for i in range(self.gene_len)]
        resourse = np.sum(resourse, axis=0)  
        if np.alltrue(resourse >= np.array(Targets_msg[self.target][2])):
            v1 = np.max(np.array(gene) * self.arrivals_time)
            v2 = sum(gene)
            v3 = np.sum(np.array(gene) * self.arrivals_time)
            return [v1, v2, v3]
        else:
            return [Insted_INF, Insted_INF, Insted_INF]

    def cal_GAPSO(self, target, target_candidate, arrivals_time):

        self.target = target
        self.candidate = target_candidate
        self.gene_len = len(target_candidate)
        self.arrivals_time = np.array(arrivals_time)  
        
        iter_num = 20
        self.InitPop()
        for i in range(iter_num):
            self.Breed()

        pareto_list = []  
        for i in range(self.popSize):
            if self.rank[i] == 1:
                flag = 1
                for j in range(len(pareto_list)):
                    if np.alltrue(self.population[i] == pareto_list[j]):
                        flag = 0
                        break
                if flag == 1:
                    pareto_list.append(self.population[i])
        return self.choose_one(pareto_list)

    def choose_one(self, pareto_list):
        priority_value = 1
        value_list = []
        for gene in pareto_list:
            value_list.append([gene, self.CalFit(gene)])
        value_list.sort(key=lambda x: x[1][priority_value])

        best_gene = value_list[0][0]
        coalition = []
        max_time = 0
        for i in range(self.gene_len):
            if best_gene[i] == 1:
                coalition.append(target_candidate[i])
                if max_time < arrivals_time[i]:
                    max_time = arrivals_time[i]
        return coalition, max_time

    def Filter(self):
        candidateindex = np.random.randint(0, self.popSize, 2)
        if self.rank[candidateindex[0]] < self.rank[candidateindex[1]]:
            return copy.deepcopy(self.population[candidateindex[0]])
        elif self.rank[candidateindex[0]] == self.rank[candidateindex[1]] and self.corwed[candidateindex[0]] >= \
                self.corwed[candidateindex[1]]:
            rand_num=np.random.rand()
            if rand_num>0.3:
                return copy.deepcopy(self.population[candidateindex[0]])
            else:
                return copy.deepcopy(self.population[candidateindex[1]])
        else:
            return copy.deepcopy(self.population[candidateindex[1]])

    def Breed(self):
        new_population = []
        for i in range(0, self.popSize, 2):  
            father = self.Filter()
            mather = self.Filter()
            babby1, babby2 = self.Crossover(father, mather)  
            babby1 = self.Mutation(babby1)
            babby2 = self.Mutation(babby2)
            if i < self.popSize:
                new_population.append(babby1)
                self.fvalue[i] = self.CalFit(babby1)
            if i + 1 < self.popSize:
                new_population.append(babby2)
                self.fvalue[i + 1] = self.CalFit(babby2)
        self.population = new_population

        self.NSGAII()
        self.Corwed()

    def Crossover(self, father, mather):
        index = int(np.floor(self.gene_len / 2))
        babby1 = copy.deepcopy(father)
        babby2 = copy.deepcopy(mather)
        if np.random.rand() < self.crossoverRate:
            babby1[index:] = mather[index:]
            babby2[index:] = father[index:]
        return babby1, babby2

    def Mutation(self, people):
        if np.random.rand() < self.mutationRate:
            index = np.random.randint(0, self.gene_len)
            people[index] = 1 - people[index]
        return people


def clash_avoid(group_find_targets):
    UAV_task = []
    group_find_targets = sorted(group_find_targets, key=lambda x: UAV_groups[x[0]].priority, reverse=True)
    for find_msg in group_find_targets:
        UAVi = find_msg[0] 
        find_target = copy.copy(find_msg[1]) 
        for cp in UAV_task:
            if cp[1] in find_target:
                find_target.remove(cp[1])
        if find_target != []:
            find_target = sorted(find_target, key=lambda tar_index: Targets_msg[tar_index][3], reverse=True)
            UAV_task.append([UAVi, find_target[0]])
    return UAV_task


def Arrivals_time(target_index, target_candidate):
    arrivals_time = []
    target = Targets_msg[target_index]
    for UAV_index in target_candidate:
        UAV = UAV_groups[UAV_index]
        arrivals_time.append(Arrival_time(UAV, target, UAV.r_min))
    return arrivals_time


def Arrival_time(UAV, target, R0):
    direction, radian, tangent_site, center = Dubins_msg(UAV, target, R0)
    path_length = R0 * radian + np.sqrt(np.sum((np.array(target[0:2]) - tangent_site) ** 2))
    return path_length / UAV.v


def Tangent_lines(circle_C, point_P):
    R = float(circle_C.radius.evalf())
    circle = [float(circle_C.center.x.evalf()), float(circle_C.center.y.evalf())]
    point = [float(point_P.x.evalf()), float(point_P.y.evalf())]

    circle_point_angle = np.arctan2(point[1] - circle[1], point[0] - circle[0])
    cos = R / np.sqrt(np.sum((np.array(circle) - np.array(point)) ** 2))
    radian_half = np.arccos(cos)

    tangent_angle1 = circle_point_angle + radian_half
    tangent_point1 = Point(circle[0] + R * np.cos(tangent_angle1), circle[1] + R * np.sin(tangent_angle1))

    tangent_angle2 = circle_point_angle - radian_half
    tangent_point2 = Point(circle[0] + R * np.cos(tangent_angle2), circle[1] + R * np.sin(tangent_angle2))

    return [Line(Point(point), Point(tangent_point1)), Line(Point(point), Point(tangent_point2))]


def Dubins_msg(UAV, target, R0):
    v = UAV.v 
    phi0 = UAV.phi
    UAV_p = Point(UAV.site)
    target_p = Point(target[0:2])

    c1 = Point(UAV_p.x + R0 * np.sin(phi0), UAV_p.y - R0 * np.cos(phi0))
    c2 = Point(UAV_p.x - R0 * np.sin(phi0), UAV_p.y + R0 * np.cos(phi0))
    len1 = c1.distance(target_p)
    len2 = c2.distance(target_p)
    center = c1

    if len2 > len1:
        center = c2

    center = Point(round(center.x.evalf(), 4), round(center.y.evalf(), 4))
    circle = Circle(center, R0)
    tangent_lines = Tangent_lines(circle, target_p)
    tangent_line1 = tangent_lines[0]  
    tangent_line1 = Line(tangent_line1.p2, tangent_line1.p1)
    tangent_point1 = tangent_line1.p1
    y = float((target_p.y - tangent_point1.y).evalf())
    x = float((target_p.x - tangent_point1.x).evalf())
    tangent_angle1 = np.arctan2(y, x)

    tangent_line2 = tangent_lines[1]
    tangent_line2 = Line(tangent_line2.p2, tangent_line2.p1)
    tangent_point2 = tangent_line2.p1
    y = float((target_p.y - tangent_point2.y).evalf())
    x = float((target_p.x - tangent_point2.x).evalf())
    tangent_angle2 = np.arctan2(y, x)

    vec1 = [UAV_p.x - center.x, UAV_p.y - center.y]
    vec2 = [np.cos(phi0), np.sin(phi0)]
    direction = np.sign(vec1[0] * vec2[1] - vec1[1] * vec2[0])
    sin1 = float(tangent_point1.distance(UAV_p).evalf()) / (2 * R0)
    angle1 = 2 * np.arcsin(sin1)
    sin2 = float(tangent_point2.distance(UAV_p).evalf()) / (2 * R0)
    angle2 = 2 * np.arcsin(sin2)

    tangent_point = []
    radian = 0

    if abs(modf(abs(direction * angle1 + phi0 - tangent_angle1) / (2 * np.pi))[0] - 0.5) > 0.5 - deviation:
        tangent_point = tangent_point1
        radian = angle1
    elif abs(modf(abs(direction * (2 * np.pi - angle1) + phi0 - tangent_angle1) / (2 * np.pi))[
                 0] - 0.5) > 0.5 - deviation:
        tangent_point = tangent_point1
        radian = 2 * np.pi - angle1
    elif abs(modf(abs(direction * angle2 + phi0 - tangent_angle2) / (2 * np.pi))[0] - 0.5) > 0.5 - deviation:
        tangent_point = tangent_point2
        radian = angle2
    elif abs(modf(abs(direction * (2 * np.pi - angle2) + phi0 - tangent_angle2) / (2 * np.pi))[
                 0] - 0.5) > 0.5 - deviation:
        tangent_point = tangent_point2
        radian = 2 * np.pi - angle2

    return direction, radian, (float(tangent_point.x.evalf()), float(tangent_point.y.evalf())), (
        float(center.x.evalf()), float(center.y.evalf()))

def Path_plan(target_index, coalition, cost_time):
    target = Targets_msg[target_index]
    for UAV_index in coalition:
        UAV = UAV_groups[UAV_index]
        fixtime_R = FixTime_R(UAV, target, cost_time)
        Dubins_path_plan(UAV, target, fixtime_R)

def Dubins_path_plan(UAV, target, R0):
    direction, radian, tangent_site, center = Dubins_msg(UAV, target, R0)
    len_interval = UAV.v * time_interval
    radian_interval = len_interval / R0
    theta0 = np.arctan2(UAV.site[1] - center[1], UAV.site[0] - center[0])
    theta_add = 0
    while abs(abs(theta_add) - radian) > deviation and abs(theta_add) < radian:
        theta_add += direction * radian_interval
        theta = theta0 + theta_add
        point = [center[0] + R0 * np.cos(theta), center[1] + R0 * np.sin(theta)]
        UAV.planning_route.append(point)

    line_angle = np.arctan2(target[1] - tangent_site[1], target[0] - tangent_site[0])

    UAV.phi = line_angle  
    UAV.condition = 2  

    start_site = UAV.planning_route[-1]
    tagent_now_dis = 0
    tagent_target_dis = np.sqrt(np.sum((np.array(target[0:2]) - tangent_site) ** 2))
    while abs(tagent_now_dis - tagent_target_dis) > deviation and tagent_now_dis < tagent_target_dis:
        new_point = np.array(
            [start_site[0] + len_interval * np.cos(line_angle), start_site[1] + len_interval * np.sin(line_angle)])
        UAV.planning_route.append(new_point)
        start_site = new_point
        tagent_now_dis = np.sqrt(np.sum((start_site - tangent_site) ** 2))
    io.savemat(r'./path.mat', {'data': np.array(UAV.planning_route)})


def FixTime_R(UAV, target, cost_time):

    dis = np.sqrt((target[0] - UAV.site[0]) ** 2 + (target[1] - UAV.site[1]) ** 2)

    t_min = Arrival_time(UAV, target, UAV.r_min)  
    R_min = UAV.r_min

    R_max = abs(border[0][0] - border[0][1]) / 2
    t_max = Arrival_time(UAV, target, R_max)  

    t = t_min
    R = R_min

    while abs(t - cost_time) > deviation:
        if t < cost_time:
            t_min = t
            R_min = R
        if t > cost_time:
            t_max = t
            R_max = R
        R = (R_min + R_max) / 2
        t = Arrival_time(UAV, target, R)
    return R


def Form_coalition(target, target_candidate):

    arrivals_time = Arrivals_time(target, target_candidate)
    return ggogo.cal_GAPSO(target, target_candidate, arrivals_time)


def plot_UAV_target():
    uav_site = np.array([i[0:2] for i in UAVs_msg])
    target_site = np.array([i[0:2] for i in Targets_msg])
    plt.scatter(uav_site[:, 0], uav_site[:, 1], s=100, marker='^', color='blue', alpha=0.8, label='UAV')
    plt.scatter(target_site[:, 0], target_site[:, 1], s=100, marker='o', color='red', alpha=0.8, label='Target')
    for i in range(UAV_num):
        plt.annotate(
            '${{A}_{%s}}$' % (i + 1),
            xy=(uav_site[i, 0], uav_site[i, 1]),
            xytext=(0, -10),
            textcoords='offset points',
            ha='center',
            va='top',fontsize=14)
    for i in range(target_num):
        plt.annotate(
            '${{T}_{%s}}$' % (i + 1),
            xy=(target_site[i, 0], target_site[i, 1]),
            xytext=(0, -10),
            textcoords='offset points',
            ha='center',
            va='top',fontsize=14)
    plt.plot([border[0][0], border[0][1]], [border[1][0], border[1][0]], linestyle='dashed',color='#000000')
    plt.plot([border[0][0], border[0][0]], [border[1][0], border[1][1]], linestyle='dashed',color='#000000')
    plt.plot([border[0][0], border[0][1]], [border[1][1], border[1][1]], linestyle='dashed',color='#000000')
    plt.plot([border[0][1], border[0][1]], [border[1][0], border[1][1]], linestyle='dashed',color='#000000')

    plt.xlabel('x/m',fontsize=13)
    plt.ylabel('y/m',fontsize=13)
    plt.legend(loc=9)



def liner_add(target, target_candidate, arrivals_time):
    data = [i for i in zip(target_candidate, arrivals_time)]
    data.sort(key=lambda x: x[1])
    coalition = []
    max_time = 0
    resource = np.zeros(len(Targets_msg[0][2]))
    for i in data:
        resource += UAV_groups[i[0]].resource
        coalition.append(i[0])
        max_time = i[1]
        if np.alltrue(resource >= np.array(Targets_msg[target][2])):
            break
    return coalition, max_time


UAV_groups = []
for msg_i in UAVs_msg:
    UAV_groups.append(UAV_Msg(msg_i))

ggogo = GAPSO()
if __name__ == '__main__':

    for t in np.arange(0, 80, time_interval):
        group_find_targets = [] 
        for UAVi, i in zip(UAV_groups, range(UAV_num)):
            if UAVi.condition == 1:
                find_targets = UAVi.search_target()

                if find_targets != []:
                    group_find_targets.append([i, find_targets])

        UAV_task = clash_avoid(group_find_targets)  

        candidate = []
        for i in range(UAV_num):
            if UAV_groups[i].condition == 1 or UAV_groups[i].condition == 3:
                candidate.append(i)
        for cp in UAV_task:
            if cp[0] in candidate:
                candidate.remove(cp[0]) 

        for cp in UAV_task:
            captain = cp[0]  
            target = cp[1]  
            target_candidate = candidate.copy() 
            target_candidate.insert(0, captain) 

            arrivals_time = Arrivals_time(target, target_candidate)
            start = time.time()
            coalition, cost_time = ggogo.cal_GAPSO(target, target_candidate, arrivals_time)
            end = time.time()
            
            print('GAPSO cost time:', end - start)
            print('------------------------------------------------')
            
            Path_plan(target, coalition, cost_time)
            for i in coalition:
                UAV_groups[i].condition = 2
                if i in candidate:
                    candidate.remove(i)
                for j in range(len(Targets_msg[target][2])):
                    if UAV_groups[i].resource[j] >= Targets_msg[target][2][j]:
                        UAV_groups[i].resource[j] -= Targets_msg[target][2][j]
                        Targets_msg[target][2][j] = 0
                    else:
                        Targets_msg[target][2][j] -= UAV_groups[i].resource[j]
                        UAV_groups[i].resource[j] = 0
            if captain not in coalition:
                candidate.append(captain)
            Targets_condition[target] = 0
        for UAVi in UAV_groups:
            UAVi.move()
    for UAVi in UAV_groups:
        plt.plot(np.array(UAVi.path)[:, 0], np.array(UAVi.path)[:, 1], linewidth=1)
    plot_UAV_target()
    plt.show()
   
