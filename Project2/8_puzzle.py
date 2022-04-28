import numpy as np

class map():
    rows = 3
    columns = 3
    res = [[0,0],[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[1,1]]

class point():
    def __init__(self,points):
        self.points = points
        self.F = 0
        self.G = 0
        self.parent = None

class A_star():
    def __init__(self,start,target):
        self.count = 1
        self.start = start.points
        self.target = target
        self.open = []
        self.closed = []
        self.record = []
        self.open.append(start)
        self.record.append(start.points)
        start.F = self.cal_F(start)

    def cal_H(self,node):
        H = 0
        for x in range(len(node.points)):
            for y in range(len(self.start[x])):
                if node.points[x][y] != self.target[x][y]:
                    # H += 1
                    H += abs(map.res[node.points[x][y] - 1][0] - x) + abs(map.res[node.points[x][y] - 1][1] - y)

        return H

    def cal_F(self,node):
        F = self.cal_H(node) + node.G

        return F

    def search_F_min(self):
        if len(self.open) <= 0:
            return False
        next = self.open[0]

        for node in self.open:
            if node.F < next.F:
                next = node
            if node.F == next.F:
                if node.G < next.G:
                    next = node

        return next

    def search_neighbour(self,node):
        neighbour = []
        x = np.where(np.array(node.points) == 0)[0][0]
        y = np.where(np.array(node.points) == 0)[1][0]

        all_possible = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        if x == 0:
            all_possible.remove((x-1,y))
        if x == map.rows - 1:
            all_possible.remove((x+1,y))
        if y == 0:
            all_possible.remove((x,y-1))
        if y == map.columns - 1:
            all_possible.remove((x,y+1))

        for i in all_possible:
            tmp = []
            for j in range(len(node.points)):
                tmp.append(node.points[j][:])
            tmp[x][y] = tmp[i[0]][i[1]]
            tmp[i[0]][i[1]] = 0
            if tmp not in self.record:
                self.record.append(tmp)
                nei = point(tmp)
                neighbour.append(nei)

        return neighbour

    def chose_open(self,neighbour,node):
        for subnode in neighbour:
            if subnode not in self.open:
                if subnode not in self.closed:
                    subnode.parent = node
                    subnode.G = 1 + node.G
                    self.G = 1 + node.G
                    subnode.F = self.cal_F(subnode)
                    # print("address:{},points:{},G:{},F:{}".format(id(subnode),(subnode.points),subnode.G,subnode.F))
                    self.open.append(subnode)

    def search(self):
        res = float('inf')
        while len(self.open) != 0:
            node = self.search_F_min()
            if node.points == self.target:
                print("{},status:{},F:{},G:{}".format(self.count, node.points,node.F,node.G))
                # print(self.record)
                return node.G
            else:
                self.open.remove(node)
                self.closed.append(node)
                neighbour = self.search_neighbour(node)
                self.chose_open(neighbour,node)
                print("{},status:{},F:{},G:{}".format(self.count,node.points,node.F,node.G))
                self.count += 1
        return node.G

if __name__=="__main__":
    # start = [[2, 8, 3], [1, 0, 4], [7, 6, 5]]
    target = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

    input = input()

    start = [[int(input[0]),int(input[1]),int(input[2])],[int(input[3]),int(input[4]),int(input[5])],
             [int(input[6]),int(input[7]),int(input[8])]]

    s = point(start)
    y = []

    test = A_star(s, target)
    print("res:{}".format(test.search()))