class point():
    def __init__(self,height,len):
        self.height = height
        self.road_len = len
        self.G = 0
        self.F = 0
        self.parent = None

class A_star():
    def __init__(self,N,M,K,road,start):
        self.N = N
        self.M = M
        self.K = K
        self.count = 0
        self.road = road
        self.open = []
        self.closed = []
        self.res = []
        self.open.append(start)

    def cal_H(self,node):
        if node.height == 1:
            return 0

        down = float('inf')
        for i in self.road:
            if i[0] == node.height:
                if down >= (i[2] + node.G) / (self.N - i[1]):
                    down = int((i[2] + node.G) / (self.N - i[1]))
        H = (node.height - 1) * down

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
        for path in self.road:
            if(path[0] == node.height):
                nei = point(path[1],path[2])
                neighbour.append(nei)

        return neighbour

    def chose_open(self,neighbour,node):
        for subnode in neighbour:
            if subnode not in self.open:
                if subnode not in self.closed:
                    subnode.parent = node
                    subnode.G = subnode.road_len + node.G
                    subnode.F = self.cal_F(subnode)
                    self.open.append(subnode)


    def search(self):
        while len(self.open) != 0:
            self.count += 1
            node = self.search_F_min()
            if node.height == 1:
                print("{},height:{},F:{},G:{}".format(self.count,node.height,node.F,node.G))
                self.open.remove(node)
                self.closed.append(node)
                self.res.append(node)
            else:
                self.open.remove(node)
                self.closed.append(node)
                neighbour = self.search_neighbour(node)
                self.chose_open(neighbour,node)
                # print("{},height:{},F:{},G:{}".format(self.count,node.height,node.F,node.G))

        self.print()

    def print(self):
        for i in range(self.K):
            if i < len(self.res):
                print(self.res[i].G)
            else:
                print(-1)

if __name__=="__main__":
    N, M, K = map(int,input().split())
    road = []

    for i in range(M):
        tmp =[]
        X, Y, D = map(int,input().split())
        tmp.append(X)
        tmp.append(Y)
        tmp.append(D)
        road.append(tmp)

    start = point(road[0][0],0)
    s = A_star(N,M,K,road,start)
    s.search()
