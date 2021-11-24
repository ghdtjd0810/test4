import sys

input  = sys.stdin.readline
#쪼매난 변수 입력이지만, 크게는 함수로 볼 수도 있음. 
# 어떤 값을 입력했을때 나오는게 함수이니까. 
INF = int(1e9) 

start = int(input())
graph = [[]for i in range(n+1)]
visited = [False] * (n+1)
distance = [INF] * (n+1)

for _ in range(m):
    a,b,c = map(int,  input().split()
    graph[a].append(b,c)

# 쪼매난 함수 입력. 가장 작은 노드를 찾는것. 반복되는 큰 변수를 만드는거지

def get_smallest_node():
    min_value = INF
    #찾은 가장 작은 값을 반환해줘야 하니까?
    index = 0
    #숨어있는 알고리즘은, 포문돌면서 알아서 최솟값을 찾아감.
    for i in range(1,n+1):
        if distance[i] < min_value and not visited[1]:
            min_value = distance[i]
            index = i
        return index


#목표는 최종 비용함수 리스트를 찾는거기 때문에 결국 distance가 어떻게 변하는지 볼꺼임. 
def dijkstra(start):
    distance[start] = 0
    visited[start] = True

    for i in graph[start]:
        #2,3,4,에 2,5,1이 들어갈꺼니까 
        distance[i[0]] = i[1]
    # 이제 기본세팅은 끝났고, cost가 어떻게 정리될껀지 쫌 알아야댐.


# 작은수 대입, 스토리를 녹여내는것, 치환과 목표, 변수설정과 함수 만들기는 똑같고 크기의 차이일 뿐.
    for i in range(n-1):
        now = get_smllest_node()
        visited = [now]
        for j in graph[now]:
            cost = distance[now] + j[1]
            if cost < distance[j[0]]:
                distance[now] = cost
