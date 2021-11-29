import heapq
import sys
input = sys.stdin.readline

INF = int(1e9)

n, m = map(int, input().split())

start = int(input())

graph = [[] for i in range(n+1)]
distance = [INF] * (n+1)

for _  in range(m):
    a, b, c = map(int, input().split())
    graph[a].append((b,c))

# 여기까지는 무난함


# q 에 리스트 넣고, 0과 스타트. (0,1) 을 넣는다. [0]이 비용이고 [1]이 노드.
# 기본적인 push pop은 우선순위 큐를 제거하는 형태임.모든 노드를 돌 필요없이 queue에서 해결하는 형태
def dijkstra(start):
    q = []
    heapq.heappush(q, (0, start))
    distance[start] = 0

    while q: # q가 비어있지 않다면?
        # 가장 최단거리가 짧은 노드에 대한 정보를 꺼낸대. 팝은 없애는 거임. 그러니까 while문에서
        # q 튜플을 자꾸 꺼낸다는거지.
        dist, now = heapq.heappop(q)
        # q 튜플이 각 dist,now에 들어감. 처음같은 경우는 0 과 1
        if distance[now] < dist:
            # 현재 노드가 이미 처리 되었으면 무시한다는 의미.
            continue


#여기 한번 살피고, 코드한번 내혼자 작성하고 그리고 마무리 짓고.

#(0,1)이 들어가게 되고, cost, 는 비용.
        # 그리고 distance[i[0]] 이거는 graph의 b,c가 들어가 있는 형태이니까,
        # 예를들어서 (2,2)가 들어간 형태이면,
        for i in graph[now]: # now는 지금 있는 곳에서 # 
            cost = dist + i[1]
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                # 여기서 q푸쉬가 들어가는 거임. 도착지,(노드) cost이고 
                heapq.heappush(q, (cost, i[0]))
        print(q)
             


dijkstra(start)

for i in range(1, n+1):
    if distance[i] == INF:
        print("INFINITY")

    else:
        print(distance[i])


