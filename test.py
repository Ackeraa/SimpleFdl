import os

path = os.path.join("cald", "modelp", "p.txt")
with open(path, 'r') as f:
    p, total_budget = list(map(int, f.readlines()[0].split()))
    print(p, total_budget)
