import numpy as np
from matplotlib import pyplot as plt
def distance_pair(pair1, pair2):
    return np.sqrt(np.sum((pair1-pair2)**2))


def recursive(sub_pairs):
    if sub_pairs.shape[0] <= 3:
        a, b, dis = 0, 1, distance_pair(sub_pairs[0], sub_pairs[1])
        for i in range(0, sub_pairs.shape[0]-1):
            for j in range(i+1, sub_pairs.shape[0]):
                if (temp := distance_pair(sub_pairs[i], sub_pairs[j])) < dis:
                    a, b, dis = i, j, temp
        return a, b, dis
    mid = sub_pairs.shape[0]//2
    before_a, before_b, before_dis = recursive(sub_pairs[:mid])
    after_a, after_b, after_dis = recursive(sub_pairs[mid:])
    a, b, dis = (before_a, before_b, before_dis) if before_dis < after_dis else (
        after_a + mid, after_b+mid, after_dis)
    left = mid-1
    right = mid+1
    while sub_pairs[mid, 0]-sub_pairs[left, 0] < dis and left >= 0:
        left -= 1
    while sub_pairs[right, 0]-sub_pairs[mid, 0] < dis and right < sub_pairs.shape[0]-1:
        right += 1
    for i in range(left, mid):
        for j in range(mid, right):
            if (temp := distance_pair(sub_pairs[i], sub_pairs[j])) < dis:
                a, b, dis = i, j, temp
    return a, b, dis


def closest_pairs(pairs):

    pairs = np.array(pairs, dtype=np.float64)
    for i in range(0, pairs.shape[0]-1):
        for j in range(i, pairs.shape[0]):
            if pairs[i, 0] > pairs[j, 0]:
                pairs[[i, j]] = pairs[[j, i]]
    a, b, dis = recursive(pairs)
    print(pairs)
    return pairs[[a, b]], dis


x = np.random.randint(-100, 100, 20)
y = np.random.randint(-100, 100, 20)
plt.scatter(x, y)

pairs = np.hstack((x.reshape(x.size, 1), y.reshape(y.size, 1)))

pair, dis = closest_pairs(pairs)
print(pair)
plt.scatter(pair[:, 0], pair[:, 1])
plt.show()
