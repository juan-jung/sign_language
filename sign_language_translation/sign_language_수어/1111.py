A = [1,2,3,4,5]
K =15
sum = 0
idx = set([])
cnt = 0
while (sum < K):
    min = K
    for i in range(len(A)):
        if (i in idx):
            continue
        if (min > A[i]):
            min = A[i]
            idx.add(i)
    sum += min
    cnt += 1
if (sum > K):
    cnt -= 1
print(cnt)