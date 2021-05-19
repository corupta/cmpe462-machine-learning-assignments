import os
for f in reversed(sorted(os.listdir("metrics"))):
    [filename, ext] = f.split('.')
    chunks = filename.split('_')
    batch_size = chunks[0].split('-')[1]
    step_size = "-".join(chunks[1].split('-')[1:])
    with open(os.path.join("metrics", f), 'r') as fInput:
        lines = fInput.readlines()
        for fold in range(1, 6):
            dat = lines[2 + fold].split('=')
            iterations = int(float(dat[1].split(',')[0].strip()))
            err_train = dat[2].split(',')[0].strip()
            err_test = dat[3].strip()
            print("|{}|{}|{}|{}|{}|{}|".format(batch_size, step_size, fold, err_train, err_test, iterations))

print()
print()
print()

for f in reversed(sorted(os.listdir("metrics"))):
    [filename, ext] = f.split('.')
    chunks = filename.split('_')
    batch_size = chunks[0].split('-')[1]
    step_size = "-".join(chunks[1].split('-')[1:])
    with open(os.path.join("metrics", f), 'r') as fInput:
        lines = fInput.readlines()
        err_train = float(lines[0].split('=')[1].strip())
        err_test = float(lines[1].split('=')[1].strip())
        avg_iteration_count = 0
        for fold in range(1, 6):
            dat = lines[2 + fold].split('=')
            avg_iteration_count += int(float(dat[1].split(',')[0].strip()))
        avg_iteration_count = avg_iteration_count / 5
        print("|{}|{}|{:.3f}|{:.3f}|{}|".format(batch_size, step_size, err_train, err_test, avg_iteration_count))