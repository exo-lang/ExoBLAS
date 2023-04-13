import sys 

def LoC(file):
    f = open(file,"r")
    alg_lines = 0
    sched_lines = 0
    alg_start= "ALGORITHM START"
    alg_end= "ALGORITHM END"
    sched_start= "SCHEDULE START"
    sched_end= "SCHEDULE END"
    #status 0 -> starting
    #status 1 -> we are in the alg part
    #status 2 -> we are in the sched part
    status = 0
    for line in f:
        l = line.strip() #we first remove the \n characters
        if l == "": #if it is an empty line, we continue
            continue
        if alg_start in l:
            status = 1
        elif sched_start in l:
            status = 2
        elif alg_end in l or sched_end in l:
            continue
        else: # if l is not a identificator we should count them
            sl = line.split()
            if sl[0] == '#' or (isinstance(sl[0], str) and sl[0][0] == '#'): # comment only line
                continue

            if status == 1:
                alg_lines=alg_lines+1
            elif status == 2:
                sched_lines = sched_lines+1
            else:
                continue

    f.close()
    return alg_lines,sched_lines



if __name__ == '__main__':
    import os, glob
    paths = ['src/level1', 'src/level2', 'src/level3']
    for path in paths:
        for filename in glob.glob(os.path.join(path, '*.py')):
            file = filename
            name = file.split("/")
            alg, sched = LoC(file)
            print("{},{},{}".format(name[-1].split(".")[0],alg,sched))


