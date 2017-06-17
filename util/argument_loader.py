def import_arguments(args):
    dataset = args[1]
    if args[2] == "1":
        query_flag = False
    else:
        query_flag = True
    if len(args) <= 3:
        return dataset, query_flag, args[0] , 500001, 500001, 0, 0
    savename = args[3] 
    steps_1 = int(args[4])
    steps_2 = int(args[5])
    steps_3 = int(args[6])
    expand_flag = int(args[7])
    if len(args) == 8 :
        return dataset, query_flag, savename, steps_1, steps_2, steps_3, expand_flag
    lr_ = float(args[8])
    matchname = args[9]
    if len(args) == 10 :
        return dataset, query_flag, savename, steps_1, steps_2, steps_3, expand_flag, lr_, matchname
    return dataset, query_flag, savename, steps_1, steps_2, steps_3, expand_flag , lr_, matchname, int(args[10])
