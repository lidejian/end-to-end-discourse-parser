import os, time
import config

scorer_path = config.DATA_PATH + "/cqa/scorer"
write_path = config.DATA_PATH + "/cqa/write.%s" % (time.time())


def get_rank_score_by_file(pred_file, id_file, tag="dev", subtask="A"):
    preds = [line.strip().split('\t') for line in open(pred_file)]
    ids = [line.strip().split('\t') for line in open(id_file)]

    with open(write_path, "w") as fw:
        for id, value in zip(ids, preds):
            if value[0] == "2":
                label = "true"
            else:
                label = "false"
            fw.write(id[0] + '\t' + id[1] + '\t' + "0\t" + value[1] + "\t" + label + '\n')

    output = os.popen(
        'python2 ' + scorer_path + '/MAP_scripts/ev.py ' + config.DATA_PATH + '/cqa/_gold/SemEval2016-Task3-CQA-QL-' + tag + \
        '.xml.subtask' + subtask + '.relevancy '+ write_path)

    map_score = 0.0
    mrr_score = 0.0
    for line in output:
        line = line.strip()
        # print line
        if "*** Official score (MAP for SYS):" in line:
            map_score = float(line.split("*** Official score (MAP for SYS):")[1].strip())

        if "MRR" in line:
            mrr_score = float(line.split()[3].strip())
    return float(map_score), float(mrr_score) / 100.0