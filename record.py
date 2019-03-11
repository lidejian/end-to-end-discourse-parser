import os
import sys
import csv


# fieldnames = ['first_name', 'last_name']
# [{'first_name': 'Baked', 'last_name': 'Beans'}, {'first_name': 'Lovely', 'last_name': 'Spam'}]
def write_dict_to_csv(fieldnames, contents, to_file):
    with open(to_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contents)


#[{'first_name': 'Baked', 'last_name': 'Beans'}, {'first_name': 'Lovely', 'last_name': 'Spam'}]
def read_dict_from_csv(in_file):
    if not os.path.exists(in_file):
        return []
    with open(in_file) as csvfile:
        return list(csv.DictReader(csvfile))


def get_current_row(configuration):
    row = {}
    for item in configuration.strip().split("\n"):
        item = item.strip()
        key = item.split(":")[0].strip()
        value = ":".join(item.split(":")[1:]).strip()
        row[key] = value

    return row


# evaluation_result = {
#     "f1": 0.0,
#     "p": 0.0,
#     "r": 0.0,
#     "acc": 0.0
# }
# configuration = {
#     "level1_sense": FLAGS.level1_sense,
#     "dataset_type": FLAGS.dataset_type,
#     "model": FLAGS.model,
#     "share_rep_weights": FLAGS.share_rep_weights,
#     "bidirectional": FLAGS.bidirectional,
#
#     "cell_type": FLAGS.cell_type,
#     "hidden_size": FLAGS.hidden_size,
#     "num_layers": FLAGS.num_layers,
#     "cell_attention": FLAGS.cell_attention,
#     "cell_attention_length": FLAGS.cell_attention_length,
#
#     "dropout_keep_prob": FLAGS.dropout_keep_prob,
#     "l2_reg_lambda": FLAGS.l2_reg_lambda,
#     "Optimizer": "AdaOptimizer",
#     "learning_rate": FLAGS.learning_rate,
#
#     "batch_size": FLAGS.batch_size,
#     "num_epochs": FLAGS.num_epochs,
#
#     "w2v_type": "Google News",
# }
# additional_conf = {}

def do_record(fieldnames, configuration, additional_conf, evaluation_result, record_file):

    curr_row = configuration

    curr_row["additional_conf"] = str(additional_conf)
    curr_row.update(evaluation_result)


    previous_rows = read_dict_from_csv(record_file)
    rows = previous_rows + [curr_row]


    # remove same rows
    # rows = list(map(eval, set(map(str, rows))))

    # sort by dev score
    # rows = sorted(rows, key=lambda x: float(x["f1"]), reverse=True)

    write_dict_to_csv(fieldnames, rows, record_file)

    print("-->record in %s" % record_file)




