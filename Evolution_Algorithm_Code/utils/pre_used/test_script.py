from collections import OrderedDict
import csv
try: 
    import wandb
except ImportError:
    pass


def update_summary(rowd, filename, write_header=False, log_wandb=False):
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header: dw.writeheader()
        dw.writerow(rowd)

# rowd = OrderedDict(epoch=1)
# rowd.update([('dd', 456), ('hh', 3463)])
# update_summary(rowd, './test_summary.csv', write_header=True)

# rowd.update([('epoch', 2), ('dd', 456734), ('hh', 'dfjhdf')])
# update_summary(rowd, './test_summary.csv', write_header=False)

rowd = OrderedDict(emm=1)
rowd.update([('jjjj', 9999), ('kkkk', 6666), ('emm', 555), ('emm', 7777)])
update_summary(rowd, './test_summary.csv', write_header=True)

# rowd.update([('epoch', 2), ('dd', 456734), ('hh', 'dfjhdf')])
# update_summary(rowd, './test_summary.csv', write_header=False)