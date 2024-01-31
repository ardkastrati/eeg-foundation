import json 


normal_paths = "/home/schepasc/tuab_eval_normal_paths_full"
abnormal_paths = "/home/schepasc/tuab_eval_abnormal_paths_full"


with open (normal_paths, 'r') as file: 
    normal = json.load(file)
with open (abnormal_paths, 'r') as file: 
    abnormal = json.load(file)

out = []
for n in normal: 
    
    path = n['path']
    print(path)
    out.append((path, 0))
for abn in abnormal: 
    path = abn['path']
    out.append((path, 1))


with open("tuab_eval_labeled", 'w') as file:
    json.dump(out, file)