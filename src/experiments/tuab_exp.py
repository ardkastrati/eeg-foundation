import json

with open("tuab_paths", 'r') as file:
    tuab = json.load(file)

with open("tueg_json", 'r') as file:
    tueg = json.load(file)

for edf in tueg:
    path = edf['path']
    slash_index = path.rfind('/')
    path = path[slash_index+1:]
    if path in tuab:
        print (edf['duration'])


