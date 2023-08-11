import json
import glob

result = None
for fn in glob.glob("*.json"):
    with open(fn, mode="rt") as fd:
        r = json.load(fd)
    if result is None:
        result = {**(r[0])}
    else:
        for k in r[0]:
            if r[0][k] and isinstance(r[0][k], list):
                # print(fn, k)
                result[k].extend(r[0][k])
                # print(len(result[k]))
            elif r[0][k] and isinstance(r[0][k], dict):
                result[k] = r[0][k]
print(json.dumps(result, indent=2))
