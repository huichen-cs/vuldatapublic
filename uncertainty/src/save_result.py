import json
import os
import sys


def init_result_list(result_fn:str) -> list:
    if os.path.exists(result_fn):
        with open(result_fn, mode='rt', encoding='utf-8') as f:
            result_list = json.load(f)
            if not isinstance(result_list, list):
                raise ValueError('File {} should contain a list'.format(result_fn))
    else:
        result_list = []
    return result_list


def parse_stdin() -> dict:
    dict_list = []
    state = 'read_begin'
    for line in sys.stdin:
        # print('line -> ', line)
        line = line.rstrip()

        if state == 'read_begin':
            # print('dict_list -> ', dict_list, ', state -> ', state)
            if line.startswith('{'):
                state = 'dict_begin'
                dict_list.append(line)
            else:
                continue
            # print(' --- dict_list -> ', dict_list, ', state -> ', state)
        elif state == 'dict_begin':
            # print('dict_list -> ', dict_list, ', state -> ', state)
            if line.startswith('}'):
                state = 'dict_end'
            dict_list.append(line)
            # print(' --- dict_list -> ', dict_list, ', state -> ', state)
            if state == 'dict_end':
                break
        else:
            # print('dict_list -> ', dict_list, ', state -> ', state)
            continue

    if dict_list:
        dict_lines = ' '.join(dict_list)
        # print(dict_lines)
        result_dict = json.loads(dict_lines)
    else:
        return None
    return result_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: save_result result.json')
        sys.exit(1)

    result_list = init_result_list(sys.argv[1])

    stdin_result_list = []
    while True:
        result_dict = parse_stdin()
        if not result_dict:
            break
        stdin_result_list.append(result_dict)

    if not stdin_result_list:
        raise ValueError("ERROR: standard input has no dict result")
    
    result_list.extend(stdin_result_list)
    with open(sys.argv[1], 'wt', encoding='utf-8') as f:
        json.dump(result_list, f, indent=2)



    